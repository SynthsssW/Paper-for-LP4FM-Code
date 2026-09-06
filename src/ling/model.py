"""Model loading — DESIGN.md §A.2.

Loads a **dense** HF causal LM in bf16 on cuda, unquantized, with thinking
disabled and eager attention.

Three things here are load-bearing and easy to get wrong:

1. **No quantization.** §A.2: "for a paper whose entire claim is about
   activation geometry, quantization is an unforced error." The loader refuses
   to build a quantization config and warns if the checkpoint carries one.

2. **``attn_implementation="eager"``.** SDPA dispatches to flash /
   mem-efficient kernels whose reduction order depends on sequence length and
   head tiling, so it is not bitwise reproducible across shapes. Gate G1
   (§A.4, prereg G1) requires *exactly* zero difference; eager attention is the
   price. Do not "optimise" this back to sdpa.

3. **``layer_modules()`` must return the real decoder block list.** §A.11 lists
   hooking the wrong module as a silent-corruption bug. The resolver below
   asserts ``len(layers) == config.num_hidden_layers`` and that the entries are
   ``nn.Module``s, and prints the resolved attribute path so it lands in the run
   log.

Chat templates: we do **not** apply one during extraction. The construct is the
geometry of the *prompt-encoding forward pass* (§A.2), and a chat template
prepends role tokens whose positions differ across the four realizations, which
would break the aligned-position contract of §A.6. ``chat_template_kwargs``
is exposed for the behavioural check (G2) only, and carries
``enable_thinking=False`` for checkpoints that honour it.
"""

from __future__ import annotations

# Import order matters: this sets CUBLAS_WORKSPACE_CONFIG before torch is
# imported. See ling/determinism.py.
from . import determinism as _det

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CONFIG_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "model.yaml",
)

#: Attribute paths tried, in order, when `layers_path` is null in the config.
#: `model.model.layers` is the DESIGN.md §A.2 target ("raw HF forward hooks on
#: model.model.layers[i]"); the rest cover the multimodal wrappers that some
#: Gemma-family instruct checkpoints ship with, where the decoder stack is
#: nested one level deeper.
_LAYERS_PATH_CANDIDATES = (
    "model.layers",
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.model.layers",
    "transformer.h",
    "model.decoder.layers",
)


class ModelLoadError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _parse_flat_yaml(text: str) -> Dict[str, Any]:
    """Minimal flat `key: value` YAML reader.

    Only used when pyyaml is unavailable (e.g. the authoring laptop). It handles
    exactly the shape of configs/model.yaml: comments, blank lines, and
    top-level scalar keys. If you add nesting to the config, install pyyaml.
    """
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or line[:1].isspace() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val in ("null", "~", ""):
            out[key] = None
        elif val in ("true", "True"):
            out[key] = True
        elif val in ("false", "False"):
            out[key] = False
        elif (val[0], val[-1]) in (('"', '"'), ("'", "'")):
            out[key] = val[1:-1]
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read configs/model.yaml (pyyaml if present, flat fallback otherwise)."""
    path = path or CONFIG_PATH_DEFAULT
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001
        cfg = _parse_flat_yaml(text)
    if "model_id" not in cfg or not cfg["model_id"]:
        raise ModelLoadError("{}: `model_id` is missing or empty".format(path))
    cfg.setdefault("checkpoint_verified", False)
    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("attn_implementation", "eager")
    cfg.setdefault("thinking", False)
    cfg.setdefault("layers_path", None)
    cfg.setdefault("seed", 20260906)
    cfg.setdefault("hf_cache_dir", None)
    cfg["_config_path"] = path
    return cfg


def _warn_unverified(cfg: Dict[str, Any]) -> None:
    if cfg.get("checkpoint_verified"):
        return
    banner = "!" * 78
    sys.stderr.write(
        "\n{b}\n"
        "UNVERIFIED CHECKPOINT: {mid}\n"
        "DESIGN.md A.2 requires checkpoint strings to be verified on HuggingFace\n"
        "before pulling; the open-weights frontier churned monthly through 2026.\n"
        "This string was written from the design document, NOT checked against the\n"
        "Hub. Verify it, then set `checkpoint_verified: true` in {cfgp}.\n"
        "{b}\n\n".format(b=banner, mid=cfg["model_id"], cfgp=cfg["_config_path"])
    )


# --------------------------------------------------------------------------
# wrapper
# --------------------------------------------------------------------------
@dataclass
class LoadedModel:
    """A loaded causal LM plus the handles the rest of the pipeline needs.

    Attributes
    ----------
    n_layers:
        Number of decoder blocks. Layer index ``i`` ALWAYS means block ``i``,
        and a captured residual at index ``i`` is the **output** of block ``i``
        (see ling.hooks for the full convention statement).
    """

    model: Any
    tokenizer: Any
    config: Dict[str, Any]
    model_id: str
    layers_path: str
    n_layers: int
    d_model: int
    device: str
    dtype: Any
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    load_info: Dict[str, Any] = field(default_factory=dict)

    def layer_modules(self) -> Any:
        """The ``nn.ModuleList`` of decoder blocks, i.e. ``model.model.layers``.

        Hooks are attached to the **forward output** of ``layer_modules()[i]``.
        """
        return _getattr_path(self.model, self.layers_path)

    def __len__(self) -> int:
        return self.n_layers


def _getattr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _try_getattr_path(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_layers_path(model: Any, expected_n_layers: Optional[int],
                        explicit: Optional[str] = None) -> str:
    """Find the decoder-block ModuleList and verify it is the right one.

    §A.11: hooking the wrong module list produces plausible-looking results, so
    this asserts rather than guesses. Raises with the model's top-level
    attribute names attached if nothing matches.
    """
    import torch.nn as nn

    candidates = [explicit] if explicit else list(_LAYERS_PATH_CANDIDATES)
    tried = []
    for path in candidates:
        mod = _try_getattr_path(model, path)
        tried.append((path, type(mod).__name__ if mod is not None else "missing"))
        if mod is None:
            continue
        if not isinstance(mod, nn.ModuleList):
            continue
        if expected_n_layers is not None and len(mod) != expected_n_layers:
            continue
        if len(mod) == 0:
            continue
        return path

    raise ModelLoadError(
        "could not resolve the decoder-block ModuleList "
        "(expected {n} blocks). Tried: {tried}. Top-level attributes: {attrs}. "
        "Set `layers_path` explicitly in configs/model.yaml — do NOT hook a "
        "module you have not confirmed (DESIGN.md A.11: hooking the wrong "
        "module silently produces plausible results).".format(
            n=expected_n_layers, tried=tried,
            attrs=sorted(k for k in vars(model).get("_modules", {})),
        )
    )


#: Sub-config attributes that can carry their own attention implementation.
#: A multimodal / wrapped checkpoint (several Gemma-family instruct releases are
#: one) builds its text tower from `config.text_config`, and that sub-config can
#: keep an `_attn_implementation` of its own — so the decoder blocks we hook can
#: be running SDPA while the top-level config says "eager".
_ATTN_SUBCONFIGS = ("text_config", "vision_config", "audio_config", "decoder")


def _read_attn_implementation(cfg_obj: Any) -> Optional[str]:
    """Read back the attention implementation a config OBJECT actually holds.

    transformers stores the resolved value on the private
    ``_attn_implementation`` attribute; the public ``attn_implementation``
    property exists on some versions and is a plain attribute on others. We try
    both and return None if neither is present, rather than guessing "eager".
    """
    for attr in ("_attn_implementation", "attn_implementation",
                 "_attn_implementation_internal"):
        val = getattr(cfg_obj, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def read_back_attn_implementations(model: Any) -> Dict[str, Optional[str]]:
    """``{"config": <str|None>, "text_config": ..., ...}`` — what is IN FORCE.

    §A.4 / gate G1 depend on eager attention: SDPA and flash kernels choose a
    reduction order that depends on sequence length and head tiling, so they are
    not bitwise reproducible across shapes. Checking the value we *requested* is
    not the same check — some transformers releases fall back to a different
    implementation with only a warning, and a wrapped checkpoint's sub-config
    can keep its own. This reads the resolved value back off the loaded model.
    """
    out: Dict[str, Optional[str]] = {"config": _read_attn_implementation(model.config)}
    for name in _ATTN_SUBCONFIGS:
        sub = getattr(model.config, name, None)
        if sub is not None and not isinstance(sub, (str, int, float, bool)):
            out[name] = _read_attn_implementation(sub)
    return out


def assert_attn_implementation(model: Any, requested: str) -> Dict[str, Optional[str]]:
    """Raise unless every config in force reports ``requested``. Returns them."""
    got = read_back_attn_implementations(model)
    bad = {k: v for k, v in got.items() if v != requested}
    if bad:
        raise ModelLoadError(
            "attention implementation mismatch: requested {!r} but the loaded "
            "model reports {}. (None means the config exposes no resolved "
            "attention attribute at all, which is equally unverifiable.) "
            "DESIGN.md A.4 / gate G1 require bitwise-identical forward passes, "
            "and SDPA/flash kernels pick a reduction order that depends on "
            "sequence length and head tiling — some transformers releases fall "
            "back with only a warning, and a wrapped/multimodal checkpoint's "
            "text_config can keep its own setting while the top-level config "
            "says eager. Fix the checkpoint/transformers version rather than "
            "loosening this check.".format(requested, bad))
    return got


def _resolve_n_layers(hf_config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(hf_config, attr):
            return int(getattr(hf_config, attr))
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None:
        return _resolve_n_layers(text_cfg)
    raise ModelLoadError("cannot determine num_hidden_layers from the model config")


def _resolve_d_model(hf_config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(hf_config, attr):
            return int(getattr(hf_config, attr))
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None:
        return _resolve_d_model(text_cfg)
    raise ModelLoadError("cannot determine hidden_size from the model config")


def load_model(config_path: Optional[str] = None,
               *,
               model_id: Optional[str] = None,
               device: Optional[str] = None,
               enforce: bool = True,
               strict_determinism: bool = True) -> LoadedModel:
    """Load the model described by configs/model.yaml.

    Parameters
    ----------
    enforce:
        Apply ``enforce_determinism(seed)`` before loading. Default True — the
        seeds must be set before any weight init / dropout state is touched,
        and the cuBLAS check must run before the first CUDA allocation.
    """
    cfg = load_config(config_path)
    if model_id:
        cfg["model_id"] = model_id
        cfg["checkpoint_verified"] = False
    if device:
        cfg["device"] = device
    _warn_unverified(cfg)

    det_info: Dict[str, Any] = {}
    if enforce:
        det_info = _det.enforce_determinism(int(cfg["seed"]), strict=strict_determinism)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(str(cfg["dtype"]).lower())
    if dtype is None:
        raise ModelLoadError("unknown dtype {!r}".format(cfg["dtype"]))
    if dtype is not torch.bfloat16:
        warnings.warn(
            "DESIGN.md A.2 specifies bf16 unquantized; running in {} is a protocol "
            "deviation and must be recorded in prereg/DEVIATIONS.md".format(dtype),
            RuntimeWarning, stacklevel=2,
        )

    device_str = str(cfg["device"])
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError(
            "config asks for {} but torch reports no CUDA device. This code is not "
            "meant to be run on CPU for the real experiment; pass device='cpu' "
            "explicitly if you are smoke-testing.".format(device_str)
        )

    hub_kwargs: Dict[str, Any] = {}
    if cfg.get("hf_cache_dir"):
        hub_kwargs["cache_dir"] = cfg["hf_cache_dir"]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        hub_kwargs["token"] = token

    hf_config = AutoConfig.from_pretrained(cfg["model_id"], **hub_kwargs)

    # §A.2 / §A.3: dense only. An MoE residual stream makes "routing modularity"
    # unanswerable (is it the construct or the literal router?).
    moe_markers = [
        a for a in ("num_experts", "num_local_experts", "n_routed_experts",
                    "moe_intermediate_size", "num_experts_per_tok")
        if getattr(hf_config, a, None) or
        getattr(getattr(hf_config, "text_config", None), a, None)
    ]
    if moe_markers:
        raise ModelLoadError(
            "{} looks like a Mixture-of-Experts checkpoint (config has {}). "
            "DESIGN.md A.3 disqualifies MoE for this experiment: routing is a "
            "literal architectural fact there, so 'routing modularity' measured in "
            "the residual stream is unfalsifiable. Dense only.".format(
                cfg["model_id"], moe_markers)
        )

    if getattr(hf_config, "quantization_config", None) is not None:
        raise ModelLoadError(
            "{} carries a quantization_config. DESIGN.md A.2: quantization is an "
            "unforced error in a paper about activation geometry — a reviewer asks "
            "whether you measured the model or the quantization scheme, and there is "
            "no rebuttal phase.".format(cfg["model_id"])
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True, **hub_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise ModelLoadError(
            "a FAST tokenizer is required: anchor character spans are converted to "
            "token indices via `return_offsets_mapping`, which only fast tokenizers "
            "provide. (configs/corpus_schema.md: spans are NEVER resolved by "
            "re-tokenizing a substring.)"
        )
    # No padding is ever used (§A.4 / §A.11 padding leakage) — batch size is 1.
    # We still set a pad token so that any accidental call to a padding code path
    # fails loudly on the batch-size assertion rather than on a missing pad id.
    tokenizer.padding_side = "right"

    load_kwargs: Dict[str, Any] = dict(
        attn_implementation=str(cfg["attn_implementation"]),
        low_cpu_mem_usage=True,
        **hub_kwargs,
    )
    # transformers renamed `torch_dtype` -> `dtype`; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], dtype=dtype, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], torch_dtype=dtype, **load_kwargs
        )

    # Explicit single-device placement. We deliberately do NOT use device_map /
    # accelerate dispatch: cross-device hooks add their own copies and can change
    # reduction order, and the whole point of G1 is that nothing does.
    model.to(device_str)
    model.eval()
    model.requires_grad_(False)

    # READ BACK the attention implementation actually in force, on the top-level
    # config AND on every sub-config, after .to(device). Checking the value we
    # asked for tells us nothing about what the decoder blocks will run.
    requested_attn = str(cfg["attn_implementation"])
    attn_readback = assert_attn_implementation(model, requested_attn)
    effective_attn = attn_readback["config"]

    n_layers = _resolve_n_layers(model.config)
    d_model = _resolve_d_model(model.config)
    layers_path = resolve_layers_path(model, n_layers, cfg.get("layers_path"))

    chat_template_kwargs = _disable_thinking(model, tokenizer, bool(cfg["thinking"]))

    info = {
        "model_id": cfg["model_id"],
        "checkpoint_verified": bool(cfg["checkpoint_verified"]),
        "dtype": str(dtype),
        "device": device_str,
        # Requested vs READ-BACK. The gate reads the effective one; nothing in
        # the record should be the value we merely asked for.
        "attn_implementation_requested": requested_attn,
        "attn_implementation_effective": effective_attn,
        "attn_implementation_subconfigs": {
            k: v for k, v in attn_readback.items() if k != "config"},
        # kept for backwards compatibility; it is the EFFECTIVE value
        "attn_implementation": effective_attn,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_path": layers_path,
        "n_params": sum(p.numel() for p in model.parameters()),
        "thinking_disabled": not bool(cfg["thinking"]),
        "determinism": det_info,
    }
    sys.stderr.write(
        "[ling.model] {mid}: {n} blocks, d_model={d}, {dt} on {dev}, "
        "attn={attn} (read back; requested {req}), layers at model.{lp}\n".format(
            mid=cfg["model_id"], n=n_layers, d=d_model, dt=dtype, dev=device_str,
            attn=effective_attn, req=requested_attn, lp=layers_path)
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        model_id=cfg["model_id"],
        layers_path=layers_path,
        n_layers=n_layers,
        d_model=d_model,
        device=device_str,
        dtype=dtype,
        chat_template_kwargs=chat_template_kwargs,
        load_info=info,
    )


def _disable_thinking(model: Any, tokenizer: Any, thinking_requested: bool) -> Dict[str, Any]:
    """Turn off reasoning/thinking mode — DESIGN.md §A.2.

    Two mechanisms, because checkpoints differ:

    * Chat-template flag. Qwen-style checkpoints gate thinking on
      ``enable_thinking`` passed to ``apply_chat_template``; Gemma-style ones on
      a ``thinking``/``reasoning_effort`` kwarg. We return the kwargs dict rather
      than applying it, because **extraction never applies a chat template at
      all** (see module docstring) — this is for the G2 behavioural check.
    * Generation config. Force greedy and strip sampling params so the G2 check
      cannot silently sample, and so nothing generates a reasoning preamble.

    Note that for the geometry measurement itself thinking is moot: we run a
    single forward pass over the prompt and never generate a token. That is the
    honest statement for the paper — plus the §A.2 sentence that transfer to the
    reasoning-trace regime is open.
    """
    if thinking_requested:
        warnings.warn(
            "thinking: true — DESIGN.md A.2 requires thinking mode OFF for the "
            "primary run. Record as a deviation.", RuntimeWarning, stacklevel=2,
        )
        return {}

    gen = getattr(model, "generation_config", None)
    if gen is not None:
        gen.do_sample = False
        for attr in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gen, attr):
                setattr(gen, attr, None)
        for attr in ("thinking", "enable_thinking", "reasoning_effort"):
            if hasattr(gen, attr):
                setattr(gen, attr, False if attr != "reasoning_effort" else "none")

    kwargs: Dict[str, Any] = {"enable_thinking": False}
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, str) and "enable_thinking" not in tpl:
        # Passing an unknown kwarg to some templates raises; only send it when
        # the template mentions it.
        kwargs = {}
    return kwargs
