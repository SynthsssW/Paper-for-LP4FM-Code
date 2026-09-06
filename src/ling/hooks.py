"""Residual-stream capture, anchored-position extraction, per-layer streaming.

Covers DESIGN.md §A.1 (never persist raw activations, checkpoint after every
layer), §A.4 (batch size 1, no padding), §A.6 (aligned positions), §A.11 items
1-3 (pre/post layernorm, layer indexing, padding leakage), and the anchor
contract in configs/corpus_schema.md.

===========================================================================
LAYER INDEXING CONVENTION  --  read this before touching anything downstream
===========================================================================
``residuals[i]`` is the **OUTPUT of decoder block i**, i.e. the residual stream
*after* block ``i`` has written into it.  ``i`` ranges over ``0 .. n_layers-1``,
so a captured stack has exactly ``n_layers`` entries.  There is no entry for the
embedding output unless you explicitly ask for one
(``ResidualCapture(..., capture_embeddings=True)``), and it is stored separately
under ``.embeddings``, never spliced into ``residuals`` — precisely so that
"layer 0" cannot come to mean two different things in two different files.

Equivalently: ``residuals[i]`` is the INPUT of block ``i+1``, and
``residuals[n_layers-1]`` is the input of the final norm + unembedding.  If a
downstream table says "layer 12", it means the output of block 12 and the input
of block 13.  §A.11 lists the input-vs-output off-by-one as a silent-corruption
bug; this is the single place the convention is defined.

===========================================================================
WHICH SIDE OF THE LAYERNORM  --  §A.11 item 1
===========================================================================
We take the residual stream **post-block and pre-normalization**: the raw value
returned by ``model.model.layers[i]``, with NO layernorm applied to it.

Why that is the right side, mechanically: these are pre-norm transformers.  A
block computes, schematically,

    h = h + Attn(Norm_1(h))
    h = h + MLP(Norm_2(h))          # Gemma-family variants additionally wrap
                                    # each sublayer output in a post-norm that
                                    # lives INSIDE the residual branch

The normalizations are applied to *copies* fed into the sublayers; the value
carried along the skip connection — the residual stream proper — is never
normalized in place.  The block's return value therefore IS the residual stream.
The only normalization applied to it is ``model.model.norm``, once, after the
last block, immediately before the unembedding (the logit lens applies that norm
at read-out time rather than treating it as part of the stream).

Why it matters: RMSNorm rescales every position to a common radius.  Taking the
post-norm value would project all token states onto (a scaling of) the unit
sphere, destroying exactly the norm-growth-with-depth structure that a spectral
affinity is sensitive to — and it would do so while still producing
plausible-looking clusters and a plausible-looking lambda2.  §A.11 calls this
"the most common silent bug in this class of work", and it is silent in both
directions.

Consequence you must handle downstream, not here: because we take the
un-normalized stream, the vector norms grow substantially with depth (Gemma also
scales the embedding by sqrt(d_model) at the input).  Any affinity built on raw
Euclidean distance will therefore be dominated by depth-dependent scale.  Build
the affinity on cosine similarity, or standardize per layer — and state which,
since lambda2 is not invariant to that choice.

===========================================================================
BATCH SIZE 1, NO PADDING  --  §A.4, §A.11 item 3
===========================================================================
Extraction is batch size 1 with an all-ones attention mask, always.  Asserted in
the hook itself (``hidden.shape[0] == 1``), not merely by convention, because a
padded position produces a garbage activation that clusters as a real state and
nothing downstream can tell the difference.  There is no batched code path in
this module to accidentally reach for.

===========================================================================
RAW ACTIVATIONS ARE NEVER PERSISTED  --  §A.1 rule 1
===========================================================================
Extract -> cluster -> write assignments -> discard.  ``LayerCheckpointWriter``
actively refuses to write anything shaped like a raw activation matrix.  The
optional spill directory of ``AlignedActivationStore`` is scratch only: it must
be on node-local disk, it is deleted on close, and it refuses to sit inside
``results/`` or ``data/``.
"""

from __future__ import annotations

from . import determinism as _det  # noqa: F401  (import-order: sets CUBLAS env var)

import glob
import json
import os
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

#: Sentinel key for "the final token of the prompt" (configs/corpus_schema.md).
FINAL = "FINAL"


class AnchorError(ValueError):
    """Raised when an anchor character span cannot be resolved to a token."""


class ExtractionError(RuntimeError):
    pass


def _require_numpy():
    if np is None:
        raise ExtractionError("numpy is required for activation extraction")
    return np


# ==========================================================================
# 1. Anchor character spans -> token indices
#    configs/corpus_schema.md: "Spans are converted to token indices at
#    extraction time via the tokenizer's offset mapping, NEVER by re-tokenizing
#    a substring."
# ==========================================================================
def anchor_order(anchors: Dict[str, Sequence[int]]) -> List[str]:
    """Deterministic, condition-invariant ordering of anchor keys.

    Quantity names sorted lexicographically, then ``FINAL`` last.

    The ordering key is the *name*, deliberately not the character offset. The
    aligned-position contract of §A.6 requires column ``j`` of the extracted
    matrix to mean the same quantity in all four realizations, and a Python
    rewrite can introduce the same quantities in a different textual order than
    the English prose. Sorting by offset would silently permute the columns
    between conditions and every cross-condition comparison would be garbage.
    """
    if FINAL not in anchors:
        raise AnchorError(
            "anchors must contain the {!r} sentinel (configs/corpus_schema.md: "
            "'FINAL ... is always present'); got keys {}".format(
                FINAL, sorted(anchors)))
    names = sorted(k for k in anchors if k != FINAL)
    return names + [FINAL]


def check_anchor_keys_consistent(realizations: Dict[str, Dict[str, Any]]) -> List[str]:
    """Enforce the corpus invariant across the four realizations of one problem.

    configs/corpus_schema.md: "every quantity key appears in the `anchors` of all
    four realizations." §A.6: only positions present in all four may be used, so
    a lambda2 difference between a 60-token NL prompt and a 28-token Python
    prompt is not confounded with length.

    Returns the shared ordered key list. Raises if any condition is missing a
    key — the caller drops the problem and counts it in the drop rate (G3).
    """
    keysets = {cond: set(r["anchors"]) for cond, r in realizations.items()}
    if not keysets:
        raise AnchorError("no realizations given")
    shared = set.intersection(*keysets.values())
    union = set.union(*keysets.values())
    if shared != union:
        missing = {c: sorted(union - k) for c, k in keysets.items() if k != union}
        raise AnchorError(
            "anchor keys differ across realizations; missing per condition: {}. "
            "configs/corpus_schema.md makes this a DROP, counted in the drop "
            "rate — do not silently intersect.".format(missing))
    return anchor_order(dict.fromkeys(shared, None))


def _spans_overlap(tok: Tuple[int, int], span: Tuple[int, int]) -> bool:
    ts, te = tok
    ss, se = span
    if te <= ts:                       # zero-width token = special token
        return False
    if se <= ss:                       # zero-width anchor: treat as a point
        return ts <= ss < te
    return ts < se and te > ss


def char_span_to_token_index(
    offsets: Sequence[Tuple[int, int]],
    span: Sequence[int],
    *,
    strategy: str = "last",
    name: str = "<anchor>",
    n_tokens: Optional[int] = None,
    allow_trailing_special: bool = False,
) -> int:
    """Map one anchor character span to a single token index.

    Parameters
    ----------
    offsets:
        The tokenizer's ``offset_mapping`` for the full prompt, one
        ``(char_start, char_end)`` per token, in token order. Special tokens
        carry a zero-width span (typically ``(0, 0)``) and are skipped.
    span:
        ``[start, end)`` character offsets, or ``[-1, -1]`` for the ``FINAL``
        sentinel.
    strategy:
        ``"last"`` (default) -> the LAST token overlapping the span.
        ``"first"`` -> the first.

        Judgment call, stated for the record: a quantity such as ``1,250`` or
        ``num_apples`` spans several tokens. In a causal model the hidden state
        at the *last* token of the span is the first state that has attended to
        the whole quantity; the state at the first token has seen only a
        fragment and is largely a function of the preceding context. §A.6 asks
        for "the positions where each quantity is introduced", which is
        satisfied by either, so we take the one that actually encodes the
        quantity. Whichever is used must be used for all four realizations —
        this argument is passed through from a single call site.

    Returns
    -------
    int
        Token index into the prompt's ``input_ids``.
    """
    if strategy not in ("first", "last"):
        raise AnchorError("strategy must be 'first' or 'last', got {!r}".format(strategy))
    n = len(offsets) if n_tokens is None else n_tokens
    if n == 0:
        raise AnchorError("empty tokenization")

    start, end = int(span[0]), int(span[1])

    # --- FINAL sentinel ---------------------------------------------------
    if start < 0 or end < 0:
        idx = n - 1
        ts, te = offsets[idx]
        if te <= ts and not allow_trailing_special:
            raise AnchorError(
                "FINAL resolves to token {} which has a zero-width offset span "
                "{} — the tokenizer appended a trailing special token (EOS / "
                "end-of-turn). The final position must be a real content token, "
                "and a trailing special can differ in presence across the four "
                "realizations, silently breaking the §A.6 alignment. Tokenize "
                "with add_special_tokens=False, or strip the chat template, or "
                "pass allow_trailing_special=True if you have verified it is "
                "present identically in all four conditions.".format(idx, (ts, te)))
        return idx

    matches = [i for i in range(n) if _spans_overlap(tuple(offsets[i]), (start, end))]
    if not matches:
        raise AnchorError(
            "anchor {!r} span [{}, {}) matches no token. Either the span is "
            "stale relative to the text (regenerate the corpus record), or it "
            "covers only whitespace that this tokenizer folds into a "
            "neighbouring token's leading space. Do NOT work around this by "
            "re-tokenizing the substring — that gives a different tokenization "
            "than the in-context one (configs/corpus_schema.md).".format(
                name, start, end))
    return matches[-1] if strategy == "last" else matches[0]


@dataclass
class AnchorResolution:
    """Resolved anchor positions for one realization."""

    names: List[str]                 # ordered, FINAL last
    token_indices: List[int]         # parallel to `names`
    input_ids: List[int]
    offsets: List[Tuple[int, int]]
    n_tokens: int
    strategy: str

    @property
    def n_positions(self) -> int:
        return len(self.names)


def resolve_anchor_positions(
    text: str,
    anchors: Dict[str, Sequence[int]],
    tokenizer: Any,
    *,
    strategy: str = "last",
    add_special_tokens: bool = True,
    allow_trailing_special: bool = False,
    allow_duplicate_positions: bool = True,
) -> AnchorResolution:
    """Tokenize ``text`` once and resolve every anchor span to a token index.

    The prompt is tokenized exactly once, as a whole, with
    ``return_offsets_mapping=True``. No substring is ever tokenized on its own:
    subword tokenizers are context sensitive (leading-space handling, digit
    grouping, byte fallback), so a substring's tokenization does not in general
    appear in the full sequence's tokenization, and matching on it produces
    positions that are quietly wrong.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise AnchorError(
            "a fast tokenizer is required for offset mapping; got {}".format(
                type(tokenizer).__name__))

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        return_tensors=None,
        padding=False,           # §A.11 item 3 — no padding, ever
        truncation=False,
    )
    input_ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(offsets) != len(input_ids):
        raise AnchorError("offset_mapping length != input_ids length")

    names = anchor_order(anchors)
    idxs: List[int] = []
    for nm in names:
        idxs.append(char_span_to_token_index(
            offsets, anchors[nm], strategy=strategy, name=nm,
            n_tokens=len(input_ids),
            allow_trailing_special=allow_trailing_special,
        ))

    if not allow_duplicate_positions and len(set(idxs)) != len(idxs):
        dup = [nm for nm, i in zip(names, idxs) if idxs.count(i) > 1]
        raise AnchorError(
            "anchors {} resolved to the same token index; the corresponding "
            "token states would be exact duplicates in the graph".format(dup))

    return AnchorResolution(
        names=names, token_indices=idxs, input_ids=input_ids,
        offsets=offsets, n_tokens=len(input_ids), strategy=strategy,
    )


# ==========================================================================
# 2. Forward hooks on the decoder blocks
# ==========================================================================
def _unpack_block_output(out: Any):
    """A decoder block returns either a tensor or a tuple whose [0] is it."""
    if isinstance(out, (tuple, list)):
        return out[0]
    hs = getattr(out, "last_hidden_state", None)
    if hs is not None:
        return hs
    return out


class ResidualCapture:
    """Context manager capturing the residual stream at the output of each block.

    Usage::

        with ResidualCapture(loaded) as cap:
            logits = forward_once(loaded, input_ids)
            resid = cap.residuals          # list, len == n_layers

    ``cap.residuals[i]`` is the OUTPUT of block ``i`` (see the module docstring
    for the full convention and the layernorm argument). Tensors are
    ``[1, T, d_model]`` — batch dimension retained and asserted equal to 1.

    Parameters
    ----------
    keep_on_gpu:
        Keep captures on the device. Default False: they are moved to host RAM
        so that a 40-layer capture does not sit next to the weights on a 48GB
        card. Set True only for the determinism gate, where you want the bits
        exactly as the kernel produced them (the H2D/D2H copy is bit-exact
        anyway, so this is paranoia, not necessity).
    cast_to:
        Optional torch dtype to cast captures to. ``None`` (default) keeps the
        model dtype — required for the G1 gate, which compares bf16 bit
        patterns. Extraction casts to float32 downstream, which is lossless from
        bf16.
    """

    def __init__(self, loaded: Any, *, keep_on_gpu: bool = False,
                 cast_to: Any = None, capture_embeddings: bool = False):
        self.loaded = loaded
        self.keep_on_gpu = keep_on_gpu
        self.cast_to = cast_to
        self.capture_embeddings = capture_embeddings
        self.n_layers = int(loaded.n_layers)
        self.residuals: List[Any] = [None] * self.n_layers
        self.embeddings: Any = None
        self._handles: List[Any] = []
        self._active = False

    # -- hook factory ------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):  # noqa: ANN001
            hidden = _unpack_block_output(output)
            # §A.4 / §A.11 item 3: batch size 1, no padding, EVER. Asserted at
            # the point of capture rather than trusted, because a padded
            # position yields a garbage activation that is indistinguishable
            # from a real token state once it is in the graph.
            if hidden.dim() != 3 or hidden.shape[0] != 1:
                raise ExtractionError(
                    "layer {}: expected residual [1, T, d] (batch size 1, no "
                    "padding) but got shape {}. Batching is forbidden during "
                    "extraction: float reduction order changes with batch shape "
                    "(§A.4) and padded positions leak into clusters "
                    "(§A.11).".format(layer_idx, tuple(hidden.shape)))
            t = hidden.detach()
            if self.cast_to is not None:
                t = t.to(self.cast_to)
            if not self.keep_on_gpu:
                t = t.to("cpu")
            self.residuals[layer_idx] = t.contiguous()
        return hook

    def _make_embed_hook(self):
        def hook(module, args, output):  # noqa: ANN001
            hidden = _unpack_block_output(output)
            self.embeddings = hidden.detach().to(
                "cpu" if not self.keep_on_gpu else hidden.device).contiguous()
        return hook

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "ResidualCapture":
        layers = self.loaded.layer_modules()
        if len(layers) != self.n_layers:
            raise ExtractionError(
                "layer_modules() has {} entries but n_layers is {}; the resolved "
                "module list is not the decoder stack".format(len(layers), self.n_layers))
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        if self.capture_embeddings:
            emb = self.loaded.model.get_input_embeddings()
            self._handles.append(emb.register_forward_hook(self._make_embed_hook()))
        self._active = True
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._active = False
        return None

    def clear(self) -> None:
        self.residuals = [None] * self.n_layers
        self.embeddings = None

    def assert_complete(self) -> None:
        missing = [i for i, r in enumerate(self.residuals) if r is None]
        if missing:
            raise ExtractionError(
                "no residual captured for blocks {}. Either the forward pass did "
                "not run, or the model skipped layers (early exit / layer "
                "dropout).".format(missing))


def forward_once(loaded: Any, input_ids: Any, *, attention_mask: Any = None) -> Any:
    """One batch-1, unpadded, no-cache forward pass. Returns the logits.

    ``use_cache=False``: a populated KV cache changes what the next call
    computes, and a cache carried between the two passes of the G1 gate is a
    classic way to get a nonzero fork signal that has nothing to do with the
    hypothesis.
    """
    import torch

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ExtractionError(
            "input_ids must be [1, T]; got {} (batch size 1, no padding — "
            "§A.4)".format(tuple(input_ids.shape)))
    if attention_mask is None:
        # All ones by construction: nothing is padded, so there is nothing to
        # mask. We pass it explicitly rather than letting it default, so that
        # the absence of padding is visible in the call.
        attention_mask = torch.ones_like(input_ids)
    elif int(attention_mask.sum().item()) != attention_mask.numel():
        raise ExtractionError(
            "attention_mask contains a zero — something is padded. Padded "
            "positions produce garbage activations that get clustered as real "
            "token states (§A.11 item 3).")

    with torch.inference_mode():
        out = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
    return out.logits


def encode(loaded: Any, text: str, *, add_special_tokens: bool = True) -> Any:
    """Tokenize one prompt into a ``[1, T]`` tensor. No padding, no truncation."""
    import torch

    enc = loaded.tokenizer(
        text, add_special_tokens=add_special_tokens, return_tensors=None,
        padding=False, truncation=False,
    )
    return torch.tensor([enc["input_ids"]], dtype=torch.long, device=loaded.device)


# ==========================================================================
# 3. Aligned extraction
# ==========================================================================
def extract_aligned(
    text: str,
    anchors: Dict[str, Sequence[int]],
    tokenizer: Any,
    model: Any,
    *,
    strategy: str = "last",
    add_special_tokens: bool = True,
    dtype: str = "float32",
    return_meta: bool = False,
):
    """Extract the residual stream at the semantically anchored positions.

    DESIGN.md §A.6: the graph is built from token states, so a lambda2
    difference between a 60-token NL prompt and a 28-token Python rewrite is
    confounded with length before semantics enter. The fix is to restrict to
    aligned positions — the final token plus the position at which each quantity
    is introduced — which are present in all four realizations.

    Returns
    -------
    numpy.ndarray
        ``[n_layers, n_anchor_positions, d_model]``, float32 by default.
        Axis 0 index ``i`` is the OUTPUT of block ``i`` (module docstring).
        Axis 1 is ordered by :func:`anchor_order` — quantity names sorted
        lexicographically, ``FINAL`` last — which is the same order in all four
        realizations of a problem.
        If ``return_meta`` is True, returns ``(array, AnchorResolution)``.

    Notes
    -----
    ``tokenizer`` is a parameter for signature clarity but must be the same
    tokenizer the model was loaded with; a mismatch is checked.
    """
    npy = _require_numpy()
    import torch

    if getattr(model, "tokenizer", None) is not None and model.tokenizer is not tokenizer:
        warnings.warn(
            "extract_aligned was given a tokenizer that is not the loaded model's "
            "tokenizer; offset mappings and input_ids would disagree.",
            RuntimeWarning, stacklevel=2)

    res = resolve_anchor_positions(
        text, anchors, tokenizer, strategy=strategy,
        add_special_tokens=add_special_tokens,
    )

    input_ids = torch.tensor([res.input_ids], dtype=torch.long, device=model.device)

    with ResidualCapture(model) as cap:
        forward_once(model, input_ids)
        cap.assert_complete()
        idx = torch.tensor(res.token_indices, dtype=torch.long)
        # [L, P, d]; index on the CPU copies, then cast. bf16 -> float32 is exact.
        out = npy.empty((model.n_layers, res.n_positions, model.d_model),
                        dtype=npy.dtype(dtype))
        for i, resid in enumerate(cap.residuals):
            sel = resid[0].index_select(0, idx.to(resid.device))
            out[i] = sel.to(torch.float32).cpu().numpy().astype(dtype, copy=False)
        cap.clear()

    if return_meta:
        return out, res
    return out


# ==========================================================================
# 4. Per-layer streaming store  (§A.1: never hold every layer of every prompt)
# ==========================================================================
@dataclass
class StateRow:
    """Provenance of one token state. Kept alongside every activation row."""

    problem_id: str
    condition: str
    anchor_name: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.problem_id, self.condition, self.anchor_name)


@dataclass
class LayerBatch:
    """All token states for ONE layer, across every prompt added to the store."""

    layer: int
    states: Any                 # numpy [N, d_model]
    rows: List[StateRow]

    def __post_init__(self):
        if self.states.shape[0] != len(self.rows):
            raise ExtractionError("states/rows length mismatch")

    @property
    def n_states(self) -> int:
        return len(self.rows)

    def index_of(self, problem_id: str, condition: str, anchor_name: str) -> int:
        return self.rows.index(StateRow(problem_id, condition, anchor_name))


class AlignedActivationStore:
    """Collect aligned activations prompt-by-prompt, hand them back layer-by-layer.

    The access pattern the pipeline needs is transposed from the one the model
    produces: a forward pass yields all layers of ONE prompt, but spectral
    clustering needs all prompts of ONE layer. This class does that transpose
    without ever materialising [all prompts x all layers] as a live array in the
    caller's hands.

    Two backends:

    * ``spill_dir=None`` (default) — per-layer buffers in host RAM.
      ``memory_estimate_bytes()`` tells you the total; for the planned corpus
      (400 prompts x ~40 layers x ~20 positions x d_model=3840, float32) that is
      about 4.9 GB, which fits comfortably in a pod's host RAM but not on the
      card. Nothing touches disk.
    * ``spill_dir=<path>`` — one append-only scratch file per layer. Use this if
      host RAM is tight. **Scratch only**: it is deleted by ``close()`` /
      context-manager exit, it must be node-local (never a network volume,
      §A.1 rule 2), and it refuses to live under ``results/`` or ``data/``,
      because §A.1 rule 1 is that raw activations are never *persisted* —
      extract, cluster, write assignments, discard.

    Either way, ``iter_layers()`` yields one ``LayerBatch`` at a time, so the
    CALLER never holds more than one layer: the spectral machinery sees
    ``[N_states, d_model]``, not ``[n_layers, N_states, d_model]``. With
    ``release=True`` the store also drops each layer's buffer after yielding it,
    so total residency falls monotonically through the sweep (do that only if
    nothing needs a second pass over the layers).
    """

    def __init__(self, n_layers: int, d_model: int, *, dtype: str = "float32",
                 spill_dir: Optional[str] = None):
        _require_numpy()
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.dtype = np.dtype(dtype)
        self.rows: List[StateRow] = []
        self._closed = False
        self._spill_dir = None
        self._buffers: Optional[List[List[Any]]] = None
        self._released: set = set()

        if spill_dir is None:
            self._buffers = [[] for _ in range(self.n_layers)]
        else:
            spill_dir = os.path.abspath(spill_dir)
            for forbidden in ("results", "data"):
                parts = spill_dir.split(os.sep)
                if forbidden in parts:
                    raise ExtractionError(
                        "refusing to spill raw activations under {}/ — DESIGN.md "
                        "§A.1 rule 1: raw activations are NEVER persisted. Use a "
                        "node-local scratch path such as /dev/shm/ling or "
                        "/tmp/ling.".format(forbidden))
            os.makedirs(spill_dir, exist_ok=True)
            self._spill_dir = spill_dir
            self._files = [
                open(os.path.join(spill_dir, "layer_{:03d}.raw".format(i)), "wb")
                for i in range(self.n_layers)
            ]

    # -- writing -----------------------------------------------------------
    def add(self, problem_id: str, condition: str, anchor_names: Sequence[str],
            values: Any) -> None:
        """Add one prompt's aligned activations, ``values`` = ``[L, P, d]``."""
        if self._closed:
            raise ExtractionError("store is closed")
        if values.shape != (self.n_layers, len(anchor_names), self.d_model):
            raise ExtractionError(
                "expected [{}, {}, {}], got {}".format(
                    self.n_layers, len(anchor_names), self.d_model, values.shape))
        arr = values.astype(self.dtype, copy=False)
        for j, nm in enumerate(anchor_names):
            self.rows.append(StateRow(problem_id, condition, nm))
        for i in range(self.n_layers):
            block = np.ascontiguousarray(arr[i])
            if self._buffers is not None:
                self._buffers[i].append(block)
            else:
                self._files[i].write(block.tobytes())

    # -- reading -----------------------------------------------------------
    def iter_layers(self, layers: Optional[Iterable[int]] = None,
                    *, release: bool = False) -> Iterator[LayerBatch]:
        """Yield one :class:`LayerBatch` at a time.

        ``release=True`` frees each layer's stored buffer once it has been
        yielded — one-shot consumption, lower peak residency, no second pass.
        """
        if self._closed:
            raise ExtractionError("store is closed")
        if self._buffers is None:
            for fh in self._files:
                fh.flush()
        wanted = list(range(self.n_layers)) if layers is None else list(layers)
        for i in wanted:
            yield LayerBatch(layer=i, states=self._load_layer(i), rows=list(self.rows))
            if release:
                self._release_layer(i)

    def _release_layer(self, i: int) -> None:
        self._released.add(i)
        if self._buffers is not None:
            self._buffers[i] = []
        elif self._spill_dir is not None:
            try:
                os.remove(os.path.join(self._spill_dir, "layer_{:03d}.raw".format(i)))
            except OSError:
                pass

    def layer(self, i: int) -> LayerBatch:
        return LayerBatch(layer=i, states=self._load_layer(i), rows=list(self.rows))

    def _load_layer(self, i: int) -> Any:
        if i in self._released:
            raise ExtractionError(
                "layer {} was released after being streamed (release=True) and "
                "cannot be read again. Use release=False if you need a second "
                "pass over the layers.".format(i))
        if self._buffers is not None:
            bufs = self._buffers[i]
            if not bufs:
                return np.empty((0, self.d_model), dtype=self.dtype)
            return np.concatenate(bufs, axis=0)
        path = os.path.join(self._spill_dir, "layer_{:03d}.raw".format(i))
        arr = np.fromfile(path, dtype=self.dtype)
        return arr.reshape(-1, self.d_model)

    # -- accounting / teardown --------------------------------------------
    @property
    def n_states(self) -> int:
        return len(self.rows)

    def memory_estimate_bytes(self) -> int:
        return self.n_layers * self.n_states * self.d_model * self.dtype.itemsize

    @staticmethod
    def projected_bytes(n_prompts: int, n_positions: int, n_layers: int,
                        d_model: int, itemsize: int = 4) -> int:
        return n_prompts * n_positions * n_layers * d_model * itemsize

    def close(self) -> None:
        """Discard everything. §A.1: extract -> cluster -> assignments -> discard."""
        if self._closed:
            return
        self._buffers = None
        if self._spill_dir is not None:
            for fh in getattr(self, "_files", []):
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._spill_dir = None
        self.rows = []
        self._closed = True

    def __enter__(self) -> "AlignedActivationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ==========================================================================
# 5. Per-layer checkpointing  (§A.1 rule 3)
# ==========================================================================
class LayerCheckpointWriter:
    """Write one checkpoint per layer, and resume from the last completed one.

    §A.1 rule 3: "Checkpoint after every layer. A crash at layer 31 should cost
    you layer 31." §A.1 rule 1: what gets written is *cluster assignments and
    metrics*, never raw activations — assignments for 400 prompts x ~40 layers x
    ~20 positions are a few hundred KB, raw activations are several GB.

    That rule is enforced, not documented: :meth:`write_layer` rejects any array
    whose trailing dimension looks like ``d_model``, and any float array bigger
    than ``max_float_elems``.
    """

    def __init__(self, root: str, run_id: str, *, d_model: Optional[int] = None,
                 max_float_elems: int = 200_000):
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.d_model = d_model
        self.max_float_elems = max_float_elems

    def path(self, layer: int) -> str:
        return os.path.join(self.dir, "layer_{:03d}.json".format(layer))

    def is_done(self, layer: int) -> bool:
        return os.path.exists(self.path(layer))

    def completed_layers(self) -> List[int]:
        out = []
        for p in glob.glob(os.path.join(self.dir, "layer_*.json")):
            base = os.path.basename(p)
            try:
                out.append(int(base[len("layer_"):-len(".json")]))
            except ValueError:
                continue
        return sorted(out)

    def _guard(self, payload: Dict[str, Any]) -> None:
        for key, val in payload.items():
            arr = val
            if np is not None and isinstance(arr, np.ndarray):
                if self.d_model is not None and arr.ndim >= 2 and arr.shape[-1] == self.d_model:
                    raise ExtractionError(
                        "refusing to checkpoint {!r}: trailing dimension is d_model="
                        "{}, i.e. this is a raw activation matrix. DESIGN.md §A.1 "
                        "rule 1 — extract, cluster, write ASSIGNMENTS, discard."
                        .format(key, self.d_model))
                if arr.dtype.kind == "f" and arr.size > self.max_float_elems:
                    raise ExtractionError(
                        "refusing to checkpoint {!r}: {} float elements exceeds the "
                        "{} guard. Checkpoints hold assignments and metrics, not "
                        "activations (§A.1 rule 1).".format(
                            key, arr.size, self.max_float_elems))

    def write_layer(self, layer: int, payload: Dict[str, Any]) -> str:
        """Atomically write one layer's checkpoint. Returns the path."""
        self._guard(payload)
        body = {"layer": layer}
        for key, val in payload.items():
            if np is not None and isinstance(val, np.ndarray):
                body[key] = val.tolist()
            elif np is not None and isinstance(val, (np.integer, np.floating)):
                body[key] = val.item()
            else:
                body[key] = val
        tmp = self.path(layer) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(layer))   # atomic: a preempted pod never
        return self.path(layer)             # leaves a half-written checkpoint

    def read_layer(self, layer: int) -> Dict[str, Any]:
        with open(self.path(layer), "r", encoding="utf-8") as fh:
            return json.load(fh)


def stream_layers_with_checkpoints(
    store: "AlignedActivationStore",
    writer: "LayerCheckpointWriter",
    fn,
    *,
    resume: bool = True,
    log=None,
) -> List[int]:
    """Drive the layer loop: for each layer, compute, checkpoint, free.

    ``fn(batch: LayerBatch) -> dict`` returns the per-layer payload (cluster
    assignments, k, eigengap, lambda2, ...). It is called once per layer, with
    only that layer's states resident. Layers already checkpointed are skipped
    when ``resume=True`` (§A.1 rule 3).

    Returns the list of layer indices actually computed this call.
    """
    done = set(writer.completed_layers()) if resume else set()
    computed: List[int] = []
    for batch in store.iter_layers():
        if batch.layer in done:
            if log:
                log("layer {:3d}: checkpoint exists, skipping".format(batch.layer))
            continue
        payload = fn(batch)
        writer.write_layer(batch.layer, payload)
        computed.append(batch.layer)
        if log:
            log("layer {:3d}: {} states -> {}".format(
                batch.layer, batch.n_states, writer.path(batch.layer)))
        del batch
    return computed
