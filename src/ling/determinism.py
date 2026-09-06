"""Determinism controls and bitwise-equality assertions.

Covers DESIGN.md §A.4 (determinism gate) and the gate G1 declared in
`prereg/PRIMARY_ENDPOINTS.md`.

Why this module exists
----------------------
Greedy decoding at temperature 0 is *not* bitwise deterministic on GPU across
batch shapes: float reduction order changes with batch size and padding, so the
same prompt in two different batches yields different logits and therefore a
nonzero "fork signal" that has nothing to do with the hypothesis. §A.4 requires
the fork signal to be **exactly zero**, not "near zero". Everything downstream
(cluster assignments, lambda2, fork layer) is otherwise measuring numerical
noise.

ORDERING REQUIREMENT (read this before editing)
-----------------------------------------------
``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS *once*, when the CUDA context is
created. Setting it after ``torch.cuda`` has initialised has no effect, and
``torch.use_deterministic_algorithms(True)`` will then raise at the first cuBLAS
GEMM (or, worse on some builds, silently use a non-deterministic workspace).

This module therefore sets the variable at **import** time, before it imports
torch. Consequence: ``import ling.determinism`` must happen before any CUDA
work. ``enforce_determinism()`` verifies this and raises if CUDA was already
initialised by the time the variable was set.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Step 0: environment, BEFORE torch is imported anywhere in this process.
# --------------------------------------------------------------------------
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

#: True if *we* were the ones who put CUBLAS_WORKSPACE_CONFIG in the environment
#: (as opposed to it already being exported by e.g. scripts/pod_bootstrap.sh).
_CUBLAS_SET_BY_US = False
#: True if torch was already imported by the time this module was first imported.
#: Not fatal on its own (torch import does not create a CUDA context), but it is
#: a warning sign that some other module may already have touched CUDA.
_TORCH_ALREADY_IMPORTED = "torch" in sys.modules

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    _CUBLAS_SET_BY_US = True

#: Value actually in force for this process.
CUBLAS_WORKSPACE_CONFIG_EFFECTIVE = os.environ["CUBLAS_WORKSPACE_CONFIG"]

# torch is imported lazily so that this module (and the anchor-mapping helpers in
# ling.hooks that import it) stay importable on a machine with no torch at all.
try:  # pragma: no cover - trivially environment dependent
    import torch as _torch
except Exception:  # noqa: BLE001
    _torch = None

try:  # pragma: no cover
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None


class DeterminismError(RuntimeError):
    """Raised when a determinism precondition or a bitwise assertion fails."""


def torch_available() -> bool:
    return _torch is not None


def _require_torch():
    if _torch is None:
        raise DeterminismError(
            "torch is not installed in this interpreter. determinism controls and "
            "model loading require torch; the anchor-mapping helpers do not."
        )
    return _torch


# --------------------------------------------------------------------------
# enforce_determinism
# --------------------------------------------------------------------------
def enforce_determinism(seed: int, *, strict: bool = True) -> Dict[str, Any]:
    """Apply every determinism control required by DESIGN.md §A.4.

    Parameters
    ----------
    seed:
        Master seed. Applied to ``random``, ``numpy``, ``torch`` (CPU) and all
        CUDA devices.
    strict:
        If True (default) raise when a control cannot be applied — notably when
        CUDA was already initialised before ``CUBLAS_WORKSPACE_CONFIG`` was set.
        Setting ``strict=False`` downgrades that to a warning; do not do this for
        the G1 gate run.

    Returns
    -------
    dict
        The settings actually applied, for logging into the run record and for
        the one-line methods sentence in the paper.

    Notes
    -----
    Controls applied, and why each one is here:

    * ``torch.use_deterministic_algorithms(True)`` — makes torch raise rather
      than silently pick a nondeterministic kernel (scatter/index_put, some
      reductions). We want the loud failure.
    * ``torch.backends.cudnn.deterministic = True`` and ``benchmark = False`` —
      cudnn's autotuner picks a different algorithm depending on timing noise,
      which changes reduction order between two runs of the *same* shape.
    * ``allow_tf32 = False`` for **both** ``cuda.matmul`` and ``cudnn`` — TF32
      truncates the mantissa to 10 bits in the multiplier and accumulates in
      fp32; the reduction noise is what §A.4 calls out. Both flags matter: they
      are separate switches for separate backends.
    * ``allow_bf16/fp16_reduced_precision_reduction = False`` — split-K matmul
      reductions in reduced precision are shape dependent. Our model runs in
      bf16, so the bf16 flag is the load-bearing one.
    * ``set_float32_matmul_precision("highest")`` — the modern front end for the
      same TF32 switch; set for redundancy against torch version drift.
    * Seeds for python ``random``, numpy, torch CPU and all CUDA devices.

    Batch size 1 and "no padding" are *not* settable here — they are enforced at
    the extraction call site (see ``ling.hooks``), because they are properties of
    how the model is called, not of global state.
    """
    torch = _require_torch()

    applied: Dict[str, Any] = {}
    problems = []

    # --- cuBLAS workspace ordering check -------------------------------------
    cuda_already_init = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    if cuda_already_init and _CUBLAS_SET_BY_US:
        problems.append(
            "CUDA was already initialised before ling.determinism was imported, and "
            "CUBLAS_WORKSPACE_CONFIG was not present in the environment beforehand. "
            "cuBLAS reads that variable when the CUDA context is created, so it is "
            "now too late for it to take effect. Fix: `import ling.determinism` (or "
            "`import ling`) as the FIRST project import, or export "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} in the shell "
            "(scripts/pod_bootstrap.sh does this)."
        )
    applied["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_EFFECTIVE
    applied["cublas_config_set_by_ling"] = _CUBLAS_SET_BY_US
    applied["cuda_initialized_before_enforce"] = cuda_already_init
    applied["torch_imported_before_ling_determinism"] = _TORCH_ALREADY_IMPORTED

    # --- PYTHONHASHSEED -------------------------------------------------------
    # Only affects str/bytes hash randomisation, which we do not rely on for any
    # numeric result, but set-iteration order can leak into e.g. landmark
    # selection if someone iterates a set. Cannot be set from inside a running
    # interpreter, so we can only report it.
    applied["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
    if applied["PYTHONHASHSEED"] is None:
        warnings.warn(
            "PYTHONHASHSEED is unset. It cannot be changed after interpreter start; "
            "export PYTHONHASHSEED=0 in the launcher (scripts/pod_bootstrap.sh does) "
            "if any code iterates a set or dict keyed by strings.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- seeds ----------------------------------------------------------------
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    applied["seed"] = seed
    applied["numpy_seeded"] = _np is not None

    # --- deterministic algorithms --------------------------------------------
    torch.use_deterministic_algorithms(True)
    applied["use_deterministic_algorithms"] = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    applied["cudnn.deterministic"] = True
    applied["cudnn.benchmark"] = False

    # --- TF32 off, both backends ---------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    applied["cuda.matmul.allow_tf32"] = False
    applied["cudnn.allow_tf32"] = False

    for flag in (
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, flag):
            setattr(torch.backends.cuda.matmul, flag, False)
            applied["cuda.matmul." + flag] = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
        applied["float32_matmul_precision"] = "highest"

    # Attention backend selection: SDPA dispatches to flash / mem-efficient
    # kernels whose reduction order depends on sequence length and head layout,
    # so it is not bitwise reproducible across shapes. We load the model with
    # attn_implementation="eager" (see ling.model) which bypasses SDPA entirely;
    # these flags are belt-and-braces in case some submodule calls SDPA directly.
    sdpa = getattr(getattr(torch.backends, "cuda", None), "enable_flash_sdp", None)
    if sdpa is not None:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            applied["sdpa_backend"] = "math_only"
        except Exception as exc:  # noqa: BLE001  - version dependent
            applied["sdpa_backend"] = "unset ({})".format(type(exc).__name__)

    applied["torch_version"] = torch.__version__
    applied["cuda_available"] = bool(torch.cuda.is_available())
    if applied["cuda_available"]:
        applied["cuda_device_name"] = torch.cuda.get_device_name(0)
        applied["cuda_version"] = torch.version.cuda
        try:
            applied["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            applied["cudnn_version"] = None

    if problems:
        msg = "determinism preconditions not met:\n  - " + "\n  - ".join(problems)
        if strict:
            raise DeterminismError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    applied["problems"] = problems

    return applied


def determinism_summary_line(applied: Dict[str, Any]) -> str:
    """One-line methods sentence, for pasting into the paper."""
    return (
        "Determinism: torch {tv} (CUDA {cv}), {dev}, bf16, eager attention, batch size 1, "
        "no padding; deterministic algorithms on, cuDNN deterministic, cuDNN benchmark off, "
        "TF32 off for matmul and cuDNN, reduced-precision reductions off, "
        "CUBLAS_WORKSPACE_CONFIG={cw}, seed={seed}."
    ).format(
        tv=applied.get("torch_version", "?"),
        cv=applied.get("cuda_version", "n/a"),
        dev=applied.get("cuda_device_name", "cpu"),
        cw=applied.get("CUBLAS_WORKSPACE_CONFIG", "?"),
        seed=applied.get("seed", "?"),
    )


# --------------------------------------------------------------------------
# bitwise comparison
# --------------------------------------------------------------------------
def _describe(x: Any) -> str:
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    if shape is None:
        return "{}(len={})".format(type(x).__name__, _safe_len(x))
    return "{}{} {}".format(type(x).__name__, tuple(shape), dtype)


def _safe_len(x: Any):
    try:
        return len(x)
    except Exception:  # noqa: BLE001
        return "?"


def _is_torch_tensor(x: Any) -> bool:
    return _torch is not None and isinstance(x, _torch.Tensor)


def _flatten_to_list(x: Any):
    """Flatten a nested sequence of numbers into a flat python list."""
    out = []
    stack = [x]
    # depth-first, preserving order
    def rec(v):
        if isinstance(v, (list, tuple)):
            for item in v:
                rec(item)
        else:
            out.append(v)
    rec(x)
    del stack
    return out


def assert_bitwise_identical(a: Any, b: Any, name: str = "tensor") -> None:
    """Assert that ``a`` and ``b`` are **exactly** equal, element for element.

    This is deliberately NOT ``allclose``. DESIGN.md §A.4: "Fork signal must be
    exactly zero, not 'near zero'." A tolerance here would let a plumbing bug —
    batch-shape-dependent reduction order, TF32 left on, a padded position
    sneaking in — pass the gate, and every downstream number would then be
    partly numerical noise.

    Accepts torch tensors, numpy arrays, or nested python sequences of numbers.
    NaN is treated as equal to NaN (bit patterns match); a signed-zero
    difference IS reported, because the bit patterns differ.

    Raises
    ------
    DeterminismError
        With the first differing flat index, the two values there, the absolute
        magnitude of the difference, the number of differing elements and the
        maximum absolute difference.
    """
    # --- structural checks ---------------------------------------------------
    sa, sb = getattr(a, "shape", None), getattr(b, "shape", None)
    if (sa is None) != (sb is None):
        raise DeterminismError(
            "{}: type mismatch: {} vs {}".format(name, _describe(a), _describe(b))
        )
    if sa is not None and tuple(sa) != tuple(sb):
        raise DeterminismError(
            "{}: shape mismatch: {} vs {}. Two passes produced different shapes, which "
            "means the two calls were not made under identical batch-1/no-padding "
            "conditions.".format(name, tuple(sa), tuple(sb))
        )
    da, db = getattr(a, "dtype", None), getattr(b, "dtype", None)
    if da is not None and db is not None and da != db:
        raise DeterminismError(
            "{}: dtype mismatch: {} vs {}. Compare like with like; an implicit upcast "
            "would mask a real bitwise difference.".format(name, da, db)
        )

    # --- torch path ----------------------------------------------------------
    if _is_torch_tensor(a) and _is_torch_tensor(b):
        torch = _torch
        ta = a.detach().to("cpu").contiguous()
        tb = b.detach().to("cpu").contiguous()
        if ta.numel() == 0:
            return
        # Byte-level comparison: catches NaN-vs-NaN (equal bits -> pass) and
        # +0.0-vs--0.0 (different bits -> fail), which value comparison gets
        # backwards in both cases.
        try:
            ba = ta.view(torch.uint8).reshape(-1)
            bb = tb.view(torch.uint8).reshape(-1)
            byte_ne = ba != bb
            if not bool(byte_ne.any()):
                return
            itemsize = ta.element_size()
            first_byte = int(torch.nonzero(byte_ne)[0].item())
            first_flat = first_byte // itemsize
        except Exception:  # noqa: BLE001 - some dtypes cannot be byte-viewed
            flat_a = ta.reshape(-1)
            flat_b = tb.reshape(-1)
            ne = flat_a != flat_b
            if flat_a.is_floating_point():
                ne = ne & ~(flat_a.isnan() & flat_b.isnan())
            if not bool(ne.any()):
                return
            first_flat = int(torch.nonzero(ne)[0].item())

        flat_a = ta.reshape(-1)
        flat_b = tb.reshape(-1)
        va = flat_a[first_flat]
        vb = flat_b[first_flat]
        if flat_a.is_floating_point():
            diff = (flat_a.to(torch.float64) - flat_b.to(torch.float64)).abs()
            n_diff = int((diff != 0).sum().item())
            max_abs = float(diff.max().item())
        else:
            diff = (flat_a.to(torch.float64) - flat_b.to(torch.float64)).abs()
            n_diff = int((flat_a != flat_b).sum().item())
            max_abs = float(diff.max().item())
        idx = _unravel(first_flat, tuple(ta.shape))
        raise DeterminismError(_failure_message(
            name, ta.shape, ta.numel(), first_flat, idx,
            float(va.to(torch.float64).item()), float(vb.to(torch.float64).item()),
            n_diff, max_abs,
        ))

    # --- numpy path ----------------------------------------------------------
    if _np is not None and isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray):
        fa = _np.ascontiguousarray(a).reshape(-1)
        fb = _np.ascontiguousarray(b).reshape(-1)
        if fa.size == 0:
            return
        try:
            ba = fa.view(_np.uint8)
            bb = fb.view(_np.uint8)
            if _np.array_equal(ba, bb):
                return
            itemsize = fa.dtype.itemsize
            first_flat = int(_np.flatnonzero(ba != bb)[0]) // itemsize
        except Exception:  # noqa: BLE001
            ne = fa != fb
            if _np.issubdtype(fa.dtype, _np.floating):
                ne = ne & ~(_np.isnan(fa) & _np.isnan(fb))
            if not ne.any():
                return
            first_flat = int(_np.flatnonzero(ne)[0])
        d = _np.abs(fa.astype(_np.float64) - fb.astype(_np.float64))
        idx = _unravel(first_flat, tuple(a.shape))
        raise DeterminismError(_failure_message(
            name, a.shape, fa.size, first_flat, idx,
            float(fa[first_flat]), float(fb[first_flat]),
            int((d != 0).sum()), float(d.max()),
        ))

    # --- plain python sequences / scalars ------------------------------------
    la = _flatten_to_list(a)
    lb = _flatten_to_list(b)
    if len(la) != len(lb):
        raise DeterminismError(
            "{}: length mismatch: {} vs {}".format(name, len(la), len(lb))
        )
    first_flat = None
    n_diff = 0
    max_abs = 0.0
    for i, (x, y) in enumerate(zip(la, lb)):
        same = (x == y) or (_is_nan(x) and _is_nan(y))
        if not same:
            n_diff += 1
            try:
                max_abs = max(max_abs, abs(float(x) - float(y)))
            except Exception:  # noqa: BLE001
                max_abs = float("nan")
            if first_flat is None:
                first_flat = i
    if first_flat is None:
        return
    shape = getattr(a, "shape", (len(la),))
    raise DeterminismError(_failure_message(
        name, shape, len(la), first_flat, (first_flat,),
        la[first_flat], lb[first_flat], n_diff, max_abs,
    ))


def _is_nan(x: Any) -> bool:
    try:
        return x != x
    except Exception:  # noqa: BLE001
        return False


def _unravel(flat: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not shape:
        return (flat,)
    idx = []
    rem = flat
    for dim in reversed(shape):
        dim = int(dim)
        if dim == 0:
            return (flat,)
        idx.append(rem % dim)
        rem //= dim
    return tuple(reversed(idx))


def _failure_message(name, shape, numel, first_flat, idx, va, vb, n_diff, max_abs) -> str:
    return (
        "BITWISE MISMATCH in {name} (shape {shape}, {numel} elements).\n"
        "  first differing element: flat index {flat}, multi-index {idx}\n"
        "    pass A = {va!r}\n"
        "    pass B = {vb!r}\n"
        "    |A - B| = {mag:.17g}\n"
        "  differing elements: {n_diff} / {numel}   max |A - B| = {max_abs:.17g}\n"
        "  This is a plumbing bug, not a result. DESIGN.md §A.4 requires EXACTLY zero.\n"
        "  Check, in order: (1) batch size is 1 and nothing is padded; (2) "
        "attn_implementation is \"eager\" (SDPA/flash reduction order varies with shape); "
        "(3) TF32 is off for BOTH cuda.matmul and cudnn; (4) CUBLAS_WORKSPACE_CONFIG was "
        "in the environment BEFORE the CUDA context was created; (5) no KV cache is being "
        "reused across the two passes."
    ).format(
        name=name, shape=tuple(shape), numel=numel, flat=first_flat, idx=idx,
        va=va, vb=vb, mag=abs(float(va) - float(vb)) if _numeric(va, vb) else float("nan"),
        n_diff=n_diff, max_abs=max_abs,
    )


def _numeric(*vals) -> bool:
    try:
        for v in vals:
            float(v)
        return True
    except Exception:  # noqa: BLE001
        return False


def max_abs_difference(a: Any, b: Any) -> float:
    """Return max |a - b| as a float. Used for the paper's one-line summary."""
    if _is_torch_tensor(a) and _is_torch_tensor(b):
        torch = _torch
        d = (a.detach().to("cpu").to(torch.float64) - b.detach().to("cpu").to(torch.float64)).abs()
        return float(d.max().item()) if d.numel() else 0.0
    if _np is not None and isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray):
        if a.size == 0:
            return 0.0
        return float(_np.abs(a.astype(_np.float64) - b.astype(_np.float64)).max())
    la, lb = _flatten_to_list(a), _flatten_to_list(b)
    if not la:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(la, lb))


def tensor_digest(x: Any) -> str:
    """Stable hex digest of the raw bytes of a tensor/array.

    Used by the G1 gate to compare a whole sweep of prompts across orderings
    without holding every activation in memory. A digest match is a bitwise
    match; a digest mismatch triggers a full ``assert_bitwise_identical`` on the
    offending pair so the report names the element.
    """
    import hashlib

    h = hashlib.sha256()
    if _is_torch_tensor(x):
        t = x.detach().to("cpu").contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(_torch.uint8).reshape(-1).numpy().tobytes())
    elif _np is not None and isinstance(x, _np.ndarray):
        arr = _np.ascontiguousarray(x)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    else:
        h.update(repr(_flatten_to_list(x)).encode())
    return h.hexdigest()
