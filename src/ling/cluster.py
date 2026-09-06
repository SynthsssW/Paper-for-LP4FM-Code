"""Spectral clustering of the pooled aligned token states, with *k* fixed on nl_base.

Covers DESIGN.md §A.6 (fixed *k*, selected per layer on NL-base by eigengap and
applied to all four realizations), §A.8 (the clustering front end that lambda2,
path entropy and fork layer are all computed on), §A.11 item 4 (*k* is never
recomputed per condition) and §A.11 item 6 (Nystrom landmarks sampled randomly
from the POOLED set, never per condition), prereg decision 1, gate G4 (Nystrom
approximation error against the exact spectrum on a held-out subset), and
prereg/DEVIATIONS.md Amendment 3 (cosine affinity primary, per-layer
standardization as the appendix robustness option, raw NEVER).

===========================================================================
ONE CLUSTERING PER LAYER, OVER THE POOLED ROWS
===========================================================================
At a given layer the four conditions are clustered TOGETHER, in one pooled
partition, and every row gets a label out of the same ``[0, k)`` vocabulary.
This is not a shortcut, it is forced by two downstream consumers:

* ``metrics.fork_layer`` decides that two realizations of a problem have forked
  when their cluster SETS become disjoint. Comparing labels across conditions
  is meaningless unless the label space is shared, so per-condition clusterings
  (each with its own arbitrary label permutation) would make H2 nonsense.
* ``metrics.lambda2`` builds a per-condition transition graph whose node set is
  ``k`` clusters at layer p and ``k`` clusters at layer p+1. §A.6's whole point
  is that this node count must not differ between conditions.

Pooled clustering makes "identical k across conditions at a layer" structurally
true rather than merely asserted. What is NOT automatic, and is therefore
enforced by the shape of this module's API, is that *k* itself is chosen from
nl_base alone:

    sel   = select_k(base_states, base_conditions)     # refuses non-base rows
    res   = apply_clustering(pooled_states, sel.k)     # cannot compute a k

:func:`select_k` raises if it is handed a row from any condition other than the
declared base condition, so it cannot see the pooled set. :func:`apply_clustering`
takes ``k`` as a required argument and contains no k-selection code at all, so it
cannot recompute one. :func:`cluster_layer` wires the two together.

===========================================================================
WHAT "COSINE AFFINITY" MEANS HERE  (judgment call, stated loudly)
===========================================================================
Amendment 3 fixes the *representation* as cosine: scale-invariant, immune to the
depth-dependent norm growth of the pre-normalization residual stream. It does
not fix the *kernel shape*, and the two are separable questions.

A literal dot-product affinity ``A = X_hat X_hat^T`` makes normalized spectral
clustering a linear method (its eigenvectors are the top singular vectors of
``X_hat``), which cannot recover non-convex structure — and H1's descriptive
note is precisely a claim about non-convex structure. So the default kernel is a
Gaussian on the cosine distance,

    A_ij = exp(-||x_hat_i - x_hat_j||^2 / (2 sigma^2)),  x_hat = x / ||x||

which is a strictly monotone function of ``1 - cos(x_i, x_j)`` and therefore
exactly as scale-invariant as Amendment 3 requires, while still being a local
kernel. ``kernel="cosine_linear"`` implements the literal reading for anyone who
wants to re-run under it. The kernel and the bandwidth are both recorded in the
per-layer output; ``affinity`` in the schema stays ``"cosine"`` because that is
the field name the Amendment and configs/assignments_schema.md use for the
representation.

``sigma`` is a self-tuning scalar: the median over rows of the distance to the
``knn_sigma``-th nearest reference point, computed ON THE POOLED SET for exactly
the reason landmarks are pooled — a per-condition bandwidth would build a
between-condition difference into the affinity before anything is measured.

===========================================================================
BACKEND
===========================================================================
numpy is used when present. A pure-python fallback exists so the tests can run
on a bare interpreter, but it is a TEST backend: it refuses more than
``PURE_PY_MAX_ROWS`` rows loudly rather than silently taking an hour. The
production pod has numpy (requirements.txt).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised by whichever backend is installed
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Amendment 3. `raw` is not in this tuple and never will be: it confounds depth
#: with geometry, and the H3 primary endpoint averages ACROSS layer-pairs, which
#: is the exact axis a depth-dependent scale artifact accumulates along.
COSINE = "cosine"
PER_LAYER_STANDARDIZED = "per_layer_standardized"
VALID_AFFINITIES = (COSINE, PER_LAYER_STANDARDIZED)

KERNEL_GAUSSIAN = "gaussian_cosine"
KERNEL_LINEAR = "cosine_linear"
VALID_KERNELS = (KERNEL_GAUSSIAN, KERNEL_LINEAR)

DEFAULT_BASE_CONDITION = "nl_base"

DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 20

#: Above this many rows the pure-python backend refuses to run. Jacobi is O(n^3)
#: per sweep in interpreted python; 250 rows is ~1s, 2000 rows is not "slow", it
#: is "the run never finishes and nobody knows why".
PURE_PY_MAX_ROWS = 250

#: Above this many rows the EXACT (dense n x n) path refuses to run without an
#: explicit override; use Nystrom instead.
DEFAULT_EXACT_MAX_ROWS = 3000

_EPS = 1e-12


class ClusterError(RuntimeError):
    """A clustering precondition was violated. Never downgraded to a warning."""


def have_numpy() -> bool:
    return np is not None


# --------------------------------------------------------------------------------------
# 0. Tiny dual-backend linear algebra
#
# Matrices are numpy 2-D float64 arrays when numpy is present, and lists of lists
# of floats otherwise. Every helper below branches once, at the top, so the
# algorithms further down are written exactly once.
# --------------------------------------------------------------------------------------


def as_matrix(x: Any) -> Any:
    """Coerce ``x`` to the backend's 2-D float matrix type."""
    if np is not None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ClusterError("expected a 2-D [n_rows, n_features] matrix, got shape %r"
                               % (arr.shape,))
        return arr
    if hasattr(x, "tolist"):
        x = x.tolist()
    if not isinstance(x, (list, tuple)) or not x:
        raise ClusterError("expected a non-empty 2-D matrix")
    out = []
    width = None
    for row in x:
        if not isinstance(row, (list, tuple)):
            raise ClusterError("expected a 2-D matrix; row is not a sequence")
        vals = [float(v) for v in row]
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise ClusterError("ragged matrix: rows of length %d and %d" % (width, len(vals)))
        out.append(vals)
    return out


def shape(M: Any) -> Tuple[int, int]:
    if np is not None and isinstance(M, np.ndarray):
        return (int(M.shape[0]), int(M.shape[1]))
    return (len(M), len(M[0]) if M else 0)


def take_rows(M: Any, idx: Sequence[int]) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return M[list(idx), :]
    return [list(M[i]) for i in idx]


def take_cols(M: Any, idx: Sequence[int]) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return M[:, list(idx)]
    return [[row[j] for j in idx] for row in M]


def transpose(M: Any) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return M.T
    n, m = shape(M)
    return [[M[i][j] for i in range(n)] for j in range(m)]


def matmul(A: Any, B: Any) -> Any:
    if np is not None and isinstance(A, np.ndarray):
        return A @ B
    n, k = shape(A)
    k2, m = shape(B)
    if k != k2:
        raise ClusterError("matmul shape mismatch %r x %r" % (shape(A), shape(B)))
    Bt = transpose(B)
    return [[sum(A[i][t] * Bt[j][t] for t in range(k)) for j in range(m)] for i in range(n)]


def row_sums(M: Any) -> List[float]:
    if np is not None and isinstance(M, np.ndarray):
        return M.sum(axis=1).tolist()
    return [float(sum(row)) for row in M]


def scale_rows(M: Any, v: Sequence[float]) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return M * np.asarray(v, dtype=np.float64)[:, None]
    return [[M[i][j] * v[i] for j in range(len(M[i]))] for i in range(len(M))]


def scale_cols(M: Any, v: Sequence[float]) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return M * np.asarray(v, dtype=np.float64)[None, :]
    return [[M[i][j] * v[j] for j in range(len(M[i]))] for i in range(len(M))]


def symmetrize(M: Any) -> Any:
    if np is not None and isinstance(M, np.ndarray):
        return 0.5 * (M + M.T)
    n, m = shape(M)
    return [[0.5 * (M[i][j] + M[j][i]) for j in range(m)] for i in range(n)]


def add(A: Any, B: Any) -> Any:
    if np is not None and isinstance(A, np.ndarray):
        return A + B
    return [[A[i][j] + B[i][j] for j in range(len(A[i]))] for i in range(len(A))]


def row_l2_normalize(X: Any) -> Any:
    """Unit-norm rows. Zero rows are left at zero (and reported by the caller)."""
    if np is not None and isinstance(X, np.ndarray):
        nrm = np.sqrt((X * X).sum(axis=1))
        nrm = np.where(nrm < _EPS, 1.0, nrm)
        return X / nrm[:, None]
    out = []
    for row in X:
        nrm = math.sqrt(sum(v * v for v in row))
        if nrm < _EPS:
            nrm = 1.0
        out.append([v / nrm for v in row])
    return out


def col_standardize(X: Any) -> Any:
    """z-score every feature across ALL pooled rows of the layer (Amendment 3 appendix).

    The mean and sd are pooled over the four conditions on purpose: per-condition
    standardization would centre each condition separately and erase precisely the
    between-condition difference H3 is trying to measure.
    """
    if np is not None and isinstance(X, np.ndarray):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < _EPS, 1.0, sd)
        return (X - mu) / sd
    n, d = shape(X)
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    sd = []
    for j in range(d):
        var = sum((X[i][j] - mu[j]) ** 2 for i in range(n)) / max(n, 1)
        s = math.sqrt(var)
        sd.append(s if s >= _EPS else 1.0)
    return [[(X[i][j] - mu[j]) / sd[j] for j in range(d)] for i in range(n)]


def sqdist(X: Any, Y: Any) -> Any:
    """Squared euclidean distances, ``[n_X, n_Y]``. Clipped at 0."""
    if np is not None and isinstance(X, np.ndarray):
        xx = (X * X).sum(axis=1)[:, None]
        yy = (Y * Y).sum(axis=1)[None, :]
        D2 = xx + yy - 2.0 * (X @ Y.T)
        return np.maximum(D2, 0.0)
    out = []
    for xi in X:
        row = []
        for yj in Y:
            s = 0.0
            for a, b in zip(xi, yj):
                d = a - b
                s += d * d
            row.append(s if s > 0.0 else 0.0)
        out.append(row)
    return out


def _jacobi_eigh(M: Any, tol: float = 1e-11, max_sweeps: int = 100):
    """Cyclic Jacobi eigendecomposition of a symmetric matrix (pure python).

    Returns ``(eigenvalues ascending, V)`` where column ``j`` of ``V`` is the
    eigenvector for eigenvalue ``j``.
    """
    n = len(M)
    A = [list(map(float, row)) for row in M]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = math.sqrt(sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) <= tol / max(n, 1):
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for r in range(n):
                    arp, arq = A[r][p], A[r][q]
                    A[r][p] = c * arp - s * arq
                    A[r][q] = s * arp + c * arq
                for r in range(n):
                    apr, aqr = A[p][r], A[q][r]
                    A[p][r] = c * apr - s * aqr
                    A[q][r] = s * apr + c * aqr
                for r in range(n):
                    vrp, vrq = V[r][p], V[r][q]
                    V[r][p] = c * vrp - s * vrq
                    V[r][q] = s * vrp + c * vrq
    vals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    vals_sorted = [vals[i] for i in order]
    V_sorted = [[V[r][i] for i in order] for r in range(n)]
    return vals_sorted, V_sorted


def eigh(M: Any):
    """Symmetric eigendecomposition -> ``(eigenvalues ascending, eigenvector columns)``."""
    if np is not None and isinstance(M, np.ndarray):
        vals, vecs = np.linalg.eigh(symmetrize(M))
        return vals, vecs
    n, m = shape(M)
    if n != m:
        raise ClusterError("eigh needs a square matrix, got %r" % (shape(M),))
    if n > PURE_PY_MAX_ROWS:
        raise ClusterError(
            "numpy is not installed and the pure-python Jacobi eigensolver refuses a "
            "%dx%d matrix (limit %d). This is a TEST backend. Install numpy "
            "(requirements.txt pins 2.2.6) for anything at pipeline scale."
            % (n, n, PURE_PY_MAX_ROWS))
    return _jacobi_eigh(symmetrize(M))


def inv_sqrt_sym(M: Any, tol: float = 1e-10):
    """``M^{-1/2}`` on the positive eigen-subspace (a pseudo-inverse square root)."""
    vals, V = eigh(M)
    if np is not None and isinstance(M, np.ndarray):
        keep = vals > tol * max(float(np.max(np.abs(vals))), _EPS)
        d = np.zeros_like(vals)
        d[keep] = vals[keep] ** -0.5
        return (V * d[None, :]) @ V.T
    mx = max(abs(v) for v in vals) or 1.0
    d = [v ** -0.5 if v > tol * mx else 0.0 for v in vals]
    n = len(vals)
    return [[sum(V[i][t] * d[t] * V[j][t] for t in range(n)) for j in range(n)]
            for i in range(n)]


def pinv_sym(M: Any, tol: float = 1e-10):
    vals, V = eigh(M)
    if np is not None and isinstance(M, np.ndarray):
        keep = np.abs(vals) > tol * max(float(np.max(np.abs(vals))), _EPS)
        d = np.zeros_like(vals)
        d[keep] = 1.0 / vals[keep]
        return (V * d[None, :]) @ V.T
    mx = max(abs(v) for v in vals) or 1.0
    d = [1.0 / v if abs(v) > tol * mx else 0.0 for v in vals]
    n = len(vals)
    return [[sum(V[i][t] * d[t] * V[j][t] for t in range(n)) for j in range(n)]
            for i in range(n)]


def _col_slice_desc(vals: Any, V: Any, n_take: int):
    """Take the ``n_take`` LARGEST eigenpairs, returned in descending order."""
    if np is not None and isinstance(V, np.ndarray):
        order = list(range(len(vals)))[::-1][:n_take]      # eigh is ascending
        return [float(vals[i]) for i in order], V[:, order]
    order = list(range(len(vals)))[::-1][:n_take]
    return ([vals[i] for i in order],
            [[row[i] for i in order] for row in V])


# --------------------------------------------------------------------------------------
# 1. Representation and kernel  (Amendment 3)
# --------------------------------------------------------------------------------------


def prepare_states(states: Any, affinity: str = COSINE) -> Any:
    """Map raw residual-stream states to the representation the kernel runs on.

    ``cosine`` (primary)                -> L2-normalize each row.
    ``per_layer_standardized`` (appendix) -> pooled z-score per feature, then
                                             L2-normalize.

    Both are scale-invariant with respect to the depth-dependent norm growth of
    the pre-normalization residual stream, which is the whole point of
    Amendment 3. ``raw`` is not accepted at any layer of this module.
    """
    if affinity == "raw":
        raise ClusterError(
            "affinity 'raw' is forbidden (DEVIATIONS Amendment 3): activation norms grow "
            "monotonically with depth in the pre-normalization residual stream, and the "
            "H3 primary endpoint averages across layer-pairs, so the artifact accumulates "
            "along the exact axis the statistic sums over.")
    if affinity not in VALID_AFFINITIES:
        raise ClusterError("affinity must be one of %r, got %r" % (VALID_AFFINITIES, affinity))
    X = as_matrix(states)
    if affinity == PER_LAYER_STANDARDIZED:
        X = col_standardize(X)
    return row_l2_normalize(X)


def choose_sigma(X: Any, ref_idx: Sequence[int], *, knn_sigma: int = 7) -> float:
    """Self-tuning scalar bandwidth: median over rows of the knn-th nearest reference.

    ``ref_idx`` is the POOLED reference set (the Nystrom landmarks, or all rows on
    the exact path). A per-condition bandwidth is as forbidden as a per-condition
    landmark set, and for the same reason (§A.11 item 6).
    """
    ref = take_rows(X, ref_idx)
    D2 = sqdist(X, ref)
    n, m = shape(D2)
    q = max(1, min(int(knn_sigma), m - 1))
    dists: List[float] = []
    if np is not None and isinstance(D2, np.ndarray):
        srt = np.sort(D2, axis=1)
        dists = np.sqrt(srt[:, q]).tolist()
    else:
        for row in D2:
            srt = sorted(row)
            dists.append(math.sqrt(srt[q]))
    dists = [d for d in dists if d > _EPS]
    if not dists:
        raise ClusterError(
            "every state is identical to its neighbours at this layer: the knn distance "
            "is zero for every row, so no bandwidth exists. This is a degenerate layer, "
            "not something to paper over with a default sigma.")
    dists.sort()
    mid = len(dists) // 2
    sigma = dists[mid] if len(dists) % 2 else 0.5 * (dists[mid - 1] + dists[mid])
    if sigma <= _EPS:
        raise ClusterError("computed bandwidth sigma=%r is not positive" % sigma)
    return float(sigma)


def kernel_block(Xa: Any, Xb: Any, sigma: float, kernel: str = KERNEL_GAUSSIAN) -> Any:
    """Affinity between two sets of PREPARED (unit-norm) rows.

    ``gaussian_cosine`` : ``exp(-||x-y||^2 / (2 sigma^2))``, a strictly monotone
        function of the cosine distance ``1 - cos``, hence scale-invariant.
    ``cosine_linear``   : ``max(0, cos)``. The literal dot-product reading of
        Amendment 3. Kept so the choice can be re-run, not because it is the
        default: it makes spectral clustering a linear method.
    """
    if kernel not in VALID_KERNELS:
        raise ClusterError("kernel must be one of %r, got %r" % (VALID_KERNELS, kernel))
    if kernel == KERNEL_LINEAR:
        A = matmul(Xa, transpose(Xb))
        if np is not None and isinstance(A, np.ndarray):
            return np.maximum(A, 0.0)
        return [[v if v > 0.0 else 0.0 for v in row] for row in A]
    D2 = sqdist(Xa, Xb)
    denom = 2.0 * sigma * sigma
    if np is not None and isinstance(D2, np.ndarray):
        return np.exp(-D2 / denom)
    return [[math.exp(-v / denom) for v in row] for row in D2]


# --------------------------------------------------------------------------------------
# 2. Spectrum of the normalized Laplacian
#
# L_sym = I - D^{-1/2} A D^{-1/2}, the NORMALIZED laplacian. Stated here once so
# nothing downstream has to guess (§A.11 item 5 is the same requirement one level
# down, for lambda2 of the transition graph).
# --------------------------------------------------------------------------------------

LAPLACIAN_KIND = "normalized_symmetric"


def normalized_laplacian_spectrum(A: Any, n_eigs: Optional[int] = None):
    """-> ``(laplacian_eigenvalues_ascending, embedding_columns)``.

    The returned vectors are the eigenvectors of ``L_sym`` for the ``n_eigs``
    SMALLEST eigenvalues, i.e. the largest of ``S = D^{-1/2} A D^{-1/2}``. The
    trivial eigenvalue is NOT skipped here: the eigengap heuristic needs it, and
    lambda2's "skip the trivial zero" rule lives in metrics.py where the graph is
    the transition graph, not this affinity graph.
    """
    n, m = shape(A)
    if n != m:
        raise ClusterError("affinity must be square, got %r" % (shape(A),))
    d = row_sums(A)
    inv = [1.0 / math.sqrt(v) if v > _EPS else 0.0 for v in d]
    S = symmetrize(scale_cols(scale_rows(A, inv), inv))
    vals, V = eigh(S)
    take = n if n_eigs is None else max(1, min(int(n_eigs), n))
    s_vals, vecs = _col_slice_desc(vals, V, take)
    lap = [1.0 - v for v in s_vals]                  # ascending, since s desc
    return lap, vecs


# --------------------------------------------------------------------------------------
# 3. Nystrom  (§A.11 item 6, gate G4)
# --------------------------------------------------------------------------------------


def sample_landmarks(n_pooled: int, n_landmarks: int, seed: int,
                     *, conditions: Optional[Sequence[str]] = None) -> List[int]:
    """Sample landmark row indices UNIFORMLY AT RANDOM from the POOLED row set.

    ``n_pooled`` is the number of rows in the pooled matrix — all four conditions
    stacked — because that is the only set landmarks may come from. §A.11 item 6:
    a per-condition landmark set builds a between-condition difference into the
    embedding before anything has been measured, and it lands directly on H3.

    Pass ``conditions`` (the pooled per-row condition names) and this function
    will refuse a set that covers fewer conditions than are present, which is
    what a per-condition landmark set looks like from the inside.
    """
    if n_pooled <= 0:
        raise ClusterError("n_pooled must be positive")
    m = int(n_landmarks)
    if m < 2:
        raise ClusterError("need at least 2 landmarks, got %r" % (n_landmarks,))
    if conditions is not None and len(conditions) != n_pooled:
        raise ClusterError(
            "conditions has %d entries but the pooled set has %d rows; if these disagree "
            "the landmark set cannot be checked for pooled-ness (§A.11 item 6)"
            % (len(conditions), n_pooled))
    m = min(m, n_pooled)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(n_pooled), m))
    if conditions is not None:
        present = set(conditions)
        covered = {conditions[i] for i in idx}
        if len(present) > 1 and covered != present:
            raise ClusterError(
                "landmark set covers conditions %r but the pooled set contains %r. "
                "Nystrom landmarks must come from the POOLED set across all conditions "
                "(§A.11 item 6, prereg decision 7); a set missing a condition builds a "
                "between-condition difference into the embedding. Raise n_landmarks."
                % (sorted(covered), sorted(present)))
    return idx


def nystrom_spectral(X: Any, landmark_idx: Sequence[int], sigma: float,
                     *, kernel: str = KERNEL_GAUSSIAN, n_eigs: Optional[int] = None):
    """Nystrom approximation of the normalized-affinity eigensystem.

    Fowlkes et al. (2004) with the orthogonalization step, so the returned
    vectors are (approximately) orthonormal rather than merely spanning the right
    subspace — the un-orthogonalized variant produces a skewed embedding and
    therefore a k-means step that quietly disagrees with the exact solution.

    Returns ``(laplacian_eigenvalues_ascending, embedding_columns)`` with the same
    meaning as :func:`normalized_laplacian_spectrum`.
    """
    lm = list(landmark_idx)
    n, _ = shape(X)
    m = len(lm)
    if m < 2:
        raise ClusterError("need at least 2 landmarks")
    if m > n:
        raise ClusterError("more landmarks (%d) than rows (%d)" % (m, n))

    Xl = take_rows(X, lm)
    W = kernel_block(X, Xl, sigma, kernel=kernel)          # n x m
    C = take_rows(W, lm)                                   # m x m
    C = symmetrize(C)

    # Degree estimate for the FULL affinity: d ~= A_nn 1 = W C^+ W^T 1.
    Cp = pinv_sym(C)
    ones = [[1.0] for _ in range(n)]
    Wt1 = matmul(transpose(W), ones)                       # m x 1
    d_col = matmul(W, matmul(Cp, Wt1))                     # n x 1
    if np is not None and isinstance(d_col, np.ndarray):
        d = np.maximum(d_col[:, 0], _EPS)
        inv = (1.0 / np.sqrt(d)).tolist()
    else:
        d = [max(row[0], _EPS) for row in d_col]
        inv = [1.0 / math.sqrt(v) for v in d]
    inv_m = [inv[i] for i in lm]

    Wt = scale_cols(scale_rows(W, inv), inv_m)             # n x m
    Ct = symmetrize(scale_cols(scale_rows(C, inv_m), inv_m))

    Cih = inv_sqrt_sym(Ct)                                 # m x m
    R = symmetrize(add(Ct, matmul(Cih, matmul(matmul(transpose(Wt), Wt), Cih))))
    rvals, rvecs = eigh(R)
    take = m if n_eigs is None else max(1, min(int(n_eigs), m))
    s_vals, rv = _col_slice_desc(rvals, rvecs, take)

    scale = [1.0 / math.sqrt(v) if v > _EPS else 0.0 for v in s_vals]
    V = matmul(Wt, matmul(Cih, scale_cols(rv, scale)))     # n x take
    lap = [1.0 - v for v in s_vals]
    return lap, V


def nystrom_error_report(X: Any, landmark_idx: Sequence[int], holdout_idx: Sequence[int],
                         sigma: float, *, kernel: str = KERNEL_GAUSSIAN,
                         n_eigs: int = 10) -> Dict[str, Any]:
    """Gate G4: Nystrom approximation error against the EXACT spectrum, held out.

    The held-out rows must be disjoint from the landmarks — otherwise the
    "approximation" is being scored on the points it interpolates exactly, which
    is not an error estimate, it is a tautology. Both spectra are of the
    normalized Laplacian restricted to the held-out submatrix.
    """
    lm = list(landmark_idx)
    ho = list(holdout_idx)
    overlap = sorted(set(lm) & set(ho))
    if overlap:
        raise ClusterError(
            "G4 held-out subset overlaps the landmark set at rows %r. Nystrom reproduces "
            "landmark rows exactly, so scoring on them understates the error by "
            "construction." % overlap[:5])
    if len(ho) < 3:
        raise ClusterError("need at least 3 held-out rows for a G4 spectrum comparison")

    Xh = take_rows(X, ho)
    A_exact = kernel_block(Xh, Xh, sigma, kernel=kernel)
    exact_lap, _ = normalized_laplacian_spectrum(A_exact, n_eigs=min(n_eigs, len(ho)))

    # Nystrom's estimate of the SAME submatrix: A_hh ~= W_h C^+ W_h^T.
    Xl = take_rows(X, lm)
    Wh = kernel_block(Xh, Xl, sigma, kernel=kernel)
    C = symmetrize(kernel_block(Xl, Xl, sigma, kernel=kernel))
    A_approx = symmetrize(matmul(Wh, matmul(pinv_sym(C), transpose(Wh))))
    if np is not None and isinstance(A_approx, np.ndarray):
        A_approx = np.maximum(A_approx, 0.0)
    else:
        A_approx = [[v if v > 0.0 else 0.0 for v in row] for row in A_approx]
    approx_lap, _ = normalized_laplacian_spectrum(A_approx, n_eigs=min(n_eigs, len(ho)))

    diffs = [abs(a - b) for a, b in zip(exact_lap, approx_lap)]
    denom = max(max(abs(v) for v in exact_lap), _EPS)
    # Frobenius error on the affinity block itself, as a second, spectrum-free view.
    num = 0.0
    den = 0.0
    nh = len(ho)
    for i in range(nh):
        for j in range(nh):
            e = A_exact[i][j] if not (np is not None and isinstance(A_exact, np.ndarray)) \
                else float(A_exact[i, j])
            p = A_approx[i][j] if not (np is not None and isinstance(A_approx, np.ndarray)) \
                else float(A_approx[i, j])
            num += (e - p) ** 2
            den += e * e
    return {
        "gate": "G4",
        "n_landmarks": len(lm),
        "n_holdout": len(ho),
        "n_eigenvalues_compared": len(diffs),
        "laplacian_kind": LAPLACIAN_KIND,
        "exact_eigenvalues": [float(v) for v in exact_lap],
        "nystrom_eigenvalues": [float(v) for v in approx_lap],
        "max_abs_eigenvalue_error": float(max(diffs)) if diffs else 0.0,
        "mean_abs_eigenvalue_error": float(sum(diffs) / len(diffs)) if diffs else 0.0,
        "max_rel_eigenvalue_error": float(max(diffs) / denom) if diffs else 0.0,
        "affinity_relative_frobenius_error": float(math.sqrt(num / den)) if den > 0 else 0.0,
    }


# --------------------------------------------------------------------------------------
# 4. k selection -- NL-BASE ONLY  (§A.6, prereg decision 1, §A.11 item 4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KSelection:
    """The result of choosing *k* at one layer, on nl_base and nothing else."""

    k: int
    eigengap: float
    eigengap_flat: bool
    condition: str
    n_rows: int
    k_min: int
    k_max: int
    laplacian_eigenvalues: Tuple[float, ...]
    gaps: Tuple[float, ...]
    flat_ratio: float
    method: str
    sigma: float
    n_zero_eigenvalues: int
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "eigengap": self.eigengap,
            "eigengap_flat": self.eigengap_flat,
            "selected_on_condition": self.condition,
            "n_rows_used": self.n_rows,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "laplacian_eigenvalues": list(self.laplacian_eigenvalues),
            "gaps": list(self.gaps),
            "flat_ratio": self.flat_ratio,
            "method": self.method,
            "sigma": self.sigma,
            "laplacian_kind": LAPLACIAN_KIND,
            "n_zero_eigenvalues": self.n_zero_eigenvalues,
            "note": self.note,
        }


def eigengap_k(laplacian_eigenvalues: Sequence[float], *, k_min: int = DEFAULT_K_MIN,
               k_max: int = DEFAULT_K_MAX, flat_ratio_threshold: float = 1.5,
               flat_abs_tol: float = 1e-4):
    """Pick *k* by the eigengap heuristic, and say whether the spectrum is flat.

    ``gap(k) = mu[k] - mu[k-1]`` with ``mu`` the ascending eigenvalues of the
    normalized Laplacian (0-indexed), i.e. the jump AFTER the k-th smallest
    eigenvalue. ``k`` maximizes it over ``[k_min, k_max]``.

    A layer is flagged ``eigengap_flat`` (§A.6: "flag flat-spectrum layers as
    planned") when the winning gap is not meaningfully larger than the typical
    gap — ``best / median(other gaps) < flat_ratio_threshold`` — or when it is
    below ``flat_abs_tol`` in absolute terms. Flat means "this k is a coin toss";
    the layer is still clustered, at the argmax k, and the flag travels with it.
    """
    mu = [float(v) for v in laplacian_eigenvalues]
    if k_min < 2:
        raise ClusterError("k_min must be >= 2 (k=1 is not a partition), got %r" % (k_min,))
    if k_max < k_min:
        raise ClusterError("k_max (%r) < k_min (%r)" % (k_max, k_min))
    if len(mu) < k_min + 1:
        raise ClusterError(
            "need at least k_min+1 = %d eigenvalues to evaluate a gap at k_min, got %d"
            % (k_min + 1, len(mu)))
    hi = min(k_max, len(mu) - 1)
    cand = list(range(k_min, hi + 1))
    gaps = [mu[k] - mu[k - 1] for k in cand]
    best_i = max(range(len(gaps)), key=lambda i: gaps[i])
    best_k = cand[best_i]
    best_gap = gaps[best_i]

    others = sorted(g for i, g in enumerate(gaps) if i != best_i)
    if others:
        mid = len(others) // 2
        med = others[mid] if len(others) % 2 else 0.5 * (others[mid - 1] + others[mid])
    else:
        med = 0.0
    ratio = (best_gap / med) if med > _EPS else float("inf")
    flat = bool(best_gap < flat_abs_tol or ratio < flat_ratio_threshold)
    return best_k, best_gap, flat, gaps, (ratio if ratio != float("inf") else float("inf")), cand


def nl_base_subset(conditions: Sequence[str],
                   base_condition: str = DEFAULT_BASE_CONDITION) -> List[int]:
    """Row indices of the base condition. The ONLY rows :func:`select_k` may see."""
    idx = [i for i, c in enumerate(conditions) if c == base_condition]
    if not idx:
        raise ClusterError(
            "no rows for the base condition %r; k is selected on NL-base and nowhere else "
            "(§A.6, prereg decision 1)" % (base_condition,))
    return idx


def select_k(base_states: Any, base_row_conditions: Sequence[str], *,
             base_condition: str = DEFAULT_BASE_CONDITION,
             affinity: str = COSINE, kernel: str = KERNEL_GAUSSIAN,
             k_min: int = DEFAULT_K_MIN, k_max: int = DEFAULT_K_MAX,
             seed: int = 0, knn_sigma: int = 7,
             n_landmarks: Optional[int] = None,
             exact_max_rows: int = DEFAULT_EXACT_MAX_ROWS,
             flat_ratio_threshold: float = 1.5,
             flat_abs_tol: float = 1e-4,
             sigma: Optional[float] = None) -> KSelection:
    """Select *k* at one layer from the BASE CONDITION'S ROWS ONLY.

    ``base_row_conditions`` is required and is checked element-by-element: every
    row handed in must belong to ``base_condition``. That check is the point of
    the signature. §A.6 and prereg decision 1 require *k* to be chosen once on
    NL-base and reused for all four realizations, and the failure mode is not
    "someone writes ``k = select_k(pooled)`` on purpose", it is that a pooled
    matrix is the thing lying around in the caller's scope.

    On the landmark question: when the base spectrum has to be approximated
    (``n_base > exact_max_rows``) the landmarks are drawn from the base rows,
    because those are the only rows in scope here. That is not the §A.11 item 6
    violation — item 6 forbids per-condition landmarks for the CLUSTERING that is
    compared across conditions, which is :func:`apply_clustering` and which does
    take pooled landmarks. k-selection is single-condition by pre-registration.
    Prefer the exact path anyway; it is affordable at corpus scale.
    """
    conds = list(base_row_conditions)
    X_all = as_matrix(base_states)
    n, _ = shape(X_all)
    if len(conds) != n:
        raise ClusterError(
            "select_k got %d rows but %d condition labels; it cannot verify that these "
            "are nl_base rows, and an unverified k-selection set is exactly the §A.11 "
            "item 4 bug." % (n, len(conds)))
    bad = sorted({c for c in conds if c != base_condition})
    if bad:
        raise ClusterError(
            "select_k was handed rows from condition(s) %r as well as %r. k is selected "
            "on NL-base ONLY and then reused for all four realizations (§A.6, prereg "
            "decision 1); selecting it on the pooled set makes k a function of the "
            "conditions it is supposed to be held constant across. Use "
            "nl_base_subset(conditions) to slice first." % (bad, base_condition))
    if n < k_min + 2:
        raise ClusterError("only %d base rows: too few to select k in [%d, %d]"
                           % (n, k_min, k_max))
    if np is None and n > PURE_PY_MAX_ROWS:
        raise ClusterError(
            "numpy is not installed and select_k refuses %d rows on the pure-python "
            "backend (limit %d). Install numpy." % (n, PURE_PY_MAX_ROWS))

    X = prepare_states(X_all, affinity)
    n_eigs = min(n, k_max + 2)

    use_exact = n <= exact_max_rows
    if use_exact:
        ref = list(range(n))
    else:
        m = int(n_landmarks or max(4 * (k_max + 2), min(n, 512)))
        ref = sample_landmarks(n, m, seed)
    sig = float(sigma) if sigma is not None else choose_sigma(X, ref, knn_sigma=knn_sigma)

    if use_exact:
        A = kernel_block(X, X, sig, kernel=kernel)
        lap, _ = normalized_laplacian_spectrum(A, n_eigs=n_eigs)
        method = "exact"
        note = ""
    else:
        lap, _ = nystrom_spectral(X, ref, sig, kernel=kernel, n_eigs=n_eigs)
        method = "nystrom"
        note = ("base spectrum approximated by Nystrom with %d landmarks drawn from the "
                "%d nl_base rows; k-selection is single-condition by pre-registration, "
                "so this is not the §A.11 item 6 pooled-landmark rule" % (len(ref), n))

    k, gap, flat, gaps, ratio, cand = eigengap_k(
        lap, k_min=k_min, k_max=k_max,
        flat_ratio_threshold=flat_ratio_threshold, flat_abs_tol=flat_abs_tol)
    n_zero = sum(1 for v in lap if v < 1e-8)
    if n_zero > 1:
        note = (note + "; " if note else "") + (
            "%d near-zero Laplacian eigenvalues: the base affinity graph is "
            "disconnected, so the eigengap is reporting components, not clusters" % n_zero)
    return KSelection(
        k=int(k), eigengap=float(gap), eigengap_flat=bool(flat),
        condition=base_condition, n_rows=n, k_min=int(k_min), k_max=int(cand[-1]),
        laplacian_eigenvalues=tuple(float(v) for v in lap),
        gaps=tuple(float(g) for g in gaps), flat_ratio=float(ratio),
        method=method, sigma=sig, n_zero_eigenvalues=int(n_zero), note=note)


# --------------------------------------------------------------------------------------
# 5. k-means  (H1's comparator, and the final step of spectral clustering)
# --------------------------------------------------------------------------------------


def _kmeans_pp_init(X: Any, k: int, rng: random.Random) -> List[int]:
    n, _ = shape(X)
    first = rng.randrange(n)
    centres = [first]
    D2 = sqdist(X, take_rows(X, [first]))
    if np is not None and isinstance(D2, np.ndarray):
        best = D2[:, 0].copy()
    else:
        best = [row[0] for row in D2]
    for _ in range(1, k):
        total = float(sum(best))
        if total <= _EPS:
            remaining = [i for i in range(n) if i not in centres]
            centres.append(rng.choice(remaining) if remaining else rng.randrange(n))
        else:
            r = rng.random() * total
            acc = 0.0
            pick = n - 1
            for i in range(n):
                acc += float(best[i])
                if acc >= r:
                    pick = i
                    break
            centres.append(pick)
        Dn = sqdist(X, take_rows(X, [centres[-1]]))
        if np is not None and isinstance(Dn, np.ndarray):
            best = np.minimum(best, Dn[:, 0])
        else:
            best = [min(best[i], Dn[i][0]) for i in range(n)]
    return centres


def kmeans(X: Any, k: int, *, seed: int = 0, n_init: int = 10,
           max_iter: int = 100, tol: float = 1e-9):
    """Lloyd's algorithm with k-means++ starts. Returns ``(labels, inertia)``.

    Deterministic given ``seed``. Empty clusters are re-seeded on the point
    furthest from its centre rather than dropped, so the returned partition
    always has exactly ``k`` labels available and the schema's
    ``0 <= cluster < k`` invariant cannot be violated from below.
    """
    Xm = as_matrix(X)
    n, d = shape(Xm)
    k = int(k)
    if k < 2:
        raise ClusterError("k must be >= 2, got %r" % (k,))
    if k > n:
        raise ClusterError("k=%d exceeds the %d available rows" % (k, n))

    best_labels: Optional[List[int]] = None
    best_inertia = float("inf")
    for init in range(max(1, int(n_init))):
        rng = random.Random((seed * 1000003) ^ (init * 7919))
        cent = take_rows(Xm, _kmeans_pp_init(Xm, k, rng))
        labels = [0] * n
        prev = float("inf")
        inertia = float("inf")
        for _ in range(max_iter):
            D2 = sqdist(Xm, cent)
            if np is not None and isinstance(D2, np.ndarray):
                labels = np.argmin(D2, axis=1).tolist()
                inertia = float(D2[np.arange(n), labels].sum())
            else:
                labels = []
                inertia = 0.0
                for row in D2:
                    j = min(range(k), key=lambda t: row[t])
                    labels.append(j)
                    inertia += row[j]
            # recompute centres; re-seed empties on the worst-fit point
            members: Dict[int, List[int]] = {}
            for i, lab in enumerate(labels):
                members.setdefault(lab, []).append(i)
            if np is not None and isinstance(Xm, np.ndarray):
                new = np.zeros((k, d), dtype=np.float64)
            else:
                new = [[0.0] * d for _ in range(k)]
            if np is not None and isinstance(D2, np.ndarray):
                dist_to_own = D2[np.arange(n), labels]
                worst_order = np.argsort(-dist_to_own).tolist()
            else:
                dist_to_own = [D2[i][labels[i]] for i in range(n)]
                worst_order = sorted(range(n), key=lambda i: -dist_to_own[i])
            spare = 0
            for j in range(k):
                idx = members.get(j, [])
                if not idx:
                    while spare < n and len(members.get(labels[worst_order[spare]], [])) <= 1:
                        spare += 1
                    if spare < n:
                        pick = worst_order[spare]
                        spare += 1
                        members[labels[pick]].remove(pick)
                        labels[pick] = j
                        idx = [pick]
                        members[j] = idx
                    else:
                        idx = [j % n]
                sub = take_rows(Xm, idx)
                if np is not None and isinstance(sub, np.ndarray):
                    new[j] = sub.mean(axis=0)
                else:
                    cnt = len(idx)
                    new[j] = [sum(r[t] for r in sub) / cnt for t in range(d)]
            cent = new
            if abs(prev - inertia) <= tol * max(1.0, abs(prev)):
                break
            prev = inertia
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = list(labels)
    assert best_labels is not None
    return best_labels, float(best_inertia)


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """ARI between two label vectors of equal length. 1.0 = identical partitions."""
    if len(a) != len(b):
        raise ClusterError("ARI needs equal-length labelings (%d vs %d)" % (len(a), len(b)))
    n = len(a)
    if n < 2:
        return 1.0
    table: Dict[Tuple[Any, Any], int] = {}
    ra: Dict[Any, int] = {}
    rb: Dict[Any, int] = {}
    for x, y in zip(a, b):
        table[(x, y)] = table.get((x, y), 0) + 1
        ra[x] = ra.get(x, 0) + 1
        rb[y] = rb.get(y, 0) + 1

    def c2(v: int) -> float:
        return v * (v - 1) / 2.0

    index = sum(c2(v) for v in table.values())
    sa = sum(c2(v) for v in ra.values())
    sb = sum(c2(v) for v in rb.values())
    total = c2(n)
    expected = sa * sb / total if total > 0 else 0.0
    maximum = 0.5 * (sa + sb)
    denom = maximum - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(index - expected) < 1e-12 else 0.0
    return float((index - expected) / denom)


# --------------------------------------------------------------------------------------
# 6. Applying a FIXED k to the pooled set
# --------------------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """One layer's pooled partition, plus everything needed to audit it."""

    labels: List[int]
    k: int
    affinity: str
    kernel: str
    sigma: float
    method: str
    n_landmarks: Optional[int]
    laplacian_eigenvalues: List[float]
    seed: int
    kmeans_labels: Optional[List[int]] = None
    kmeans_inertia: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "k": self.k,
            "affinity": self.affinity,
            "kernel": self.kernel,
            "sigma": self.sigma,
            "method": self.method,
            "n_landmarks": self.n_landmarks,
            "laplacian_kind": LAPLACIAN_KIND,
            "laplacian_eigenvalues": list(self.laplacian_eigenvalues),
            "seed": self.seed,
        }
        if self.kmeans_inertia is not None:
            out["kmeans_inertia"] = self.kmeans_inertia
        out.update(self.extra)
        return out


def apply_clustering(states: Any, k: int, *, affinity: str = COSINE,
                     kernel: str = KERNEL_GAUSSIAN, seed: int = 0,
                     landmark_idx: Optional[Sequence[int]] = None,
                     conditions: Optional[Sequence[str]] = None,
                     n_landmarks: Optional[int] = None,
                     exact_max_rows: int = DEFAULT_EXACT_MAX_ROWS,
                     knn_sigma: int = 7, sigma: Optional[float] = None,
                     n_init: int = 10,
                     with_kmeans: bool = True) -> ClusterResult:
    """Cluster the POOLED rows at one layer into exactly ``k`` clusters.

    ``k`` is a required parameter and this function contains no code that could
    produce one: no eigengap search, no k range, no heuristic. It is handed the
    ``k`` that :func:`select_k` chose on nl_base and it applies it (§A.6, prereg
    decision 1, §A.11 item 4). If it could compute a k it would eventually be
    asked to.

    Landmarks: pass ``landmark_idx`` (indices into the POOLED rows) or let this
    function sample them with :func:`sample_landmarks`, which requires the pooled
    row count and — when ``conditions`` is supplied — verifies that the sample
    spans every condition present (§A.11 item 6).
    """
    k = int(k)
    if k < 2:
        raise ClusterError("k must be an int >= 2, got %r" % (k,))
    X_raw = as_matrix(states)
    n, _ = shape(X_raw)
    if k > n:
        raise ClusterError("k=%d exceeds the %d pooled rows at this layer" % (k, n))
    if conditions is not None and len(conditions) != n:
        raise ClusterError("conditions has %d entries for %d pooled rows"
                           % (len(conditions), n))
    if np is None and n > PURE_PY_MAX_ROWS:
        raise ClusterError(
            "numpy is not installed and apply_clustering refuses %d rows on the "
            "pure-python backend (limit %d). Install numpy (requirements.txt)."
            % (n, PURE_PY_MAX_ROWS))

    X = prepare_states(X_raw, affinity)

    use_exact = (landmark_idx is None) and (n <= exact_max_rows)
    if use_exact:
        ref = list(range(n))
        lm: Optional[List[int]] = None
    else:
        if landmark_idx is not None:
            lm = [int(i) for i in landmark_idx]
            if not lm:
                raise ClusterError("landmark_idx is empty")
            if min(lm) < 0 or max(lm) >= n:
                raise ClusterError("landmark index out of range for %d pooled rows" % n)
            if len(set(lm)) != len(lm):
                raise ClusterError("landmark_idx contains duplicates")
            if conditions is not None:
                present = set(conditions)
                covered = {conditions[i] for i in lm}
                if len(present) > 1 and covered != present:
                    raise ClusterError(
                        "landmark set covers conditions %r but the pooled set contains "
                        "%r — §A.11 item 6 requires landmarks from the POOLED set."
                        % (sorted(covered), sorted(present)))
        else:
            m = int(n_landmarks or max(4 * k, min(n, 512)))
            lm = sample_landmarks(n, m, seed, conditions=conditions)
        ref = lm

    sig = float(sigma) if sigma is not None else choose_sigma(X, ref, knn_sigma=knn_sigma)

    if lm is None:
        A = kernel_block(X, X, sig, kernel=kernel)
        lap, V = normalized_laplacian_spectrum(A, n_eigs=k)
        method = "exact"
    else:
        lap, V = nystrom_spectral(X, lm, sig, kernel=kernel, n_eigs=k)
        method = "nystrom"

    embed = row_l2_normalize(V)                       # Ng-Jordan-Weiss row normalization
    labels, _ = kmeans(embed, k, seed=seed, n_init=n_init)

    km_labels = None
    km_inertia = None
    if with_kmeans:
        # H1's comparator: same k, same layer, same token states -- run on the SAME
        # prepared representation the affinity is built from, so the comparison is
        # spectral-vs-centroid and not "one method got the depth-norm artifact".
        km_labels, km_inertia = kmeans(X, k, seed=seed, n_init=n_init)

    return ClusterResult(
        labels=[int(v) for v in labels], k=k, affinity=affinity, kernel=kernel,
        sigma=sig, method=method, n_landmarks=(len(lm) if lm is not None else None),
        laplacian_eigenvalues=[float(v) for v in lap], seed=seed,
        kmeans_labels=([int(v) for v in km_labels] if km_labels is not None else None),
        kmeans_inertia=km_inertia,
        extra={"kmeans_input": "prepared_states" if with_kmeans else None})


# --------------------------------------------------------------------------------------
# 7. ARI stability across seeds  (§A.12, hours 8-10)
# --------------------------------------------------------------------------------------


def ari_stability(states: Any, k: int, *, seeds: Sequence[int] = (0, 1, 2),
                  affinity: str = COSINE, kernel: str = KERNEL_GAUSSIAN,
                  conditions: Optional[Sequence[str]] = None,
                  n_landmarks: Optional[int] = None,
                  exact_max_rows: int = DEFAULT_EXACT_MAX_ROWS,
                  knn_sigma: int = 7, n_init: int = 10) -> Dict[str, Any]:
    """Re-cluster at the SAME fixed k under different seeds; report pairwise ARI.

    Both sources of seed dependence are exercised: the landmark sample and the
    k-means step on the spectral embedding. ``k`` is held fixed across seeds —
    re-selecting it per seed would measure k-selection variance, which is a
    different (also interesting, not pre-registered) quantity.
    """
    seeds = [int(s) for s in seeds]
    if len(seeds) < 2:
        raise ClusterError("ARI stability needs at least 2 seeds")
    runs = []
    for s in seeds:
        res = apply_clustering(states, k, affinity=affinity, kernel=kernel, seed=s,
                               conditions=conditions, n_landmarks=n_landmarks,
                               exact_max_rows=exact_max_rows, knn_sigma=knn_sigma,
                               n_init=n_init, with_kmeans=False)
        runs.append(res.labels)
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairs.append({"seeds": [seeds[i], seeds[j]],
                          "ari": adjusted_rand_index(runs[i], runs[j])})
    vals = [p["ari"] for p in pairs]
    return {
        "k": int(k),
        "seeds": seeds,
        "pairwise": pairs,
        "mean_ari": float(sum(vals) / len(vals)),
        "min_ari": float(min(vals)),
    }


# --------------------------------------------------------------------------------------
# 8. The per-layer entry point
# --------------------------------------------------------------------------------------


@dataclass
class LayerClustering:
    """Everything one layer contributes to assignments.jsonl."""

    layer: int
    selection: KSelection
    result: ClusterResult
    n_rows: int
    conditions: Tuple[str, ...]
    clusters_used_by_condition: Dict[str, int]

    @property
    def k(self) -> int:
        return self.result.k

    def record_fields(self) -> Dict[str, Any]:
        """Record-level fields for the layer's assignments.jsonl line."""
        out = {
            "layer": self.layer,
            "k": self.result.k,
            "eigengap": self.selection.eigengap,
            "eigengap_flat": self.selection.eigengap_flat,
            "affinity": self.result.affinity,
        }
        out.update({
            "kernel": self.result.kernel,
            "sigma": self.result.sigma,
            "method": self.result.method,
            "n_landmarks": self.result.n_landmarks,
            "laplacian_kind": LAPLACIAN_KIND,
            "seed": self.result.seed,
            "k_selection": self.selection.as_dict(),
            "clusters_used_by_condition": dict(self.clusters_used_by_condition),
        })
        if self.result.kmeans_inertia is not None:
            out["kmeans_inertia"] = self.result.kmeans_inertia
        return out


def cluster_layer(states: Any, conditions: Sequence[str], layer: int, *,
                  base_condition: str = DEFAULT_BASE_CONDITION,
                  affinity: str = COSINE, kernel: str = KERNEL_GAUSSIAN,
                  k_min: int = DEFAULT_K_MIN, k_max: int = DEFAULT_K_MAX,
                  seed: int = 0, landmark_idx: Optional[Sequence[int]] = None,
                  n_landmarks: Optional[int] = None,
                  exact_max_rows: int = DEFAULT_EXACT_MAX_ROWS,
                  knn_sigma: int = 7, n_init: int = 10,
                  with_kmeans: bool = True) -> LayerClustering:
    """Select *k* on nl_base, then cluster the POOLED rows at that fixed *k*.

    The two steps are separate calls on purpose, and the k-selection call is
    handed a strict subset of the rows. §A.11 item 4 is the checklist entry this
    function exists to make unfailable.
    """
    conds = list(conditions)
    X = as_matrix(states)
    n, _ = shape(X)
    if len(conds) != n:
        raise ClusterError("got %d rows and %d condition labels" % (n, len(conds)))

    base_idx = nl_base_subset(conds, base_condition)
    selection = select_k(
        take_rows(X, base_idx), [conds[i] for i in base_idx],
        base_condition=base_condition, affinity=affinity, kernel=kernel,
        k_min=k_min, k_max=k_max, seed=seed, knn_sigma=knn_sigma,
        exact_max_rows=exact_max_rows)

    result = apply_clustering(
        X, selection.k, affinity=affinity, kernel=kernel, seed=seed,
        landmark_idx=landmark_idx, conditions=conds, n_landmarks=n_landmarks,
        exact_max_rows=exact_max_rows, knn_sigma=knn_sigma, n_init=n_init,
        with_kmeans=with_kmeans)

    used: Dict[str, set] = {}
    for lab, cond in zip(result.labels, conds):
        used.setdefault(cond, set()).add(lab)
    for cond, labs in used.items():
        if len(labs) > result.k:
            raise ClusterError(
                "layer %d condition %r uses %d distinct clusters > k=%d"
                % (layer, cond, len(labs), result.k))
        if labs and (min(labs) < 0 or max(labs) >= result.k):
            raise ClusterError(
                "layer %d condition %r has a cluster label outside [0, k=%d)"
                % (layer, cond, result.k))

    return LayerClustering(
        layer=int(layer), selection=selection, result=result, n_rows=n,
        conditions=tuple(sorted(set(conds))),
        clusters_used_by_condition={c: len(v) for c, v in sorted(used.items())})
