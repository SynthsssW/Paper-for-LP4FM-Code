"""Laptop-side metrics over `results/<run_id>/assignments.jsonl`.

This module reads the ONLY artifact the GPU pod produces (configs/assignments_schema.md)
and computes the three retained metrics of DESIGN.md §A.8:

    lambda2   -- algebraic connectivity of the POOLED cluster-transition graph
    path entropy -- entropy of the cluster-transition chain, in [0, log k]
    fork layer   -- first layer at which two conditions' clusters become disjoint,
                    RIGHT-CENSORED when they never do

plus the correlation matrix across those three that §A.8 requires ("a reviewer from the
GNN-rewiring world assumes redundancy unless shown otherwise"), and the style-normalized
displacement of DEVIATIONS Amendment 6 (INTERPRETIVE SECONDARY; a guarded ratio that
refuses to report itself through a near-zero denominator).

No torch. No activations. numpy is used when importable, purely for speed; every
computation has a pure-Python fallback so the analysis runs on a laptop with nothing
installed.

CONVENTIONS (an off-by-one here is the §A.11 item-2 failure mode)
-----------------------------------------------------------------
* `layer` indexes the OUTPUT of block `layer` (src/ling/hooks.py convention).
* Layer-pair `p` = (output of block p, output of block p+1), p in 0 .. L-2.
  L = 48 -> 47 pairs -> n = 47 for H3's Wilcoxon (DEVIATIONS Amendment 3).
* The layer-pair graph is BIPARTITE: source nodes are the clusters of layer p,
  target nodes are the clusters of layer p+1. They are distinct node sets even when
  they carry the same integer label, because cluster 3 at layer 12 and cluster 3 at
  layer 13 are unrelated objects (spectral clustering labels are arbitrary per layer).
* lambda2 is POOLED over problems (DEVIATIONS Amendment 2). A per-problem graph has
  ~4-6 edges over 2k nodes and is degenerate, not merely noisy.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence, Union

try:  # numpy is a speed path only; every code path below works without it.
    import numpy as _np
except Exception:  # pragma: no cover - exercised on machines without numpy
    _np = None

__all__ = [
    "Assignments",
    "load_assignments",
    "TransitionGraph",
    "transition_graph",
    "lambda2",
    "path_entropy",
    "style_normalized_displacement",
    "snd_range",
    "SND_METRICS",
    "SND_STATUS",
    "ForkResult",
    "fork_layer",
    "fork_layers",
    "metric_correlation_matrix",
    "HAVE_NUMPY",
]

HAVE_NUMPY = _np is not None

VALID_AFFINITIES = ("cosine", "per_layer_standardized")


class InvariantViolation(AssertionError):
    """A schema invariant of configs/assignments_schema.md was violated.

    This is deliberately a hard failure and never a warning. Every invariant this
    module asserts is one whose violation silently produces plausible-looking wrong
    numbers with nothing in the output to reveal it.
    """


# --------------------------------------------------------------------------------------
# Loading + invariants
# --------------------------------------------------------------------------------------


class Assignments:
    """In-memory view of `assignments.jsonl` with every schema invariant asserted.

    Attributes
    ----------
    layers : tuple[int]           sorted layer indices present
    keys   : tuple[(problem_id, condition, anchor)]   row identity, SHARED by all layers
    clusters : dict[int, tuple[int]]   layer -> per-row cluster label, aligned with `keys`
    k_by_layer : dict[int, int]
    eigengap_by_layer : dict[int, float|None]
    problems : tuple[str]         sorted distinct problem ids
    conditions : tuple[str]       sorted distinct condition names
    """

    def __init__(self, records: Sequence[Mapping], source: Optional[str] = None):
        if not records:
            raise InvariantViolation("assignments.jsonl is empty: no layer records")
        self.source = source
        self._records = list(records)
        self._validate()

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str) -> "Assignments":
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError as exc:
                    raise InvariantViolation(
                        "%s:%d is not valid JSON: %s" % (path, lineno, exc)
                    )
        return cls(records, source=path)

    @classmethod
    def coerce(cls, obj: Union["Assignments", str, Sequence[Mapping]]) -> "Assignments":
        """Accept an Assignments, a path to assignments.jsonl, or a list of records."""
        if isinstance(obj, Assignments):
            return obj
        if isinstance(obj, str):
            return cls.from_jsonl(obj)
        return cls(obj)

    # -- invariants ---------------------------------------------------------------

    def _validate(self) -> None:
        recs = self._records
        seen_layers = set()
        keys = None
        self.clusters = {}
        self.k_by_layer = {}
        self.eigengap_by_layer = {}
        self.affinity_by_layer = {}

        for i, rec in enumerate(recs):
            where = "record %d" % i
            for field in ("layer", "k", "rows"):
                if field not in rec:
                    raise InvariantViolation("%s: missing required field %r" % (where, field))
            layer = rec["layer"]
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise InvariantViolation("%s: `layer` must be an int, got %r" % (where, layer))
            if layer in seen_layers:
                raise InvariantViolation("duplicate record for layer %d" % layer)
            seen_layers.add(layer)
            where = "layer %d" % layer

            k = rec["k"]
            if not isinstance(k, int) or isinstance(k, bool) or k < 2:
                raise InvariantViolation("%s: `k` must be an int >= 2, got %r" % (where, k))

            affinity = rec.get("affinity")
            if affinity is not None:
                if affinity == "raw":
                    raise InvariantViolation(
                        "%s: affinity 'raw' is forbidden (DEVIATIONS Amendment 3); it "
                        "confounds depth with geometry" % where
                    )
                if affinity not in VALID_AFFINITIES:
                    raise InvariantViolation(
                        "%s: affinity %r is not one of %r" % (where, affinity, VALID_AFFINITIES)
                    )

            rows = rec["rows"]
            if not isinstance(rows, list) or not rows:
                raise InvariantViolation("%s: `rows` must be a non-empty list" % where)

            row_keys = []
            row_clusters = []
            # k identical across conditions at a layer: `k` is a record-level field, so a
            # per-condition k can only show up as (a) a k carried on a row / condition
            # block, or (b) a condition whose cluster labels leave [0, k). Both are
            # checked. A per-condition k invalidates EVERY cross-condition lambda2
            # comparison (configs/assignments_schema.md), so it is fatal.
            used_by_condition = {}
            for j, row in enumerate(rows):
                rwhere = "%s row %d" % (where, j)
                for field in ("problem_id", "condition", "anchor", "cluster"):
                    if field not in row:
                        raise InvariantViolation("%s: missing field %r" % (rwhere, field))
                if "k" in row and row["k"] != k:
                    raise InvariantViolation(
                        "%s: row-level k=%r != layer k=%r. k is selected ONCE per layer on "
                        "nl_base and reused across all conditions (prereg decision 1)."
                        % (rwhere, row["k"], k)
                    )
                cluster = row["cluster"]
                if not isinstance(cluster, int) or isinstance(cluster, bool):
                    raise InvariantViolation("%s: cluster must be an int, got %r" % (rwhere, cluster))
                if not (0 <= cluster < k):
                    raise InvariantViolation(
                        "%s: cluster %d outside [0, k=%d). A condition clustered at a "
                        "different k breaks every cross-condition lambda2 comparison."
                        % (rwhere, cluster, k)
                    )
                row_keys.append((row["problem_id"], row["condition"], row["anchor"]))
                row_clusters.append(cluster)
                used_by_condition.setdefault(row["condition"], set()).add(cluster)

            for cond, used in used_by_condition.items():
                if len(used) > k:
                    raise InvariantViolation(
                        "%s: condition %r uses %d distinct clusters > k=%d"
                        % (where, cond, len(used), k)
                    )

            row_keys = tuple(row_keys)
            if len(set(row_keys)) != len(row_keys):
                dupes = sorted({x for x in row_keys if row_keys.count(x) > 1})[:3]
                raise InvariantViolation(
                    "%s: duplicate (problem_id, condition, anchor) rows, e.g. %r" % (where, dupes)
                )
            if keys is None:
                keys = row_keys
                first_layer = layer
            elif row_keys != keys:
                if len(row_keys) != len(keys):
                    detail = "row count %d != %d" % (len(row_keys), len(keys))
                else:
                    bad = next(i for i, (a, b) in enumerate(zip(row_keys, keys)) if a != b)
                    detail = "row %d is %r here but %r at layer %d" % (
                        bad, row_keys[bad], keys[bad], first_layer)
                raise InvariantViolation(
                    "%s: row order is NOT stable across layers (%s). Layer-pair transition "
                    "graphs are only well-defined if row j means the same (problem, "
                    "condition, anchor) at every layer." % (where, detail)
                )

            self.clusters[layer] = tuple(row_clusters)
            self.k_by_layer[layer] = k
            self.eigengap_by_layer[layer] = rec.get("eigengap")
            self.affinity_by_layer[layer] = affinity

        self.keys = keys
        self.layers = tuple(sorted(seen_layers))

        problems, conditions = [], []
        index_by_pc = {}
        anchors_by_pc = {}
        for j, (pid, cond, anchor) in enumerate(self.keys):
            index_by_pc.setdefault((pid, cond), []).append(j)
            anchors_by_pc.setdefault((pid, cond), []).append(anchor)
            problems.append(pid)
            conditions.append(cond)
        self.index_by_pc = {kk: tuple(v) for kk, v in index_by_pc.items()}
        self.problems = tuple(sorted(set(problems)))
        self.conditions = tuple(sorted(set(conditions)))

        # Same anchor set for every condition of a problem (DESIGN.md §A.6 invariant).
        by_problem = {}
        for (pid, cond), anchors in anchors_by_pc.items():
            by_problem.setdefault(pid, {})[cond] = frozenset(anchors)
        for pid, per_cond in by_problem.items():
            sets = set(per_cond.values())
            if len(sets) > 1:
                raise InvariantViolation(
                    "problem %r has different anchor sets across conditions: %r. §A.6 "
                    "requires every (problem, condition) to share one anchor set."
                    % (pid, {c: sorted(s) for c, s in per_cond.items()})
                )

    # -- convenience ---------------------------------------------------------------

    def layer_pairs(self) -> tuple:
        """All p such that both layer p and layer p+1 are present."""
        present = set(self.layers)
        return tuple(p for p in self.layers if (p + 1) in present)

    def rows_for(self, problem_id: str, condition: str) -> tuple:
        return self.index_by_pc.get((problem_id, condition), ())

    def clusters_at(self, layer: int) -> tuple:
        if layer not in self.clusters:
            raise InvariantViolation("layer %d is not present in %s" % (layer, self.source or "assignments"))
        return self.clusters[layer]

    def record(self, layer: int) -> Mapping:
        """The raw JSON record for `layer` (for optional fields outside the core schema)."""
        for rec in self._records:
            if rec["layer"] == layer:
                return rec
        raise InvariantViolation("layer %d is not present" % layer)

    def max_eigengap_layer(self) -> int:
        """Layer with the largest eigengap on nl_base (H1's layer, §A.6 k-selection)."""
        gaps = [(v, layer) for layer, v in self.eigengap_by_layer.items() if v is not None]
        if not gaps:
            raise InvariantViolation("no `eigengap` field on any layer record")
        return max(gaps)[1]

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        return "Assignments(layers=%d, rows=%d, problems=%d, conditions=%r)" % (
            len(self.layers), len(self.keys), len(self.problems), self.conditions)


def load_assignments(path: str) -> Assignments:
    """Load and fully validate `results/<run_id>/assignments.jsonl`."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Assignments.from_jsonl(path)


# --------------------------------------------------------------------------------------
# The pooled cluster-transition graph
# --------------------------------------------------------------------------------------


class TransitionGraph(NamedTuple):
    """Pooled bipartite cluster-transition graph for one layer-pair, one condition.

    nodes    : tuple of ("src", cluster) / ("dst", cluster), only nodes with degree > 0
    weights  : dict (i, j) -> weight, i < j, indices into `nodes`
    k_src, k_dst : the layer's k (identical across conditions by invariant)
    n_states : number of pooled token-state transitions (edge multiplicity total)
    n_problems : number of problem draws pooled (a bootstrap multiset counts repeats)
    """

    nodes: tuple
    weights: dict
    k_src: int
    k_dst: int
    n_states: int
    n_problems: int


def _condition_of(condition, problem_id):
    """`condition` may be a name, or a mapping problem_id -> name.

    The mapping form is what the H3 permutation null uses: swapping the two condition
    labels WITHIN a problem is a problem-level permutation, which is the only unit
    prereg decision 3 allows.
    """
    if isinstance(condition, str):
        return condition
    try:
        return condition[problem_id]
    except KeyError:
        raise InvariantViolation(
            "condition mapping has no entry for problem %r" % (problem_id,))


def transition_graph(
    assignments,
    layer_pair: int,
    condition,
    problems: Optional[Iterable[str]] = None,
) -> TransitionGraph:
    """Build the POOLED transition graph for layer-pair `layer_pair`.

    An edge (src cluster a at layer p) -- (dst cluster b at layer p+1) has weight equal
    to the number of token states that make that transition, pooled over `problems`
    (DEVIATIONS Amendment 2). `problems` may be a multiset (bootstrap resample); a
    problem drawn twice contributes twice, which is what "rebuild the pooled graph from
    the resampled problems" means.

    Zero-degree nodes are DROPPED. A cluster no token state of this condition visits at
    this layer-pair is absence of data, not a disconnected region of the routing graph;
    keeping it would force lambda2 = 0 for every condition at every layer-pair and
    destroy the metric. This is a judgment call and it is stated in the paper.
    """
    a = Assignments.coerce(assignments)
    p = layer_pair
    if p not in a.clusters or (p + 1) not in a.clusters:
        raise InvariantViolation(
            "layer-pair %d needs layers %d and %d; present layers are %r"
            % (p, p, p + 1, a.layers))

    if problems is None:
        problems = a.problems
    problems = list(problems)
    if not problems:
        raise ValueError("cannot build a pooled graph from zero problems")

    src = a.clusters[p]
    dst = a.clusters[p + 1]
    k_src = a.k_by_layer[p]
    k_dst = a.k_by_layer[p + 1]

    counts = {}
    n_states = 0
    for pid in problems:
        cond = _condition_of(condition, pid)
        idx = a.index_by_pc.get((pid, cond))
        if idx is None:
            raise InvariantViolation(
                "no rows for (problem %r, condition %r)" % (pid, cond))
        for j in idx:
            key = (src[j], dst[j])
            counts[key] = counts.get(key, 0) + 1
            n_states += 1

    used_src = sorted({s for s, _ in counts})
    used_dst = sorted({d for _, d in counts})
    nodes = tuple([("src", c) for c in used_src] + [("dst", c) for c in used_dst])
    pos = {n: i for i, n in enumerate(nodes)}
    weights = {}
    for (s, d), w in counts.items():
        i, j = pos[("src", s)], pos[("dst", d)]
        weights[(min(i, j), max(i, j))] = float(w)

    return TransitionGraph(nodes, weights, k_src, k_dst, n_states, len(problems))


# --------------------------------------------------------------------------------------
# Eigenvalues (pure-python fallback: cyclic Jacobi on a symmetric matrix)
# --------------------------------------------------------------------------------------


def _jacobi_eigvalsh(mat, tol=1e-12, max_sweeps=100):
    n = len(mat)
    m = [list(map(float, row)) for row in mat]
    for _ in range(max_sweeps):
        off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                off += m[i][j] * m[i][j]
        if off <= tol * tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) < 1e-18:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * apq)
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                app, aqq = m[p][p], m[q][q]
                m[p][p] = app - t * apq
                m[q][q] = aqq + t * apq
                m[p][q] = m[q][p] = 0.0
                for r in range(n):
                    if r == p or r == q:
                        continue
                    arp, arq = m[r][p], m[r][q]
                    m[r][p] = m[p][r] = c * arp - s * arq
                    m[r][q] = m[q][r] = s * arp + c * arq
    return sorted(m[i][i] for i in range(n))


def _eigvalsh(mat):
    """Ascending eigenvalues of a real symmetric matrix (numpy if present, else Jacobi)."""
    if not mat:
        return []
    if _np is not None:
        return sorted(float(x) for x in _np.linalg.eigvalsh(_np.asarray(mat, dtype=float)))
    return _jacobi_eigvalsh(mat)


def _n_components(n_nodes, weights):
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), w in weights.items():
        if w <= 0:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(x) for x in range(n_nodes)})


def _laplacian(graph: TransitionGraph, kind: str):
    n = len(graph.nodes)
    deg = [0.0] * n
    for (i, j), w in graph.weights.items():
        deg[i] += w
        deg[j] += w
    if kind == "combinatorial":
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = deg[i]
        for (i, j), w in graph.weights.items():
            L[i][j] -= w
            L[j][i] -= w
        return L
    if kind != "normalized":
        raise ValueError("laplacian must be 'normalized' or 'combinatorial', got %r" % kind)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0 if deg[i] > 0 else 0.0
    for (i, j), w in graph.weights.items():
        v = w / math.sqrt(deg[i] * deg[j])
        L[i][j] -= v
        L[j][i] -= v
    return L


def lambda2(
    assignments,
    layer_pair: int,
    condition,
    problems: Optional[Iterable[str]] = None,
    laplacian: str = "normalized",
) -> float:
    """Second-smallest Laplacian eigenvalue of the pooled cluster-transition graph.

    LAPLACIAN CHOICE (DESIGN.md §A.11 item 5 calls for this to be stated and held to):
    the **symmetric NORMALIZED** Laplacian

        L_sym = I - D^{-1/2} W D^{-1/2}

    is the default and is the quantity reported everywhere in this project. The reason
    is that the pre-committed reading of lambda2 in prereg/PRIMARY_ENDPOINTS.md is the
    Cheeger one -- "algebraic connectivity is LOW when a good cut exists" -- and the
    Cheeger inequality h^2/2 <= lambda2 <= 2h is a statement about the NORMALIZED
    Laplacian. Its spectrum also lies in [0, 2] regardless of how many token states were
    pooled or how large k is at that layer, so per-layer-pair values are comparable
    along the exact axis the H3 primary endpoint averages over. The combinatorial
    Laplacian D - W is available as `laplacian="combinatorial"` for a robustness check
    only; its eigenvalues scale with edge weight and with k and are NOT comparable
    across layers.

    The trivial lambda1 = 0 (constant eigenvector) is skipped: this returns the
    SECOND-smallest eigenvalue, index 1 of the ascending spectrum.

    DISCONNECTED GRAPHS: a graph with c >= 2 connected components has exactly c zero
    eigenvalues, so lambda2 = 0. That is a REAL VALUE, not an error and not a missing
    value -- it says the condition's routing at this layer-pair splits into groups with
    no shared cluster, which is the strongest possible version of the H3 effect. It is
    returned as 0.0, computed combinatorially from the component count rather than read
    off a numerically noisy eigensolve. Downstream log-transforms (prereg decision 6)
    must handle it; see `ling.stats.log_lambda2`.

    Zero-degree nodes are dropped when the graph is built; see `transition_graph`.
    """
    g = transition_graph(assignments, layer_pair, condition, problems)
    n = len(g.nodes)
    if n < 2:
        raise InvariantViolation(
            "layer-pair %d, condition %r: pooled graph has %d node(s); lambda2 is "
            "undefined. This means the condition contributed no token states."
            % (layer_pair, condition, n))
    if _n_components(n, g.weights) > 1:
        return 0.0
    ev = _eigvalsh(_laplacian(g, laplacian))
    return max(0.0, float(ev[1]))


# --------------------------------------------------------------------------------------
# Path entropy
# --------------------------------------------------------------------------------------


def path_entropy(
    assignments,
    layer_pair: int,
    condition,
    problems: Optional[Iterable[str]] = None,
    normalized: bool = False,
) -> float:
    """Entropy of the cluster-transition chain for one layer-pair, in [0, log k].

    DEFINITION. Over the pooled transitions (a at layer p) -> (b at layer p+1), this is
    the CONDITIONAL entropy of the next cluster given the current one,

        H = - sum_a P(a) sum_b P(b|a) log P(b|a)

    i.e. the one-step entropy rate of the transition chain. The conditional form is what
    makes the bound in configs/assignments_schema.md correct: H lies in
    [0, log k_{p+1}], attained at 0 when every source cluster routes deterministically
    and at log k when every source spreads uniformly over all k target clusters. The
    JOINT transition entropy would be bounded by 2 log k instead, and would confound
    "how spread out are the transitions" with "how spread out is the source layer".

    Natural log. `normalized=True` divides by log k_{p+1}, giving a value in [0, 1],
    which is the support §A.9.2's Beta regression needs.

    Pooled over problems, per DEVIATIONS Amendment 2 (same structural reason as
    lambda2: a single problem contributes ~4-6 transitions).
    """
    a = Assignments.coerce(assignments)
    p = layer_pair
    if p not in a.clusters or (p + 1) not in a.clusters:
        raise InvariantViolation("layer-pair %d not available; layers present: %r" % (p, a.layers))
    if problems is None:
        problems = a.problems
    problems = list(problems)
    if not problems:
        raise ValueError("cannot compute pooled path entropy from zero problems")

    src, dst = a.clusters[p], a.clusters[p + 1]
    joint = {}
    total = 0
    for pid in problems:
        cond = _condition_of(condition, pid)
        idx = a.index_by_pc.get((pid, cond))
        if idx is None:
            raise InvariantViolation("no rows for (problem %r, condition %r)" % (pid, cond))
        for j in idx:
            joint[(src[j], dst[j])] = joint.get((src[j], dst[j]), 0) + 1
            total += 1
    if total == 0:
        raise InvariantViolation(
            "layer-pair %d, condition %r: no token states" % (p, condition))

    row_total = {}
    for (s, _), c in joint.items():
        row_total[s] = row_total.get(s, 0) + c

    h = 0.0
    for (s, _), c in joint.items():
        p_ba = c / row_total[s]
        h -= (c / total) * math.log(p_ba)

    if normalized:
        k = a.k_by_layer[p + 1]
        return h / math.log(k) if k > 1 else 0.0
    return h


# --------------------------------------------------------------------------------------
# Style-normalized displacement (DEVIATIONS Amendment 6) -- INTERPRETIVE SECONDARY
# --------------------------------------------------------------------------------------


SND_STATUS = (
    "INTERPRETIVE SECONDARY (DEVIATIONS Amendment 6). The PRIMARY quantity is the raw "
    "difference m(c) - m(base) with a problem-level bootstrap CI, reported here as "
    "`raw_difference`. SND divides that by a second, also-estimated difference, and is "
    "reported ONLY where the denominator's bootstrap CI excludes zero."
)

SND_METRICS = ("lambda2", "path_entropy")


def _metric_value(assignments, metric, layer_pair, condition, problems, normalized_pe):
    if metric == "lambda2":
        return lambda2(assignments, layer_pair, condition, problems=problems)
    if metric == "path_entropy":
        return path_entropy(assignments, layer_pair, condition, problems=problems,
                            normalized=normalized_pe)
    raise ValueError("metric must be one of %r, got %r" % (SND_METRICS, metric))


def _mean_displacement(assignments, metric, condition, base_condition, pairs, problems,
                       normalized_pe):
    """Mean over layer-pairs of m(condition) - m(base_condition), pooled over `problems`.

    Same shape as H3's primary statistic: the metric is POOLED over problems at each
    layer-pair (Amendment 2), and the average runs over LAYER-PAIRS (Amendment 3).
    """
    total = 0.0
    for p in pairs:
        total += (_metric_value(assignments, metric, p, condition, problems, normalized_pe)
                  - _metric_value(assignments, metric, p, base_condition, problems,
                                  normalized_pe))
    return total / len(pairs)


def style_normalized_displacement(
    assignments,
    metric: str,
    condition: str,
    floor_condition: str,
    base_condition: str = "nl_base",
    problems: Optional[Iterable[str]] = None,
    layer_pairs: Optional[Sequence[int]] = None,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    normalized_path_entropy: bool = True,
    precommitted: Optional[str] = None,
    cache: Optional[dict] = None,
) -> dict:
    """SND against ONE style floor (DEVIATIONS Amendment 6), with the ratio guard.

        SND(c, m) = [ m(c) - m(base) ] / [ m(floor) - m(base) ]

    both terms being the mean over layer-pairs of the pooled metric difference -- the
    same statistic H3's primary endpoint uses, so the numerator IS the primary raw
    difference and is returned as `raw_difference` with its own problem-level bootstrap
    CI.

    THE GUARD IS THE POINT, AND IT LIVES HERE RATHER THAN IN THE CALLER. A ratio whose
    denominator is a small estimated difference is not a large effect, it is a division
    artifact: as the denominator approaches zero the ratio diverges, and it diverges
    most enthusiastically exactly when the style floor is least well measured. This
    function therefore bootstraps the DENOMINATOR over PROBLEMS and computes the ratio
    at all only inside the branch where that CI excludes zero. Where it does not, `snd`
    is None and `undefined_reason` says why. There is no code path in which a
    near-zero denominator produces a number, no keyword that disables the guard, and no
    fallback value: a caller that ignores the guard gets None, not a big float.

    `n_boot = 0` does not skip the guard, it fails it -- with no interval there is no
    evidence the denominator is nonzero, so SND stays undefined.

    `precommitted="abs_gt_1"` records Amendment 6's one directional pre-commitment for
    this metric (|SND(formal)| > 1 for lambda2 against the UPPER bracket) and evaluates
    it; when the guard fires the evaluation is None, not False -- an undefined ratio is
    not evidence against the pre-commitment.
    """
    from . import stats as _stats  # local: keeps `metrics` importable with no cycle risk

    a = Assignments.coerce(assignments)
    for name in (condition, floor_condition, base_condition):
        if name not in a.conditions:
            raise InvariantViolation(
                "condition %r is not present in the assignments (present: %r)"
                % (name, list(a.conditions)))
    if metric not in SND_METRICS:
        raise ValueError("metric must be one of %r, got %r" % (SND_METRICS, metric))
    if precommitted not in (None, "abs_gt_1"):
        raise ValueError("precommitted must be None or 'abs_gt_1', got %r" % precommitted)
    pairs = tuple(layer_pairs) if layer_pairs is not None else a.layer_pairs()
    if not pairs:
        raise ValueError("no layer-pairs available")
    ids = sorted(a.problems) if problems is None else sorted(set(problems))

    def numerator(problem_multiset):
        return _mean_displacement(a, metric, condition, base_condition, pairs,
                                  problem_multiset, normalized_path_entropy)

    def denominator(problem_multiset):
        return _mean_displacement(a, metric, floor_condition, base_condition, pairs,
                                  problem_multiset, normalized_path_entropy)

    num_point = numerator(list(ids))
    den_point = denominator(list(ids))

    def _boot(role, other_condition, stat, boot_seed):
        # `cache` is a caller-supplied memo across floors and target conditions: the
        # numerator bootstrap is identical for every floor and the denominator
        # bootstrap for every target condition, and each draw rebuilds every pooled
        # graph. Recomputing them is only expensive, never different.
        key = (role, metric, other_condition, base_condition, normalized_path_entropy,
               n_boot, alpha, boot_seed, tuple(pairs), tuple(ids))
        if cache is not None and key in cache:
            return cache[key]
        res = _stats.bootstrap_ci(ids, stat, n_boot=n_boot, alpha=alpha, seed=boot_seed)
        if cache is not None:
            cache[key] = res
        return res

    num_ci = None
    den_ci = None
    snd = None
    reason = None
    guard_passed = False
    if n_boot > 0:
        num_ci = _boot("num", condition, numerator, seed)
        den_ci = _boot("den", floor_condition, denominator, seed + 1)
        guard_passed = den_ci["ci_low"] > 0.0 or den_ci["ci_high"] < 0.0
        if guard_passed and den_point != 0.0:
            snd = num_point / den_point
        elif guard_passed:
            guard_passed = False
            reason = (
                "the denominator's bootstrap CI excludes zero but its point estimate is "
                "exactly 0.0; the ratio is undefined and is NOT reported as a number.")
        else:
            reason = (
                "the denominator m(%s) - m(%s) = %.6g has a %d%% problem-level bootstrap "
                "CI [%.6g, %.6g] that INCLUDES zero, so the style floor is not "
                "distinguishable from no displacement at all. SND is undefined here. It "
                "is not reported as a large number: a ratio through a near-zero "
                "denominator is a division artifact, not an effect (Amendment 6)."
                % (floor_condition, base_condition, den_point,
                   int(round((1 - alpha) * 100)), den_ci["ci_low"], den_ci["ci_high"]))
    else:
        reason = (
            "n_boot = 0, so the denominator has no bootstrap CI and the Amendment 6 "
            "guard cannot be satisfied. SND is undefined by construction rather than "
            "computed unguarded.")

    out = {
        "status": SND_STATUS,
        "metric": metric,
        "condition": condition,
        "floor_condition": floor_condition,
        "base_condition": base_condition,
        "path_entropy_normalized": (normalized_path_entropy if metric == "path_entropy"
                                    else None),
        "raw_difference": {
            "note": ("PRIMARY quantity: m(%s) - m(%s), mean over layer-pairs, with a "
                     "problem-level bootstrap CI." % (condition, base_condition)),
            "point": num_point,
            "bootstrap_ci": num_ci,
        },
        "denominator": {
            "note": "m(%s) - m(%s), mean over layer-pairs." % (floor_condition,
                                                               base_condition),
            "point": den_point,
            "bootstrap_ci": den_ci,
        },
        "denominator_ci_excludes_zero": guard_passed,
        "snd": snd,
        "undefined_reason": reason,
        "n_layer_pairs": len(pairs),
        "n_problems": len(ids),
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
        "resampling_unit": "problem",
    }
    if precommitted == "abs_gt_1":
        out["direction_precommitted"] = (
            "|SND| > 1 (Amendment 6): %s displaces the geometry more than a "
            "semantically-null edit of the same kind does" % condition)
        out["consistent_with_precommitted_direction"] = (
            None if snd is None else abs(snd) > 1.0)
    return out


def snd_range(
    assignments,
    metric: str,
    condition: str,
    upper_floor: str = "nl_minimal_orth",
    lower_floor: str = "nl_minimal_lex",
    base_condition: str = "nl_base",
    problems: Optional[Iterable[str]] = None,
    layer_pairs: Optional[Sequence[int]] = None,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    normalized_path_entropy: bool = True,
    precommitted_upper: Optional[str] = None,
    cache: Optional[dict] = None,
) -> dict:
    """SND against BOTH style floors, reported as a RANGE (Amendment 6).

    Neither floor is trustworthy alone and Amendment 6 says so explicitly:
    `nl_minimal_orth` re-renders a quantity as a number word ON the anchor, which may
    be a genuinely different encoding and so gives a floor that is likely too HIGH;
    `nl_minimal_lex` swaps a proper name, which is semantically null but OFF-anchor and
    so gives a floor that is likely too LOW. They BRACKET the true style floor, which is
    the reason both exist. A single-floor SND must never be quoted as "the" SND.

    A floor whose guard fires contributes no endpoint of the range; the range is
    reported only when BOTH floors pass, and the surviving one-sided value is still
    shown per-floor with its reason. Which floor gives the higher SND is read off the
    numbers, not assumed.
    """
    floors = [(upper_floor, "upper bracket (likely too HIGH a floor: on-anchor "
                            "orthographic re-rendering may be genuinely differently "
                            "encoded)", precommitted_upper),
              (lower_floor, "lower bracket (likely too LOW a floor: semantically null "
                            "but OFF-anchor)", None)]
    by_floor = {}
    for name, role, pre in floors:
        block = style_normalized_displacement(
            assignments, metric, condition, name, base_condition=base_condition,
            problems=problems, layer_pairs=layer_pairs, n_boot=n_boot, alpha=alpha,
            seed=seed, normalized_path_entropy=normalized_path_entropy,
            precommitted=pre, cache=cache)
        block["bracket_role"] = role
        by_floor[name] = block

    defined = {n: b["snd"] for n, b in by_floor.items() if b["snd"] is not None}
    rng = None
    if len(defined) == 2:
        vals = sorted(defined.values())
        rng = {"low": vals[0], "high": vals[1],
               "note": ("SND is a RANGE bracketed by the two style floors, not a point "
                        "estimate. Report it as a range (Amendment 6).")}
    return {
        "status": SND_STATUS,
        "metric": metric,
        "condition": condition,
        "base_condition": base_condition,
        "by_floor": by_floor,
        "range": rng,
        "range_available": rng is not None,
        "range_unavailable_reason": None if rng is not None else (
            "SND is undefined against %s; a range needs both floors."
            % ", ".join(sorted(n for n, b in by_floor.items() if b["snd"] is None))),
    }


# --------------------------------------------------------------------------------------
# Fork layer (right-censored)
# --------------------------------------------------------------------------------------


class ForkResult(NamedTuple):
    """Fork layer for one (problem, condition-pair), with EXPLICIT censoring.

    layer    : the first layer at which the two conditions' cluster sets are disjoint,
               or None when they never are.
    censored : True iff `layer is None`. A never-forking pair is right-censored; it is
               NEITHER dropped NOR given a fabricated event layer (DESIGN.md §A.9.1).
               There is deliberately no sentinel layer number: encoding "never forked"
               as layer 40 (or 47, or -1) is exactly the bias §A.9.1 forbids, and any
               caller that wants a number must go through the survival model.
    layers   : the ordered layers examined, up to and including the fork layer. This is
               what the person-period expansion consumes.
    """

    problem_id: str
    conditions: tuple
    layer: Optional[int]
    censored: bool
    layers: tuple


def fork_layer(assignments, problem_id: str, condition_pair, layers=None) -> ForkResult:
    """First layer at which the two conditions' cluster assignments become disjoint.

    "Disjoint" is set-level over the problem's anchors: at layer L, condition c1's rows
    for this problem occupy a set of clusters and c2's rows occupy another; the pair has
    forked at L when those two sets share no cluster. §A.6 guarantees both conditions
    have the same anchor set, so the two sets are built from the same positions.

    Returns a ForkResult; a pair that never forks is (layer=None, censored=True).
    """
    a = Assignments.coerce(assignments)
    c1, c2 = condition_pair
    i1 = a.index_by_pc.get((problem_id, c1))
    i2 = a.index_by_pc.get((problem_id, c2))
    if i1 is None or i2 is None:
        missing = c1 if i1 is None else c2
        raise InvariantViolation(
            "problem %r has no rows for condition %r" % (problem_id, missing))

    layer_seq = tuple(a.layers) if layers is None else tuple(layers)
    seen = []
    for L in layer_seq:
        cl = a.clusters_at(L)
        s1 = {cl[j] for j in i1}
        s2 = {cl[j] for j in i2}
        seen.append(L)
        if s1.isdisjoint(s2):
            return ForkResult(problem_id, (c1, c2), L, False, tuple(seen))
    return ForkResult(problem_id, (c1, c2), None, True, tuple(seen))


def fork_layers(assignments, condition_pair, problems=None, layers=None) -> list:
    """`fork_layer` for every problem; the input to the discrete-time survival model."""
    a = Assignments.coerce(assignments)
    if problems is None:
        problems = a.problems
    return [fork_layer(a, pid, condition_pair, layers=layers) for pid in problems]


# --------------------------------------------------------------------------------------
# Correlation across the three retained metrics (DESIGN.md §A.8)
# --------------------------------------------------------------------------------------


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys) -> float:
    """Spearman rank correlation, average ranks for ties. Pure python."""
    return _pearson(_rank(list(xs)), _rank(list(ys)))


def metric_correlation_matrix(
    assignments,
    condition: str,
    reference_condition: str = "nl_base",
    problems: Optional[Iterable[str]] = None,
    layer_pairs: Optional[Sequence[int]] = None,
    method: str = "spearman",
) -> dict:
    """Correlation matrix across the three retained metrics (DESIGN.md §A.8).

    §A.8: "Report the correlation matrix across retained metrics... A reviewer from that
    world assumes redundancy unless shown otherwise."

    The three metrics do not share a natural unit -- lambda2 and path entropy are
    per (layer-pair, condition), fork layer is per (problem, condition-pair). The common
    index used here is the LAYER-PAIR, and fork layer enters as the cumulative FORKED
    FRACTION by the end of layer-pair p: the share of problems whose
    (reference_condition, condition) fork layer is <= p+1. Censored pairs count as
    not-yet-forked at every p, which is the censoring-safe summary; no fabricated event
    layer enters.

    This whole quantity is EXPLORATORY / descriptive. It is a redundancy check on the
    metric set, not a test of anything.
    """
    a = Assignments.coerce(assignments)
    if problems is None:
        problems = a.problems
    problems = list(problems)
    pairs = tuple(layer_pairs) if layer_pairs is not None else a.layer_pairs()
    if len(pairs) < 3:
        raise ValueError("need at least 3 layer-pairs for a correlation matrix, got %d" % len(pairs))

    forks = fork_layers(a, (reference_condition, condition), problems=problems)
    fork_at = [f.layer for f in forks if not f.censored]

    l2, pe, ff = [], [], []
    for p in pairs:
        l2.append(lambda2(a, p, condition, problems=problems))
        pe.append(path_entropy(a, p, condition, problems=problems, normalized=True))
        ff.append(sum(1 for L in fork_at if L <= p + 1) / float(len(problems)))

    names = ("lambda2", "path_entropy", "forked_fraction")
    cols = (l2, pe, ff)
    corr = spearman if method == "spearman" else _pearson
    matrix = [[1.0 if i == j else corr(cols[i], cols[j]) for j in range(3)] for i in range(3)]
    return {
        "status": "EXPLORATORY",
        "metrics": list(names),
        "method": method,
        "matrix": matrix,
        "n_layer_pairs": len(pairs),
        "n_problems": len(problems),
        "condition": condition,
        "reference_condition": reference_condition,
        "n_censored_pairs": sum(1 for f in forks if f.censored),
        "note": (
            "Common index is the layer-pair. Fork layer enters as the cumulative forked "
            "fraction by the end of layer-pair p; censored pairs never count as forked."
        ),
    }
