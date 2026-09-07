"""Tests for src/ling/metrics.py, src/ling/stats.py and scripts/04_analyze.py.

The point of the eigenvalue tests below is that they check the code against
HAND-COMPUTABLE answers, not against itself. Synthetic assignments are built so that the
pooled cluster-transition graph is exactly a graph whose normalized-Laplacian spectrum is
known in closed form:

    path P_n  ->  eigenvalues 1 - cos(k*pi/(n-1)),  so lambda2(P3) = 1, lambda2(P4) = 0.5
    K_{2,2}   ->  eigenvalues 0, 1, 1, 2,           so lambda2 = 1
    two disjoint edges -> 2 components,             so lambda2 = 0 exactly

Also checked: the censoring path (a never-forking pair is (None, censored=True) and never
a sentinel layer), every schema invariant fails loudly, and the resampling unit really is
the PROBLEM -- a problem-level resample moves the statistic, a row-level reshuffle inside
a problem does not.

Run with:  python3 -m unittest discover -s tests -v
(stdlib unittest; no numpy, no scipy, no pytest required)
"""

import json
import math
import os
import random
import sys
import tempfile
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from ling import metrics as M  # noqa: E402
from ling import stats as S  # noqa: E402


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def build(rows, ks, eigengaps=None, affinity="cosine"):
    """rows: list of (problem_id, condition, anchor, [cluster_at_layer_0, ...]).

    ks: per-layer k. Returns a list of layer records in the assignments.jsonl shape.
    """
    n_layers = len(ks)
    recs = []
    for layer in range(n_layers):
        recs.append({
            "layer": layer,
            "k": ks[layer],
            "eigengap": (eigengaps or [0.1] * n_layers)[layer],
            "eigengap_flat": False,
            "affinity": affinity,
            "rows": [
                {"problem_id": p, "condition": c, "anchor": an, "cluster": cl[layer]}
                for (p, c, an, cl) in rows
            ],
        })
    return recs


def two_layer(transitions, condition="nl_base", k=4, problem="P0"):
    """One condition, one problem, one transition per anchor -> a chosen pooled graph."""
    rows = []
    for i, (s, d) in enumerate(transitions):
        rows.append((problem, condition, "a%d" % i, [s, d]))
    return build(rows, [k, k])


# --------------------------------------------------------------------------------------
# lambda2 against closed-form spectra
# --------------------------------------------------------------------------------------


class TestLambda2KnownGraphs(unittest.TestCase):

    def test_path_p3_normalized(self):
        # s0 -> d0, s1 -> d0 : path s0 - d0 - s1, normalized spectrum {0, 1, 2}
        a = M.Assignments(two_layer([(0, 0), (1, 0)]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 1.0, places=9)

    def test_path_p4_normalized(self):
        # s0 - d0 - s1 - d1 : normalized spectrum {0, 0.5, 1.5, 2}
        a = M.Assignments(two_layer([(0, 0), (1, 0), (1, 1)]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 0.5, places=9)

    def test_complete_bipartite_k22(self):
        # all four transitions present once: K_{2,2}, normalized spectrum {0, 1, 1, 2}
        a = M.Assignments(two_layer([(0, 0), (0, 1), (1, 0), (1, 1)]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 1.0, places=9)

    def test_single_edge_k2(self):
        # one transition: K_2, normalized spectrum {0, 2}
        a = M.Assignments(two_layer([(0, 0)]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 2.0, places=9)

    def test_disconnected_is_exactly_zero_and_not_an_error(self):
        # s0 -> d0 and s1 -> d1 : two components. lambda2 = 0 is a REAL value.
        a = M.Assignments(two_layer([(0, 0), (1, 1)]))
        self.assertEqual(M.lambda2(a, 0, "nl_base"), 0.0)

    def test_path_p5_normalized(self):
        # s0 - d0 - s1 - d1 - s2 : 1 - cos(k*pi/4) -> lambda2 = 1 - cos(pi/4)
        a = M.Assignments(two_layer([(0, 0), (1, 0), (1, 1), (2, 1)]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"),
                               1.0 - math.cos(math.pi / 4.0), places=9)

    def test_combinatorial_laplacian_path_p3(self):
        # combinatorial spectrum of P3 is {0, 1, 3}
        a = M.Assignments(two_layer([(0, 0), (1, 0)]))
        self.assertAlmostEqual(
            M.lambda2(a, 0, "nl_base", laplacian="combinatorial"), 1.0, places=9)

    def test_combinatorial_scales_with_weight_normalized_does_not(self):
        # doubling every transition doubles the combinatorial lambda2 and leaves the
        # normalized one alone -- the reason normalized is the reported quantity.
        single = M.Assignments(two_layer([(0, 0), (1, 0)]))
        doubled_rows = []
        for rep in range(2):
            for i, (s, d) in enumerate([(0, 0), (1, 0)]):
                doubled_rows.append(("P0", "nl_base", "a%d_%d" % (i, rep), [s, d]))
        doubled = M.Assignments(build(doubled_rows, [4, 4]))
        self.assertAlmostEqual(M.lambda2(doubled, 0, "nl_base"),
                               M.lambda2(single, 0, "nl_base"), places=9)
        self.assertAlmostEqual(
            M.lambda2(doubled, 0, "nl_base", laplacian="combinatorial"),
            2.0 * M.lambda2(single, 0, "nl_base", laplacian="combinatorial"), places=9)

    def test_source_and_target_clusters_are_distinct_nodes(self):
        # cluster 0 at layer 0 and cluster 0 at layer 1 are unrelated labels; if they
        # were merged, this single self-transition would be a 1-node graph.
        a = M.Assignments(two_layer([(0, 0)]))
        g = M.transition_graph(a, 0, "nl_base")
        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(sorted(g.nodes), [("dst", 0), ("src", 0)])

    def test_pooled_over_problems_not_per_problem(self):
        # each problem alone is a single edge (disconnected once pooled naively);
        # pooled they form P3 and lambda2 is 1.
        rows = [("P0", "nl_base", "a0", [0, 0]),
                ("P1", "nl_base", "a0", [1, 0])]
        a = M.Assignments(build(rows, [4, 4]))
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 1.0, places=9)
        self.assertEqual(M.lambda2(a, 0, "nl_base", problems=["P0", "P0"]), 2.0)

    def test_eigensolver_matches_numpy_when_available(self):
        mat = [[2.0, -1.0, 0.0, 0.5],
               [-1.0, 3.0, -0.5, 0.0],
               [0.0, -0.5, 1.0, -0.25],
               [0.5, 0.0, -0.25, 4.0]]
        jac = M._jacobi_eigvalsh(mat)
        if M.HAVE_NUMPY:
            import numpy as np
            ref = sorted(np.linalg.eigvalsh(np.array(mat)).tolist())
            for x, y in zip(jac, ref):
                self.assertAlmostEqual(x, y, places=9)
        # trace and determinant identities hold with or without numpy
        self.assertAlmostEqual(sum(jac), sum(mat[i][i] for i in range(4)), places=9)


# --------------------------------------------------------------------------------------
# path entropy
# --------------------------------------------------------------------------------------


class TestPathEntropy(unittest.TestCase):

    def test_deterministic_routing_is_zero(self):
        a = M.Assignments(two_layer([(0, 1), (0, 1), (1, 2), (1, 2)]))
        self.assertAlmostEqual(M.path_entropy(a, 0, "nl_base"), 0.0, places=12)
        self.assertAlmostEqual(M.path_entropy(a, 0, "nl_base", normalized=True), 0.0, places=12)

    def test_uniform_spread_hits_log_k(self):
        # one source spreading uniformly over all k=4 targets: H = log 4, normalized 1.0
        a = M.Assignments(two_layer([(0, 0), (0, 1), (0, 2), (0, 3)], k=4))
        self.assertAlmostEqual(M.path_entropy(a, 0, "nl_base"), math.log(4), places=12)
        self.assertAlmostEqual(M.path_entropy(a, 0, "nl_base", normalized=True), 1.0, places=12)

    def test_bounded_by_log_k(self):
        a = M.Assignments(two_layer([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)], k=4))
        h = M.path_entropy(a, 0, "nl_base")
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, math.log(4) + 1e-12)

    def test_conditional_not_joint(self):
        # two deterministic sources: joint entropy would be log 2, conditional is 0.
        a = M.Assignments(two_layer([(0, 0), (1, 1)]))
        self.assertAlmostEqual(M.path_entropy(a, 0, "nl_base"), 0.0, places=12)


# --------------------------------------------------------------------------------------
# fork layer and censoring
# --------------------------------------------------------------------------------------


def fork_fixture():
    """P0 forks at layer 2; P1 never forks (right-censored); 4 layers."""
    rows = [
        ("P0", "nl_base", "x", [0, 0, 0, 0]),
        ("P0", "nl_base", "FINAL", [0, 1, 1, 1]),
        ("P0", "nl_break", "x", [0, 0, 2, 2]),
        ("P0", "nl_break", "FINAL", [0, 1, 3, 3]),
        ("P1", "nl_base", "x", [0, 0, 0, 0]),
        ("P1", "nl_base", "FINAL", [1, 1, 1, 1]),
        ("P1", "nl_break", "x", [0, 0, 0, 0]),
        ("P1", "nl_break", "FINAL", [1, 1, 1, 1]),
    ]
    return M.Assignments(build(rows, [4, 4, 4, 4]))


class TestForkLayer(unittest.TestCase):

    def test_fork_detected_at_first_disjoint_layer(self):
        a = fork_fixture()
        f = M.fork_layer(a, "P0", ("nl_base", "nl_break"))
        self.assertEqual(f.layer, 2)
        self.assertFalse(f.censored)
        self.assertEqual(f.layers, (0, 1, 2))

    def test_never_forking_pair_is_none_and_censored(self):
        a = fork_fixture()
        f = M.fork_layer(a, "P1", ("nl_base", "nl_break"))
        self.assertIsNone(f.layer)
        self.assertTrue(f.censored)
        # no sentinel layer anywhere in the result
        self.assertNotIn(f.layer, (0, 1, 2, 3, -1, 40, 47, 48))
        self.assertEqual(f.layers, (0, 1, 2, 3))

    def test_censoring_flag_is_the_only_channel(self):
        a = fork_fixture()
        results = M.fork_layers(a, ("nl_base", "nl_break"))
        self.assertEqual(len(results), 2)
        for f in results:
            self.assertEqual(f.censored, f.layer is None)

    def test_partial_overlap_is_not_a_fork(self):
        # sets {0,1} and {1,2} share cluster 1 -> not disjoint -> no fork
        rows = [
            ("P0", "nl_base", "x", [0]),
            ("P0", "nl_base", "FINAL", [1]),
            ("P0", "nl_break", "x", [1]),
            ("P0", "nl_break", "FINAL", [2]),
        ]
        a = M.Assignments(build(rows, [4]))
        f = M.fork_layer(a, "P0", ("nl_base", "nl_break"))
        self.assertTrue(f.censored)


# --------------------------------------------------------------------------------------
# schema invariants -- loud failures
# --------------------------------------------------------------------------------------


class TestInvariants(unittest.TestCase):

    def test_cluster_outside_k_is_fatal(self):
        # a per-condition k shows up as a condition whose labels leave [0, k)
        rows = [("P0", "nl_base", "x", [0]), ("P0", "formal", "x", [7])]
        with self.assertRaises(M.InvariantViolation):
            M.Assignments(build(rows, [4]))

    def test_row_level_k_disagreeing_with_layer_k_is_fatal(self):
        recs = build([("P0", "nl_base", "x", [0])], [4])
        recs[0]["rows"][0]["k"] = 9
        with self.assertRaises(M.InvariantViolation) as cm:
            M.Assignments(recs)
        self.assertIn("k is selected ONCE per layer", str(cm.exception))

    def test_row_order_instability_is_fatal(self):
        recs = build([("P0", "nl_base", "x", [0, 0]), ("P0", "nl_base", "y", [1, 1])], [4, 4])
        recs[1]["rows"].reverse()
        with self.assertRaises(M.InvariantViolation) as cm:
            M.Assignments(recs)
        self.assertIn("row order is NOT stable", str(cm.exception))

    def test_row_count_change_across_layers_is_fatal(self):
        recs = build([("P0", "nl_base", "x", [0, 0]), ("P0", "nl_base", "y", [1, 1])], [4, 4])
        recs[1]["rows"].pop()
        with self.assertRaises(M.InvariantViolation):
            M.Assignments(recs)

    def test_raw_affinity_is_refused(self):
        recs = build([("P0", "nl_base", "x", [0])], [4], affinity="raw")
        with self.assertRaises(M.InvariantViolation) as cm:
            M.Assignments(recs)
        self.assertIn("Amendment 3", str(cm.exception))

    def test_per_layer_standardized_affinity_is_allowed(self):
        recs = build([("P0", "nl_base", "x", [0])], [4], affinity="per_layer_standardized")
        self.assertEqual(len(M.Assignments(recs)), 1)

    def test_anchor_set_mismatch_across_conditions_is_fatal(self):
        rows = [("P0", "nl_base", "x", [0]), ("P0", "nl_base", "FINAL", [1]),
                ("P0", "formal", "FINAL", [1])]
        with self.assertRaises(M.InvariantViolation) as cm:
            M.Assignments(build(rows, [4]))
        self.assertIn("anchor sets", str(cm.exception))

    def test_duplicate_rows_are_fatal(self):
        rows = [("P0", "nl_base", "x", [0]), ("P0", "nl_base", "x", [1])]
        with self.assertRaises(M.InvariantViolation):
            M.Assignments(build(rows, [4]))

    def test_duplicate_layer_record_is_fatal(self):
        recs = build([("P0", "nl_base", "x", [0])], [4])
        with self.assertRaises(M.InvariantViolation):
            M.Assignments(recs + recs)

    def test_missing_layer_pair_is_fatal(self):
        a = M.Assignments(build([("P0", "nl_base", "x", [0, 0])], [4, 4]))
        with self.assertRaises(M.InvariantViolation):
            M.lambda2(a, 5, "nl_base")

    def test_layer_pairs_convention(self):
        a = M.Assignments(build([("P0", "nl_base", "x", [0] * 6)], [4] * 6))
        self.assertEqual(a.layer_pairs(), (0, 1, 2, 3, 4))  # L layers -> L-1 pairs

    def test_jsonl_roundtrip(self):
        recs = build([("P0", "nl_base", "x", [0, 1])], [4, 4])
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            path = fh.name
        try:
            a = M.load_assignments(path)
            self.assertEqual(a.layers, (0, 1))
        finally:
            os.unlink(path)


# --------------------------------------------------------------------------------------
# resampling unit: PROBLEM, never rows
# --------------------------------------------------------------------------------------


def resample_fixture():
    """Four problems whose pooled transitions differ, so a problem resample moves lambda2."""
    spec = {
        "P0": [(0, 0), (1, 0)],
        "P1": [(1, 1), (2, 1)],
        "P2": [(0, 1), (2, 0)],
        "P3": [(0, 0), (2, 1)],
    }
    rows = []
    for pid, trans in spec.items():
        for i, (s, d) in enumerate(trans):
            rows.append((pid, "nl_base", "a%d" % i, [s, d]))
    return M.Assignments(build(rows, [4, 4]))


class TestResamplingUnit(unittest.TestCase):

    def test_problem_level_resample_changes_the_statistic(self):
        a = resample_fixture()
        full = M.lambda2(a, 0, "nl_base")
        resampled = M.lambda2(a, 0, "nl_base", problems=["P0", "P0", "P1", "P3"])
        self.assertNotAlmostEqual(full, resampled, places=6)

    def test_row_shuffle_within_problems_does_not_change_the_statistic(self):
        a = resample_fixture()
        before = M.lambda2(a, 0, "nl_base")
        entropy_before = M.path_entropy(a, 0, "nl_base")

        # Permute rows WITHIN each problem, identically at every layer (which keeps the
        # cross-layer row-order invariant intact). The pooled graph is a multiset of
        # transitions, so nothing may move.
        rng = random.Random(7)
        recs = [json.loads(json.dumps(a.record(L))) for L in a.layers]
        n = len(recs[0]["rows"])
        by_problem = {}
        for j in range(n):
            by_problem.setdefault(recs[0]["rows"][j]["problem_id"], []).append(j)
        order = list(range(n))
        for idxs in by_problem.values():
            shuffled = idxs[:]
            rng.shuffle(shuffled)
            for slot, src in zip(idxs, shuffled):
                order[slot] = src
        for rec in recs:
            rec["rows"] = [rec["rows"][i] for i in order]
        b = M.Assignments(recs)

        self.assertNotEqual([r["anchor"] for r in recs[0]["rows"]],
                            [k[2] for k in a.keys])  # the shuffle really happened
        self.assertAlmostEqual(M.lambda2(b, 0, "nl_base"), before, places=12)
        self.assertAlmostEqual(M.path_entropy(b, 0, "nl_base"), entropy_before, places=12)

    def test_bootstrap_draws_problems_with_replacement(self):
        a = resample_fixture()
        seen = []

        def stat(problems):
            seen.append(list(problems))
            return M.lambda2(a, 0, "nl_base", problems=problems)

        res = S.bootstrap_ci(list(a.problems), stat, n_boot=50, seed=1)
        self.assertEqual(res["resampling_unit"], "problem")
        self.assertEqual(seen[0], list(a.problems))          # point estimate first
        self.assertTrue(all(len(s) == 4 for s in seen[1:]))   # same n each draw
        self.assertTrue(any(len(set(s)) < 4 for s in seen[1:]))  # with replacement
        self.assertTrue(all(set(s) <= set(a.problems) for s in seen[1:]))
        self.assertLessEqual(res["ci_low"], res["ci_high"])

    def test_bootstrap_refuses_row_level_units(self):
        with self.assertRaises(S.ResamplingUnitError):
            S.bootstrap_ci([{"problem_id": "P0", "anchor": "x"}], lambda p: 0.0)
        with self.assertRaises(S.ResamplingUnitError):
            S.bootstrap_ci([("P0", "x"), ("P0", "FINAL")], lambda p: 0.0)
        with self.assertRaises(S.ResamplingUnitError):
            S.bootstrap_ci(["P0", "P0", "P1"], lambda p: 0.0)   # duplicates
        with self.assertRaises(S.ResamplingUnitError):
            S.bootstrap_ci("P0P1", lambda p: 0.0)

    def test_bootstrap_is_deterministic_under_seed(self):
        a = resample_fixture()

        def stat(problems):
            return M.lambda2(a, 0, "nl_base", problems=problems)

        r1 = S.bootstrap_ci(list(a.problems), stat, n_boot=40, seed=3)
        r2 = S.bootstrap_ci(list(a.problems), stat, n_boot=40, seed=3)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertEqual(r1["ci_high"], r2["ci_high"])

    def test_permutation_swaps_condition_labels_per_problem(self):
        ids = ["P0", "P1", "P2", "P3"]
        seen = []

        def stat(swapped):
            seen.append(swapped)
            return 1.0 if swapped else 0.0

        res = S.permutation_null(ids, stat, n_perm=64, alternative="less", seed=5)
        self.assertEqual(seen[0], frozenset())            # observed first
        self.assertTrue(all(isinstance(s, frozenset) for s in seen))
        self.assertTrue(all(s <= set(ids) for s in seen))
        self.assertTrue(any(len(s) not in (0, 4) for s in seen[1:]))
        self.assertGreater(res["p"], 0.0)                 # never exactly zero
        self.assertEqual(res["permutation_unit"],
                         "problem (condition labels swapped within a problem)")

    def test_permutation_condition_mapping_reaches_metrics(self):
        a = resample_fixture()
        rows = []
        for pid in a.problems:
            for k in a.index_by_pc[(pid, "nl_base")]:
                rows.append(k)
        # a mapping problem -> condition is accepted wherever a condition name is
        mapping = {pid: "nl_base" for pid in a.problems}
        self.assertAlmostEqual(M.lambda2(a, 0, mapping),
                               M.lambda2(a, 0, "nl_base"), places=12)


# --------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------


class TestWilcoxon(unittest.TestCase):

    def test_known_small_example(self):
        # differences -1, -2, -3, +4 : ranks 1,2,3,4 -> W+ = 4, W- = 6
        d = [-1.0, -2.0, -3.0, 4.0]
        r = S.wilcoxon_paired(d, alternative="two-sided")
        self.assertEqual(r["w_plus"], 4.0)
        self.assertEqual(r["w_minus"], 6.0)
        self.assertEqual(r["method"], "exact")
        # P(W+ <= 4) with n=4 : W+ counts {0..10}; #{<=4} = 1+1+1+2+2 = ... check directly
        counts, total = S._signed_rank_null_cdf(4)
        p_le = sum(counts[:5]) / total
        self.assertAlmostEqual(r["p"], min(1.0, 2 * min(p_le, 1 - sum(counts[:4]) / total)),
                               places=12)

    def test_all_negative_differences_one_sided_less(self):
        d = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        r = S.wilcoxon_paired(d, alternative="less")
        self.assertEqual(r["w_plus"], 0.0)
        self.assertAlmostEqual(r["p"], 1.0 / 2 ** 6, places=12)
        self.assertLess(r["mean_difference"], 0)

    def test_paired_two_sample_form(self):
        x = [1.0, 2.0, 3.0]
        y = [2.0, 4.0, 6.0]
        r = S.wilcoxon_paired(x, y, alternative="less")
        self.assertEqual(r["w_plus"], 0.0)
        self.assertEqual(r["n"], 3)

    def test_zeros_are_dropped_and_counted(self):
        r = S.wilcoxon_paired([0.0, 0.0, -1.0, -2.0], alternative="less")
        self.assertEqual(r["n_zero"], 2)
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["n_total"], 4)

    def test_n47_uses_normal_approximation(self):
        d = [-(i + 1) * 0.001 for i in range(47)]
        r = S.wilcoxon_paired(d, alternative="less")
        self.assertIn("normal", r["method"])
        self.assertEqual(r["n"], 47)
        self.assertLess(r["p"], 1e-6)

    def test_symmetric_data_is_not_significant(self):
        d = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        r = S.wilcoxon_paired(d, alternative="two-sided")
        self.assertGreater(r["p"], 0.5)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            S.wilcoxon_paired([1.0, 2.0], [1.0])


class TestBenjaminiHochberg(unittest.TestCase):

    def test_known_example(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        r = S.benjamini_hochberg(p, q=0.05)
        # BH at q=.05, m=8: largest i with p_i <= .05*i/8 is i=5 (0.042 <= 0.03125? no)
        # i=1: .001<=.00625 yes; i=2: .008<=.0125 yes; i=3: .039<=.01875 no; ...
        self.assertEqual(r["n_rejected"], 2)
        self.assertEqual(r["rejected"][:2], [True, True])
        self.assertTrue(all(a >= b for a, b in zip(r["adjusted"][1:], r["adjusted"][:1])))

    def test_adjusted_are_monotone_and_bounded(self):
        p = [0.01, 0.5, 0.02, 0.9, 0.001]
        r = S.benjamini_hochberg(p)
        for v in r["adjusted"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        order = sorted(range(5), key=lambda i: p[i])
        adj_sorted = [r["adjusted"][i] for i in order]
        self.assertEqual(adj_sorted, sorted(adj_sorted))

    def test_all_null_rejects_nothing(self):
        r = S.benjamini_hochberg([0.4, 0.6, 0.8, 0.99])
        self.assertEqual(r["n_rejected"], 0)

    def test_out_of_range_pvalue_raises(self):
        with self.assertRaises(ValueError):
            S.benjamini_hochberg([0.1, 1.4])


class TestLogLambda2(unittest.TestCase):

    def test_masks_disconnected_zeros(self):
        r = S.log_lambda2([0.0, 0.5, 1.0])
        self.assertEqual(r["n_zero"], 1)
        self.assertEqual(r["mask"], [False, True, True])
        self.assertIsNone(r["log_values"][0])
        self.assertAlmostEqual(r["log_values"][2], 0.0, places=12)

    def test_raise_policy(self):
        with self.assertRaises(ValueError):
            S.log_lambda2([0.0, 1.0], zero_policy="raise")

    def test_negative_lambda2_is_rejected(self):
        with self.assertRaises(ValueError):
            S.log_lambda2([-0.1])


class TestSurvival(unittest.TestCase):

    def make_records(self, n=40, seed=0):
        """Answer-breaking pairs (survival=0) fork early; surviving pairs fork late or
        are censored. The coefficient on answer_survival must come out NEGATIVE."""
        rng = random.Random(seed)
        layers = list(range(12))
        recs = []
        for i in range(n):
            for surv in (0, 1):
                if surv == 0:
                    layer = rng.choice([1, 2, 3])
                    cens = False
                else:
                    layer = rng.choice([8, 9, 10, None, None])
                    cens = layer is None
                recs.append({
                    "problem_id": "P%03d" % i,
                    "fork_layer": layer,
                    "censored": cens,
                    "answer_survival": float(surv),
                    "lex_dissim": rng.random(),
                })
        return recs, layers

    def test_person_period_row_counts(self):
        recs = [
            {"problem_id": "P0", "fork_layer": 2, "censored": False, "answer_survival": 0.0},
            {"problem_id": "P1", "fork_layer": None, "censored": True, "answer_survival": 1.0},
        ]
        rows = S.person_period_expand(recs, layers=[0, 1, 2, 3])
        self.assertEqual(len(rows), 3 + 4)                      # up to fork; all for censored
        self.assertEqual([r["event"] for r in rows[:3]], [0, 0, 1])
        self.assertEqual([r["event"] for r in rows[3:]], [0, 0, 0, 0])
        self.assertEqual([r["t"] for r in rows[3:]], [1, 2, 3, 4])

    def test_censored_pairs_contribute_all_non_fork_rows(self):
        a = fork_fixture()
        forks = M.fork_layers(a, ("nl_base", "nl_break"))
        recs = [{"problem_id": f.problem_id, "fork": f, "answer_survival": 0.0}
                for f in forks]
        rows = S.person_period_expand(recs, layers=a.layers)
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["problem_id"], []).append(r)
        self.assertEqual(len(by_pair["P0"]), 3)     # forked at layer 2 (t = 1, 2, 3)
        self.assertEqual(len(by_pair["P1"]), 4)     # censored: every layer, no event
        self.assertEqual(sum(r["event"] for r in by_pair["P1"]), 0)
        self.assertEqual(sum(r["event"] for r in by_pair["P0"]), 1)

    def test_censored_with_a_fork_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            S.person_period_expand(
                [{"problem_id": "P0", "fork_layer": 3, "censored": True}], layers=[0, 1, 2, 3])

    def test_uncensored_without_a_fork_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            S.person_period_expand(
                [{"problem_id": "P0", "fork_layer": None, "censored": False}], layers=[0, 1])

    def test_missing_censoring_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            S.person_period_expand([{"problem_id": "P0", "fork_layer": 3}], layers=[0, 1, 2, 3])

    def test_cloglog_recovers_the_direction(self):
        recs, layers = self.make_records()
        fit = S.discrete_time_survival(recs, layers=layers, covariates=["lex_dissim"])
        self.assertEqual(fit["link"], "cloglog")
        self.assertTrue(fit["converged"])
        prim = fit["survival_term"]
        self.assertLess(prim["coef"], 0.0)             # surviving pairs fork LATER
        self.assertTrue(prim["consistent_with_precommitted_direction"])
        self.assertLess(prim["p"], 0.01)
        self.assertEqual(fit["n_problem_clusters"], 40)
        self.assertIn("lex_dissim", fit["coefficients"])
        self.assertFalse(fit["random_intercept_for_problem"])

    def test_cluster_robust_se_differs_from_model_based(self):
        recs, layers = self.make_records()
        fit = S.discrete_time_survival(recs, layers=layers)
        prim = fit["survival_term"]
        self.assertNotAlmostEqual(prim["se"], prim["se_model"], places=6)
        self.assertEqual(fit["se_type"], "cluster-robust (problem)")

    def test_all_censored_is_refused(self):
        recs = [{"problem_id": "P%d" % i, "fork_layer": None, "censored": True,
                 "answer_survival": 1.0} for i in range(5)]
        with self.assertRaises(ValueError):
            S.discrete_time_survival(recs, layers=[0, 1, 2])

    def test_censored_pairs_are_not_dropped_from_the_fit(self):
        recs, layers = self.make_records()
        fit_all = S.discrete_time_survival(recs, layers=layers)
        kept = [r for r in recs if not r["censored"]]
        fit_dropped = S.discrete_time_survival(kept, layers=layers)
        self.assertGreater(fit_all["n_rows"], fit_dropped["n_rows"])
        self.assertGreater(fit_all["n_censored"], 0)
        # dropping the censored pairs biases the coefficient -- the §A.9.1 point
        self.assertNotAlmostEqual(fit_all["survival_term"]["coef"],
                                  fit_dropped["survival_term"]["coef"], places=3)

    def test_baseline_hazard_variants_run(self):
        recs, layers = self.make_records(n=20, seed=2)
        for baseline in ("linear", "log", "none"):
            fit = S.discrete_time_survival(recs, layers=layers, baseline=baseline)
            self.assertLess(fit["survival_term"]["coef"], 0.0)


class TestMetricCorrelation(unittest.TestCase):

    def test_shape_and_labelling(self):
        rng = random.Random(11)
        rows = []
        for pid in ["P%d" % i for i in range(6)]:
            for cond in ("nl_base", "formal"):
                for anchor in ("q1", "q2", "FINAL"):
                    rows.append((pid, cond, anchor,
                                 [rng.randrange(4) for _ in range(6)]))
        a = M.Assignments(build(rows, [4] * 6))
        r = M.metric_correlation_matrix(a, "formal")
        self.assertEqual(r["status"], "EXPLORATORY")
        self.assertEqual(r["metrics"], ["lambda2", "path_entropy", "forked_fraction"])
        self.assertEqual(len(r["matrix"]), 3)
        for i in range(3):
            self.assertEqual(len(r["matrix"][i]), 3)
            self.assertEqual(r["matrix"][i][i], 1.0)
        self.assertEqual(r["n_layer_pairs"], 5)

    def test_spearman_known_values(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=12)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=12)


# --------------------------------------------------------------------------------------
# scripts/04_analyze.py end to end
# --------------------------------------------------------------------------------------


class TestAnalyzeScript(unittest.TestCase):

    def _corpus(self, path, n_problems=12, n_layers=6, seed=4):
        rng = random.Random(seed)
        rows = []
        for i in range(n_problems):
            pid = "P%02d" % i
            anchors = ["q1", "q2", "FINAL"]
            for cond in ("nl_base", "formal", "nl_break"):
                for an in anchors:
                    seq = []
                    for L in range(n_layers):
                        if cond == "formal" and L > 0:
                            seq.append(rng.randrange(2))    # formal collapses -> lower l2
                        else:
                            seq.append(rng.randrange(4))
                    rows.append((pid, cond, an, seq))
        recs = build(rows, [4] * n_layers,
                     eigengaps=[0.01 * (L + 1) for L in range(n_layers)])
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    def test_end_to_end(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "analyze04", os.path.join(_REPO, "scripts", "04_analyze.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "assignments.jsonl")
            out = os.path.join(d, "analysis.json")
            self._corpus(src)
            rc = mod.main(["--assignments", src, "--out", out,
                           "--n-boot", "20", "--n-perm", "20"])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)

        self.assertEqual(res["primary_endpoints"]["count"], 2)
        self.assertEqual(set(res["primary_endpoints"]) - {"count", "note"}, {"h2", "h3"})

        h3 = res["primary_endpoints"]["h3"]
        self.assertEqual(h3["status"], "PRIMARY")
        self.assertEqual(h3["pairing_unit"], "layer-pair")
        self.assertEqual(h3["n_layer_pairs"], 5)
        self.assertEqual(len(h3["paired_differences"]), 5)
        self.assertEqual(h3["laplacian"], "symmetric normalized (I - D^-1/2 W D^-1/2)")
        self.assertEqual(h3["bootstrap_ci"]["resampling_unit"], "problem")
        self.assertIn("problem", h3["permutation_null"]["permutation_unit"])
        self.assertEqual(h3["per_layer_pair"]["status"], "EXPLORATORY")
        self.assertEqual(len(h3["per_layer_pair"]["p"]), 5)

        h2 = res["primary_endpoints"]["h2"]
        self.assertEqual(h2["status"], "PRIMARY")
        self.assertEqual(h2["fit"]["link"], "cloglog")
        self.assertFalse(h2["primary_endpoint_complete"])   # no --covariates given
        self.assertIn("NOT the pre-registered primary endpoint", h2["caveat"])

        h1 = res["h1_descriptive"]
        self.assertFalse(h1["is_test"])
        self.assertTrue(h1["status"].startswith("DESCRIPTIVE"))
        self.assertNotIn("p", h1)                            # H1 emits no p-value
        self.assertFalse(h1["available"])                    # no labels in the fixture
        self.assertIn("will not fabricate", h1["reason"])

        for key, block in res["exploratory"].items():
            if isinstance(block, dict):
                self.assertEqual(block.get("status"), "EXPLORATORY", key)

    def test_h1_reports_ami_when_labels_are_present(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "analyze04b", os.path.join(_REPO, "scripts", "04_analyze.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # spectral cluster == linguistic label, kmeans == noise: AMI difference > 0.
        rng = random.Random(9)
        recs = []
        for layer in range(2):
            rows = []
            for i in range(24):
                lab = i % 3
                rows.append({
                    "problem_id": "P%02d" % (i // 3),
                    "condition": "nl_base",
                    "anchor": "a%d" % (i % 3),
                    "cluster": lab,
                    "linguistic_label": "L%d" % lab,
                    "kmeans_cluster": rng.randrange(3),
                })
            recs.append({"layer": layer, "k": 4, "eigengap": 0.1 * (layer + 1),
                         "affinity": "cosine", "rows": rows})

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "assignments.jsonl")
            out = os.path.join(d, "analysis.json")
            with open(src, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            mod.main(["--assignments", src, "--out", out, "--n-boot", "10",
                      "--n-perm", "0"])
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)

        h1 = res["h1_descriptive"]
        self.assertTrue(h1["available"])
        self.assertEqual(h1["layer"], 1)                      # largest eigengap
        self.assertAlmostEqual(h1["ami_spectral"], 1.0, places=9)
        self.assertGreater(h1["ami_difference"], 0.0)
        self.assertEqual(h1["bootstrap_ci_difference"]["resampling_unit"], "problem")
        self.assertFalse(h1["is_test"])

    def test_ami_matches_sklearn_if_available(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "analyze04c", os.path.join(_REPO, "scripts", "04_analyze.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 2, 1, 0]
        b = [0, 1, 1, 1, 2, 2, 0, 0, 2, 1, 1, 0]
        self.assertAlmostEqual(mod.adjusted_mutual_information(a, a), 1.0, places=9)
        try:
            from sklearn.metrics import adjusted_mutual_info_score
        except Exception:
            self.skipTest("sklearn not installed: AMI cross-check skipped LOUDLY")
        self.assertAlmostEqual(mod.adjusted_mutual_information(a, b),
                               adjusted_mutual_info_score(a, b, average_method="arithmetic"),
                               places=9)


# --------------------------------------------------------------------------------------
# Style-normalized displacement (DEVIATIONS Amendment 6)
# --------------------------------------------------------------------------------------


def snd_fixture(cond_trans, n_problems=6, k=4, skip=()):
    """Two layers (one layer-pair); every problem repeats the same transitions.

    `cond_trans` maps condition -> [(src_cluster, dst_cluster)] with one entry per
    anchor. `skip` is a set of (condition, problem_index) cells to leave OUT entirely --
    an Amendment 6 unavailable cell, which is an absent row, never a placeholder.

    Repeating one transition multiset across problems scales every edge weight by the
    number of problems, and the symmetric normalized Laplacian is invariant to that, so
    the pooled lambda2 has a closed form independent of n_problems.
    """
    rows = []
    for i in range(n_problems):
        pid = "P%02d" % i
        for cond, trans in cond_trans.items():
            if (cond, i) in skip:
                continue
            for j, (s, d) in enumerate(trans):
                rows.append((pid, cond, "a%d" % j, [s, d]))
    return M.Assignments(build(rows, [k, k]))


# nl_base    : s0-d0, s1-d0, s1-d1 -> the path P4, normalized spectrum {0, 1/2, 3/2, 2}
# orth floor : one edge of weight 3   -> spectrum {0, 2}
# lex floor  : star centred on d0     -> spectrum {0, 1, 2}
# formal     : s0-d0 and s1-d1        -> TWO components, so lambda2 is exactly 0
SND_BASE = [(0, 0), (1, 0), (1, 1)]
SND_ORTH = [(0, 0), (0, 0), (0, 0)]
SND_LEX = [(0, 0), (1, 0), (0, 0)]
SND_FORMAL = [(0, 0), (1, 1), (0, 0)]
SND_CONDS = {"nl_base": SND_BASE, "formal": SND_FORMAL,
             "nl_minimal_orth": SND_ORTH, "nl_minimal_lex": SND_LEX}


class TestStyleNormalizedDisplacement(unittest.TestCase):

    def test_the_closed_form_lambda2_values_the_ratios_are_built_from(self):
        a = snd_fixture(SND_CONDS)
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_base"), 0.5, places=12)
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_minimal_orth"), 2.0, places=12)
        self.assertAlmostEqual(M.lambda2(a, 0, "nl_minimal_lex"), 1.0, places=12)
        self.assertEqual(M.lambda2(a, 0, "formal"), 0.0)

    def test_snd_matches_the_hand_computed_ratio_against_each_floor(self):
        a = snd_fixture(SND_CONDS)
        # (0 - 1/2) / (2 - 1/2) = -1/3
        up = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_orth", n_boot=40, seed=3)
        self.assertTrue(up["denominator_ci_excludes_zero"])
        self.assertAlmostEqual(up["raw_difference"]["point"], -0.5, places=12)
        self.assertAlmostEqual(up["denominator"]["point"], 1.5, places=12)
        self.assertAlmostEqual(up["snd"], -1.0 / 3.0, places=12)
        # (0 - 1/2) / (1 - 1/2) = -1
        lo = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_lex", n_boot=40, seed=3)
        self.assertAlmostEqual(lo["snd"], -1.0, places=12)

    def test_snd_for_path_entropy_hand_computed(self):
        # normalized H: base = (2/3)log2 / log4 = 1/3; every other condition routes
        # deterministically, so H = 0. Both floors give (0 - 1/3)/(0 - 1/3) = 1.
        a = snd_fixture(SND_CONDS)
        self.assertAlmostEqual(
            M.path_entropy(a, 0, "nl_base", normalized=True), 1.0 / 3.0, places=12)
        r = M.style_normalized_displacement(
            a, "path_entropy", "formal", "nl_minimal_orth", n_boot=40, seed=3)
        self.assertAlmostEqual(r["snd"], 1.0, places=12)

    def test_range_reports_both_floors_and_their_span(self):
        a = snd_fixture(SND_CONDS)
        r = M.snd_range(a, "lambda2", "formal", n_boot=40, seed=3)
        self.assertTrue(r["range_available"])
        self.assertAlmostEqual(r["range"]["low"], -1.0, places=12)
        self.assertAlmostEqual(r["range"]["high"], -1.0 / 3.0, places=12)
        self.assertEqual(set(r["by_floor"]), {"nl_minimal_orth", "nl_minimal_lex"})
        self.assertIn("INTERPRETIVE SECONDARY", r["status"])
        for block in r["by_floor"].values():
            self.assertIn("bracket", block["bracket_role"])
            self.assertNotIn("matched magnitude", block["bracket_role"].lower())

    def test_precommitted_abs_gt_1_is_evaluated_not_assumed(self):
        a = snd_fixture(SND_CONDS)
        small = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_orth", n_boot=40, seed=3,
            precommitted="abs_gt_1")
        self.assertFalse(small["consistent_with_precommitted_direction"])   # |−1/3| < 1
        # (2 - 1/2) / (1 - 1/2) = 3
        big = M.style_normalized_displacement(
            a, "lambda2", "nl_minimal_orth", "nl_minimal_lex", n_boot=40, seed=3,
            precommitted="abs_gt_1")
        self.assertAlmostEqual(big["snd"], 3.0, places=12)
        self.assertTrue(big["consistent_with_precommitted_direction"])


class TestSNDNearZeroDenominatorGuard(unittest.TestCase):
    """The ratio artifact is how this metric fails, so the guard is tested directly."""

    def test_zero_denominator_returns_null_with_a_reason(self):
        conds = dict(SND_CONDS)
        conds["nl_minimal_orth"] = list(SND_BASE)        # floor == base: denominator 0
        a = snd_fixture(conds)
        r = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_orth", n_boot=40, seed=3)
        self.assertIsNone(r["snd"])
        self.assertFalse(r["denominator_ci_excludes_zero"])
        self.assertIn("INCLUDES zero", r["undefined_reason"])
        self.assertEqual(r["denominator"]["point"], 0.0)

    def test_near_zero_denominator_is_null_and_not_a_spectacular_number(self):
        # The floor differs from nl_base for ONE problem in eight, so the denominator is
        # small but nonzero and its problem-level bootstrap CI straddles zero. The naive
        # ratio is enormous; the guard must return null instead of that number.
        n = 30
        rows = []
        for i in range(n):
            pid = "P%02d" % i
            for cond, trans in (("nl_base", SND_BASE), ("formal", SND_FORMAL),
                                ("nl_minimal_lex", SND_LEX)):
                for j, (s, d) in enumerate(trans):
                    rows.append((pid, cond, "a%d" % j, [s, d]))
            floor = list(SND_BASE)
            if i == 0:
                floor[2] = (0, 1)                        # one problem perturbs one edge
            for j, (s, d) in enumerate(floor):
                rows.append((pid, "nl_minimal_orth", "a%d" % j, [s, d]))
        a = M.Assignments(build(rows, [4, 4]))

        r = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_orth", n_boot=200, seed=5)
        den = r["denominator"]["point"]
        num = r["raw_difference"]["point"]
        self.assertNotEqual(den, 0.0)
        self.assertGreater(abs(num / den), 10.0)         # the artifact the guard prevents
        self.assertIsNone(r["snd"])
        self.assertFalse(r["denominator_ci_excludes_zero"])
        self.assertIn("division artifact", r["undefined_reason"])
        # and the raw difference -- the PRIMARY quantity -- is still reported
        self.assertIsNotNone(r["raw_difference"]["bootstrap_ci"])

    def test_no_bootstrap_means_no_snd_rather_than_an_unguarded_ratio(self):
        a = snd_fixture(SND_CONDS)
        r = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_orth", n_boot=0)
        self.assertIsNone(r["snd"])
        self.assertIn("guard", r["undefined_reason"])

    def test_a_failing_floor_removes_the_range_but_keeps_the_other_floor(self):
        conds = dict(SND_CONDS)
        conds["nl_minimal_orth"] = list(SND_BASE)
        a = snd_fixture(conds)
        r = M.snd_range(a, "lambda2", "formal", n_boot=40, seed=3)
        self.assertFalse(r["range_available"])
        self.assertIsNone(r["range"])
        self.assertIn("nl_minimal_orth", r["range_unavailable_reason"])
        self.assertAlmostEqual(r["by_floor"]["nl_minimal_lex"]["snd"], -1.0, places=12)


class TestUnavailableStyleControlCells(unittest.TestCase):
    """Amendment 6: 31 problem x condition cells are genuine holes."""

    def test_problems_for_and_availability(self):
        a = snd_fixture(SND_CONDS, n_problems=6,
                        skip={("nl_minimal_lex", 0), ("nl_minimal_lex", 1),
                              ("nl_minimal_orth", 5)})
        self.assertEqual(len(a.problems_for("nl_base")), 6)
        self.assertEqual(len(a.problems_for("nl_minimal_lex")), 4)
        self.assertEqual(len(a.problems_for("nl_minimal_lex", "nl_minimal_orth")), 3)
        av = a.availability("nl_minimal_lex")
        self.assertEqual((av["n_available"], av["n_unavailable"]), (4, 2))
        self.assertEqual(av["unavailable_problems"], ["P00", "P01"])

    def test_snd_uses_the_common_problem_set_and_counts_the_exclusions(self):
        a = snd_fixture(SND_CONDS, n_problems=6, skip={("nl_minimal_lex", 0),
                                                       ("nl_minimal_lex", 1)})
        r = M.style_normalized_displacement(
            a, "lambda2", "formal", "nl_minimal_lex", n_boot=40, seed=3)
        self.assertEqual(r["n_problems"], 4)
        self.assertEqual(r["n_problems_excluded_unavailable"], 2)
        self.assertAlmostEqual(r["snd"], -1.0, places=12)      # ratio is unchanged

    def test_fork_layers_skips_cells_with_no_realization(self):
        a = snd_fixture(SND_CONDS, n_problems=6, skip={("nl_minimal_orth", 3)})
        forks = M.fork_layers(a, ("nl_base", "nl_minimal_orth"))
        self.assertEqual(len(forks), 5)
        self.assertNotIn("P03", [f.problem_id for f in forks])


# --------------------------------------------------------------------------------------
# H2's pre-registered SECONDARY contrast, and the no-events degradation
# --------------------------------------------------------------------------------------


def _load_analyze(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, "scripts", "04_analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def six_condition_corpus(path, fork_at, n_layers=10, k=4, seed=1,
                         randomize_formal=True):
    """`fork_at(cond, i)` -> the layer at which condition `cond` of problem i diverges
    from nl_base, or None for a pair that never forks (right-censored)."""
    rng = random.Random(seed)
    conds = ("nl_base", "nl_para", "formal", "nl_break", "nl_minimal_orth",
             "nl_minimal_lex")
    rows = []
    for i in range(24):
        pid = "P%02d" % i
        for cond in conds:
            L = None if cond == "nl_base" else fork_at(cond, i)
            if L == "absent":
                continue
            for an in ("q1", "FINAL"):
                seq = [0 if (L is None or layer < L) else 1 for layer in range(n_layers)]
                if cond == "formal" and randomize_formal:
                    seq = [s if layer < 1 else rng.randrange(2)
                           for layer, s in enumerate(seq)]
                rows.append((pid, cond, an, seq))
    recs = build(rows, [k] * n_layers,
                 eigengaps=[0.01 * (L + 1) for L in range(n_layers)])
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


class TestH2SecondaryHazardContrast(unittest.TestCase):

    def _run(self, fork_at, extra=()):
        mod = _load_analyze("analyze04_sec")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "assignments.jsonl")
            out = os.path.join(d, "analysis.json")
            six_condition_corpus(src, fork_at)
            rc = mod.main(["--assignments", src, "--out", out, "--n-boot", "10",
                           "--n-perm", "0"] + list(extra))
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                return json.load(fh)

    @staticmethod
    def _ordering(cond, i):
        rng = random.Random(1000 + i)
        if cond == "nl_break":
            return rng.choice([1, 2, 3])            # breaks fork EARLY
        if cond == "nl_minimal_orth":
            return rng.choice([5, 6, 7, None])      # the style floor forks LATE
        if cond == "nl_minimal_lex":
            return "absent" if i % 8 == 0 else rng.choice([5, 6, None])
        return rng.choice([4, 5, 6, None])

    def test_known_ordering_gives_a_hazard_ratio_above_one(self):
        res = self._run(self._ordering)
        sec = res["secondary_endpoints"]["h2_fork_hazard_break_vs_style_floor"]
        self.assertTrue(sec["status"].startswith("PRE-REGISTERED SECONDARY"))
        self.assertTrue(sec["available"])
        self.assertTrue(sec["model_fitted"])
        self.assertGreater(sec["hazard_ratio"], 1.0)
        self.assertGreater(sec["fit"]["survival_term"]["coef"], 0.0)
        self.assertTrue(sec["consistent_with_precommitted_direction"])
        self.assertLess(sec["p"], 0.01)
        self.assertEqual(sec["fit"]["link"], "cloglog")

    def test_the_secondary_never_sits_among_the_primary_endpoints(self):
        res = self._run(self._ordering)
        self.assertEqual(res["primary_endpoints"]["count"], 2)
        self.assertEqual(set(res["primary_endpoints"]) - {"count", "note"}, {"h2", "h3"})
        sec = res["secondary_endpoints"]["h2_fork_hazard_break_vs_style_floor"]
        self.assertFalse(sec["replaces_primary"])
        self.assertIn("does NOT replace", sec["status"])
        self.assertEqual(res["primary_endpoints"]["h2"]["status"], "PRIMARY")
        self.assertTrue(res["primary_endpoints"]["h2"]["model_fitted"])
        for block in res["secondary_endpoints"].values():
            if isinstance(block, dict):
                self.assertNotEqual(block.get("status"), "PRIMARY")

    def test_the_contrast_is_labelled_matched_site_never_matched_magnitude(self):
        res = self._run(self._ordering)
        sec = res["secondary_endpoints"]["h2_fork_hazard_break_vs_style_floor"]
        blob = json.dumps(sec).lower()
        self.assertNotIn("matched-magnitude", blob)
        self.assertNotIn("matched magnitude", blob.replace("not on edit magnitude", ""))
        self.assertIn("site", sec["matching"].lower())

    def test_unavailable_cells_are_excluded_and_counted(self):
        res = self._run(self._ordering)
        h2 = res["primary_endpoints"]["h2"]
        lex = h2["availability"]["by_condition"]["nl_minimal_lex"]
        self.assertEqual(lex["n_excluded_no_realization"], 3)      # i % 8 == 0 of 24
        self.assertEqual(lex["n_entered"], 21)
        self.assertEqual(h2["availability"]["total_cells_excluded"], 3)
        meta = res["meta"]["availability"]
        self.assertEqual(meta["total_cells_unavailable"], 3)
        self.assertEqual(meta["by_condition"]["nl_minimal_lex"]["n_available"], 21)

    def test_metric_correlation_matrix_covers_every_non_reference_condition(self):
        res = self._run(self._ordering)
        corr = res["exploratory"]["metric_correlation_matrix"]
        self.assertEqual(corr["n_conditions_covered"], 5)
        self.assertEqual(corr["conditions_covered"],
                         ["formal", "nl_break", "nl_minimal_lex", "nl_minimal_orth",
                          "nl_para"])
        for cond, block in corr["by_condition"].items():
            self.assertEqual(block["metrics"],
                             ["lambda2", "path_entropy", "forked_fraction"], cond)

    def test_snd_block_is_present_and_labelled_interpretive_secondary(self):
        res = self._run(self._ordering)
        snd = res["secondary_endpoints"]["style_normalized_displacement"]
        self.assertTrue(snd["available"])
        self.assertIn("INTERPRETIVE SECONDARY", snd["status"])
        self.assertEqual(sorted(snd["by_metric"]), ["lambda2", "path_entropy"])
        for metric, by_cond in snd["by_metric"].items():
            self.assertEqual(sorted(by_cond), ["formal", "nl_break", "nl_para"])
            for block in by_cond.values():
                for floor in block["by_floor"].values():
                    # never a number without the guard having passed
                    if floor["snd"] is not None:
                        self.assertTrue(floor["denominator_ci_excludes_zero"])
                    else:
                        self.assertIsNotNone(floor["undefined_reason"])


class TestSurvivalWithNoForkEvents(unittest.TestCase):
    """A 3am crash after the pod is torn down costs a GPU rental to recover from."""

    def test_analysis_reports_an_unfittable_model_and_keeps_going(self):
        mod = _load_analyze("analyze04_noevents")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "assignments.jsonl")
            out = os.path.join(d, "analysis.json")
            # every condition tracks nl_base at every layer: nothing ever forks
            six_condition_corpus(src, lambda cond, i: None,
                                 randomize_formal=False)
            rc = mod.main(["--assignments", src, "--out", out, "--n-boot", "10",
                           "--n-perm", "0"])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)

        h2 = res["primary_endpoints"]["h2"]
        self.assertTrue(h2["available"])
        self.assertFalse(h2["model_fitted"])
        self.assertFalse(h2["fit"]["fitted"])
        self.assertEqual(h2["fit"]["n_events"], 0)
        self.assertIn("no fork events", h2["fit"]["unfittable_reason"])
        self.assertNotIn("primary", h2["fit"])          # no coefficient is invented
        sec = res["secondary_endpoints"]["h2_fork_hazard_break_vs_style_floor"]
        self.assertFalse(sec["model_fitted"])
        self.assertEqual(sec["fit"]["n_events"], 0)
        # ...and the rest of the analysis was still computed and written
        self.assertTrue(res["primary_endpoints"]["h3"]["available"])
        self.assertIn("lambda2_by_condition", res["exploratory"])
        self.assertTrue(res["secondary_endpoints"]["style_normalized_displacement"]
                        ["available"])

    def test_the_statistics_layer_itself_still_refuses(self):
        # the graceful path belongs to the SCRIPT; the library still says no, loudly
        recs = [{"problem_id": "P%d" % i, "fork_layer": None, "censored": True,
                 "answer_survival": 1.0} for i in range(5)]
        with self.assertRaises(ValueError):
            S.discrete_time_survival(recs, layers=[0, 1, 2])



class TestUnavailableCovariateCells(unittest.TestCase):
    """Amendment 6 holes have no fork layer, so they never reach the covariate lookup.

    A pair that HAS a fork layer and no covariate entry is therefore a
    covariates/assignments mismatch, not a hole, and must stop the run rather than be
    silently dropped from H2's PRIMARY fit.
    """

    def _run(self, covariates):
        mod = _load_analyze("analyze04_cov")
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "assignments.jsonl")
            cov = os.path.join(d, "covariates.json")
            out = os.path.join(d, "analysis.json")
            six_condition_corpus(
                src, lambda cond, i: (1 + (i % 3) if cond == "nl_break"
                                      else 5 + (i % 4)))
            with open(cov, "w", encoding="utf-8") as fh:
                json.dump(covariates, fh)
            rc = mod.main(["--assignments", src, "--out", out, "--covariates", cov,
                           "--n-boot", "0", "--n-perm", "0"])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                return json.load(fh)

    @staticmethod
    def _covariates(omit=()):
        out = {}
        for i in range(24):
            pid = "P%02d" % i
            for cond in ("nl_para", "formal", "nl_break", "nl_minimal_orth",
                         "nl_minimal_lex"):
                if (cond, i) in omit:
                    continue
                rng = random.Random("%s/%d" % (cond, i))
                out.setdefault(pid, {})[cond] = {"lex_dissim": rng.random(),
                                                 "syn_dissim": rng.random()}
        return out

    def test_the_complete_file_fits_every_pair(self):
        res = self._run(self._covariates())
        h2 = res["primary_endpoints"]["h2"]
        self.assertTrue(h2["model_fitted"])
        self.assertEqual(h2["covariates_entered"], ["lex_dissim", "syn_dissim"])
        self.assertTrue(h2["primary_endpoint_complete"])
        by_cond = h2["availability"]["by_condition"]
        for cond in ("nl_minimal_lex", "nl_minimal_orth"):
            self.assertEqual(by_cond[cond]["n_excluded_no_covariate_entry"], 0)
        self.assertEqual(h2["availability"]["total_cells_excluded"], 0)
        self.assertEqual(h2["n_pairs_entered"], 24 * 5)

    def test_a_missing_entry_for_a_pair_that_forked_stops_the_run(self):
        # These pairs exist in the assignments and have a fork layer, so they are NOT
        # Amendment 6 holes -- dropping them would fit the PRIMARY coefficient on fewer
        # pairs than exist.
        omit = {("nl_minimal_lex", i) for i in range(5)} | {("nl_minimal_orth", 7)}
        with self.assertRaises(SystemExit):
            self._run(self._covariates(omit))

    def test_a_partial_covariate_entry_is_still_refused(self):
        cov = self._covariates()
        del cov["P03"]["formal"]["syn_dissim"]
        with self.assertRaises(SystemExit):
            self._run(cov)


if __name__ == "__main__":
    unittest.main()
