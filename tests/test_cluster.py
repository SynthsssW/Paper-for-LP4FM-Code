"""Tests for src/ling/cluster.py and scripts/03_extract_and_cluster.py.

The two clustering tests check the front end against structure whose right answer
is known by construction, not against itself:

    three well-separated angular Gaussian blobs -> spectral recovers them (ARI 1.0)
    a two-moons manifold lifted onto the unit sphere -> spectral recovers it and
        K-means does NOT, which is the entire content of H1's descriptive note
        (Amendment 1: "spectral clustering recovers non-convex structure and
        K-means does not")

The rest are the two structural guarantees the pod cannot get wrong and survive:
k is selected on nl_base rows and nothing else (§A.6, prereg decision 1, §A.11
item 4), and Nystrom landmarks come from the POOLED set (§A.11 item 6).

Everything below runs on a bare interpreter. ling.cluster's pure-python backend
is a TEST backend and refuses more than PURE_PY_MAX_ROWS rows, so the fixtures
are deliberately small; with numpy installed the identical tests run on the
production path.

Run with:  python3 -m unittest discover -s tests -v
"""

import importlib.util
import math
import os
import random
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import cluster as C  # noqa: E402


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "extract_and_cluster03", os.path.join(_REPO, "scripts", "03_extract_and_cluster.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def blobs(n_per=15, sd=0.04, seed=7):
    """Three angularly separated Gaussian blobs in 3-D. Truth labels returned.

    Separated in DIRECTION, not just position, because the cosine representation
    L2-normalizes every row (Amendment 3) — a fixture separated only by radius
    would be testing nothing.
    """
    rng = random.Random(seed)
    centres = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    X, y = [], []
    for c, centre in enumerate(centres):
        for _ in range(n_per):
            X.append([v + rng.gauss(0.0, sd) for v in centre])
            y.append(c)
    return X, y


def moons_on_sphere(n_per=30, noise=0.02, scale=0.35, seed=11):
    """Two interleaved half-moons, lifted onto the unit sphere.

    The lift matters: cosine affinity is euclidean distance on the unit sphere
    (``||x_hat - y_hat||^2 = 2 - 2 cos``), so a flat 2-D two-moons fixture would
    be collapsed to angle-around-the-origin and neither method would see the
    moons. Placing the moons on a small spherical cap preserves the non-convex
    geometry under the exact representation the pipeline uses.
    """
    rng = random.Random(seed)
    pts, y = [], []
    for i in range(n_per):
        t = math.pi * i / (n_per - 1)
        pts.append((math.cos(t), math.sin(t)))
        y.append(0)
    for i in range(n_per):
        t = math.pi * i / (n_per - 1)
        pts.append((1.0 - math.cos(t), 1.0 - math.sin(t) - 0.5))
        y.append(1)
    X = []
    for (u, v) in pts:
        u = scale * u + rng.gauss(0.0, noise)
        v = scale * v + rng.gauss(0.0, noise)
        w = math.sqrt(max(1.0 - u * u - v * v, 1e-9))
        X.append([u, v, w])
    return X, y


def uniform_arc(n=60, seed=4, noise=0.005, span=0.6):
    """Uniformly sampled points along a 1-D arc: no cluster structure at any k."""
    rng = random.Random(seed)
    X = []
    for i in range(n):
        u = span * i / (n - 1) + rng.gauss(0.0, noise)
        v = rng.gauss(0.0, noise)
        w = math.sqrt(max(1.0 - u * u - v * v, 1e-9))
        X.append([u, v, w])
    return X


def pooled_fixture(n_per=10, seed=3):
    """A pooled four-condition layer: rows, condition labels, truth labels.

    nl_base carries three clean blobs; the other conditions are jittered copies.
    Deliberately built so the non-base conditions have a MESSIER structure, which
    is what makes the "k came from nl_base only" test able to fail.
    """
    rng = random.Random(seed)
    conds = ["nl_base", "formal", "nl_break", "nl_para"]
    centres = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    X, cond_labels, truth = [], [], []
    for cond in conds:
        sd = 0.03 if cond == "nl_base" else 0.10
        for c, centre in enumerate(centres):
            for _ in range(n_per):
                X.append([v + rng.gauss(0.0, sd) for v in centre])
                cond_labels.append(cond)
                truth.append(c)
    return X, cond_labels, truth


# --------------------------------------------------------------------------------------
# recovery of known structure
# --------------------------------------------------------------------------------------


class TestKnownStructure(unittest.TestCase):

    def test_well_separated_gaussians_are_recovered(self):
        X, y = blobs()
        res = C.apply_clustering(X, 3, seed=0, n_init=5)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(C.adjusted_rand_index(res.labels, y), 1.0, places=6)

    def test_eigengap_finds_three_blobs(self):
        X, y = blobs()
        conds = ["nl_base"] * len(X)
        sel = C.select_k(X, conds, k_min=2, k_max=8)
        self.assertEqual(sel.k, 3)
        self.assertGreater(sel.eigengap, 0.0)
        self.assertFalse(sel.eigengap_flat)
        self.assertEqual(sel.condition, "nl_base")

    def test_two_moons_spectral_beats_kmeans(self):
        """H1's descriptive note, as a test: non-convex structure separates the two.

        ``knn_sigma=3`` rather than the default 7 because the bandwidth heuristic
        is density-adaptive and this fixture has 40 points per manifold where the
        pipeline has thousands: at 40 points the 7th-nearest-neighbour distance is
        wider than the gap between the moons, and a kernel wider than the gap
        merges them. The comparator is unaffected either way -- K-means never
        touches the kernel, and scores 0.15-0.35 at every bandwidth tried.
        """
        X, y = moons_on_sphere(n_per=40)
        res = C.apply_clustering(X, 2, seed=0, n_init=5, knn_sigma=3)
        ari_spectral = C.adjusted_rand_index(res.labels, y)
        ari_kmeans = C.adjusted_rand_index(res.kmeans_labels, y)
        self.assertGreater(ari_spectral, 0.9,
                           "spectral clustering failed to recover the moons (ARI %.3f)"
                           % ari_spectral)
        self.assertLess(ari_kmeans, ari_spectral - 0.2,
                        "K-means recovered the moons as well as spectral did (%.3f vs "
                        "%.3f); the H1 comparator is not doing its job" % (ari_kmeans,
                                                                          ari_spectral))

    def test_kmeans_is_deterministic_given_seed(self):
        X, _ = blobs()
        a, ia = C.kmeans(C.prepare_states(X), 3, seed=5, n_init=3)
        b, ib = C.kmeans(C.prepare_states(X), 3, seed=5, n_init=3)
        self.assertEqual(a, b)
        self.assertAlmostEqual(ia, ib, places=10)


# --------------------------------------------------------------------------------------
# k is selected on nl_base and NOWHERE ELSE  (§A.6, prereg decision 1, §A.11 item 4)
# --------------------------------------------------------------------------------------


class TestKSelectionSeesOnlyBase(unittest.TestCase):

    def test_select_k_refuses_non_base_rows(self):
        X, conds, _ = pooled_fixture()
        with self.assertRaises(C.ClusterError) as ctx:
            C.select_k(X, conds)
        msg = str(ctx.exception)
        self.assertIn("nl_base", msg)
        self.assertIn("§A.6", msg)

    def test_select_k_refuses_unlabelled_rows(self):
        X, _ = blobs()
        with self.assertRaises(C.ClusterError):
            C.select_k(X, ["nl_base"] * (len(X) - 1))

    def test_cluster_layer_k_equals_base_only_k(self):
        """The k a full pooled layer gets is exactly the k the base rows alone give."""
        X, conds, _ = pooled_fixture()
        base_idx = C.nl_base_subset(conds)
        base_rows = [X[i] for i in base_idx]
        expected = C.select_k(base_rows, [conds[i] for i in base_idx], k_min=2, k_max=8)
        lc = C.cluster_layer(X, conds, 12, k_min=2, k_max=8, seed=0, n_init=3)
        self.assertEqual(lc.k, expected.k)
        self.assertEqual(lc.selection.n_rows, len(base_idx))
        self.assertEqual(lc.selection.condition, "nl_base")

    def test_k_is_identical_across_conditions_at_a_layer(self):
        X, conds, _ = pooled_fixture()
        lc = C.cluster_layer(X, conds, 3, k_min=2, k_max=8, seed=0, n_init=3)
        for cond, n_used in lc.clusters_used_by_condition.items():
            self.assertLessEqual(n_used, lc.k, "condition %r used more than k clusters" % cond)
        for lab in lc.result.labels:
            self.assertTrue(0 <= lab < lc.k)

    def test_apply_clustering_uses_the_k_it_is_given(self):
        """It has no k-selection code; hand it a k the eigengap would not choose."""
        X, _ = blobs()
        for k in (2, 4, 5):
            res = C.apply_clustering(X, k, seed=1, n_init=3, with_kmeans=False)
            self.assertEqual(res.k, k)
            self.assertLessEqual(len(set(res.labels)), k)
            self.assertTrue(all(0 <= v < k for v in res.labels))

    def test_flat_spectrum_is_flagged(self):
        """A uniform 1-D arc has no cluster structure: the flag fires, k still comes out.

        §A.6 wants flat-spectrum layers FLAGGED, not dropped, so the assertion is
        on the flag and on the layer still being clusterable.
        """
        X = uniform_arc()
        sel = C.select_k(X, ["nl_base"] * len(X), k_min=2, k_max=10)
        self.assertTrue(sel.eigengap_flat,
                        "a structureless layer was not flagged flat (gap=%.5f ratio=%.3f)"
                        % (sel.eigengap, sel.flat_ratio))
        self.assertGreaterEqual(sel.k, 2)

    def test_clustered_spectrum_is_not_flagged_flat(self):
        """The contrast case, so the flag is not just always-on."""
        X, _ = blobs()
        sel = C.select_k(X, ["nl_base"] * len(X), k_min=2, k_max=10)
        self.assertFalse(sel.eigengap_flat)
        self.assertGreater(sel.flat_ratio, C.DEFAULT_FLAT_RATIO_THRESHOLD)

    def test_raw_affinity_is_refused(self):
        X, _ = blobs()
        with self.assertRaises(C.ClusterError) as ctx:
            C.prepare_states(X, "raw")
        self.assertIn("Amendment 3", str(ctx.exception))
        self.assertNotIn("raw", C.VALID_AFFINITIES)

    def test_per_layer_standardized_is_available(self):
        X, y = blobs()
        res = C.apply_clustering(X, 3, affinity=C.PER_LAYER_STANDARDIZED, seed=0, n_init=5)
        self.assertEqual(res.affinity, C.PER_LAYER_STANDARDIZED)
        self.assertEqual(len(res.labels), len(X))


# --------------------------------------------------------------------------------------
# Nystrom landmarks come from the POOLED set  (§A.11 item 6, prereg decision 7)
# --------------------------------------------------------------------------------------


class TestNystromLandmarks(unittest.TestCase):

    def test_landmarks_are_drawn_from_the_pooled_set(self):
        X, conds, _ = pooled_fixture()
        n = len(X)
        lm = C.sample_landmarks(n, 24, seed=0, conditions=conds)
        self.assertEqual(len(lm), 24)
        self.assertEqual(len(set(lm)), 24)
        self.assertTrue(all(0 <= i < n for i in lm))
        # every condition is represented -- that is what "pooled" means
        self.assertEqual({conds[i] for i in lm}, set(conds))

    def test_landmarks_are_reproducible_and_seed_dependent(self):
        X, conds, _ = pooled_fixture()
        n = len(X)
        a = C.sample_landmarks(n, 24, seed=0, conditions=conds)
        b = C.sample_landmarks(n, 24, seed=0, conditions=conds)
        c = C.sample_landmarks(n, 24, seed=1, conditions=conds)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_landmark_set_missing_a_condition_is_refused(self):
        X, conds, _ = pooled_fixture()
        with self.assertRaises(C.ClusterError) as ctx:
            C.sample_landmarks(len(X), 2, seed=0, conditions=conds)
        self.assertIn("POOLED", str(ctx.exception))

    def test_per_condition_landmarks_are_refused_by_apply_clustering(self):
        X, conds, _ = pooled_fixture()
        base_only = C.nl_base_subset(conds)[:20]
        with self.assertRaises(C.ClusterError) as ctx:
            C.apply_clustering(X, 3, landmark_idx=base_only, conditions=conds,
                               seed=0, n_init=2, with_kmeans=False)
        self.assertIn("§A.11 item 6", str(ctx.exception))

    def test_nystrom_path_recovers_the_same_structure_as_exact(self):
        X, conds, truth = pooled_fixture()
        lm = C.sample_landmarks(len(X), 40, seed=0, conditions=conds)
        approx = C.apply_clustering(X, 3, landmark_idx=lm, conditions=conds,
                                    seed=0, n_init=5, with_kmeans=False)
        self.assertEqual(approx.method, "nystrom")
        self.assertEqual(approx.n_landmarks, 40)
        self.assertGreater(C.adjusted_rand_index(approx.labels, truth), 0.9)

    def test_g4_holdout_must_not_overlap_the_landmarks(self):
        X, conds, _ = pooled_fixture()
        Xp = C.prepare_states(X)
        lm = C.sample_landmarks(len(X), 30, seed=0, conditions=conds)
        with self.assertRaises(C.ClusterError) as ctx:
            C.nystrom_error_report(Xp, lm, lm[:5], sigma=0.3)
        self.assertIn("understates the error", str(ctx.exception))

    def test_g4_error_report_is_computed(self):
        X, conds, _ = pooled_fixture()
        Xp = C.prepare_states(X)
        lm = C.sample_landmarks(len(X), 40, seed=0, conditions=conds)
        holdout = [i for i in range(len(X)) if i not in set(lm)][:15]
        sigma = C.choose_sigma(Xp, lm, knn_sigma=5)
        rep = C.nystrom_error_report(Xp, lm, holdout, sigma, n_eigs=6)
        self.assertEqual(rep["gate"], "G4")
        self.assertEqual(rep["n_landmarks"], 40)
        self.assertEqual(rep["n_holdout"], 15)
        self.assertEqual(len(rep["exact_eigenvalues"]), len(rep["nystrom_eigenvalues"]))
        self.assertGreaterEqual(rep["max_abs_eigenvalue_error"], 0.0)
        self.assertLess(rep["max_abs_eigenvalue_error"], 1.0)


# --------------------------------------------------------------------------------------
# stability, ARI
# --------------------------------------------------------------------------------------


class TestStability(unittest.TestCase):

    def test_ari_of_a_relabelled_partition_is_one(self):
        a = [0, 0, 1, 1, 2, 2]
        b = [2, 2, 0, 0, 1, 1]
        self.assertAlmostEqual(C.adjusted_rand_index(a, b), 1.0, places=9)

    def test_ari_of_unrelated_partitions_is_near_zero(self):
        rng = random.Random(0)
        a = [rng.randrange(3) for _ in range(200)]
        b = [rng.randrange(3) for _ in range(200)]
        self.assertLess(abs(C.adjusted_rand_index(a, b)), 0.1)

    def test_ari_stability_across_seeds(self):
        X, y = blobs()
        rep = C.ari_stability(X, 3, seeds=(0, 1, 2), n_init=3)
        self.assertEqual(rep["k"], 3)
        self.assertEqual(len(rep["pairwise"]), 3)
        self.assertGreater(rep["min_ari"], 0.9,
                           "well-separated blobs should be seed-stable, got %r" % rep)

    def test_ari_stability_needs_two_seeds(self):
        X, _ = blobs()
        with self.assertRaises(C.ClusterError):
            C.ari_stability(X, 3, seeds=(0,))


# --------------------------------------------------------------------------------------
# backend guard
# --------------------------------------------------------------------------------------


class TestBackendGuard(unittest.TestCase):

    @unittest.skipIf(C.have_numpy(), "numpy present: the production path is in use")
    def test_pure_python_backend_refuses_pipeline_scale_loudly(self):
        X = [[0.0, 1.0] for _ in range(C.PURE_PY_MAX_ROWS + 1)]
        with self.assertRaises(C.ClusterError) as ctx:
            C.apply_clustering(X, 3, seed=0)
        self.assertIn("numpy", str(ctx.exception))


# --------------------------------------------------------------------------------------
# the driver's write-time invariants  (configs/assignments_schema.md)
# --------------------------------------------------------------------------------------


def _rows(k=3, conditions=("formal", "nl_base"), anchors=("FINAL", "x"), problems=("p0", "p1")):
    rows = []
    for pid in problems:
        for cond in conditions:
            for a in anchors:
                rows.append({"problem_id": pid, "condition": cond, "anchor": a,
                             "cluster": (hash((pid, cond, a)) % k), "kmeans_cluster": 0,
                             "linguistic_label": None})
    return rows


def _records(n_layers=3, k=3):
    rows = _rows(k=k)
    return [{"layer": i, "k": k, "eigengap": 0.1, "eigengap_flat": False,
             "affinity": C.COSINE, "rows": [dict(r) for r in rows]}
            for i in range(n_layers)]


class TestDriverInvariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.D = _load_driver()

    def test_valid_records_pass(self):
        self.assertTrue(self.D.assert_schema_invariants(_records()))

    def test_missing_kmeans_cluster_is_fatal(self):
        recs = _records()
        del recs[0]["rows"][0]["kmeans_cluster"]
        with self.assertRaises(self.D.InvariantError):
            self.D.assert_schema_invariants(recs)

    def test_missing_linguistic_label_is_fatal(self):
        recs = _records()
        del recs[1]["rows"][2]["linguistic_label"]
        with self.assertRaises(self.D.InvariantError):
            self.D.assert_schema_invariants(recs)

    def test_cluster_outside_k_is_fatal(self):
        recs = _records(k=3)
        recs[0]["rows"][0]["cluster"] = 3
        with self.assertRaises(self.D.InvariantError) as ctx:
            self.D.assert_schema_invariants(recs)
        self.assertIn("per-condition k", str(ctx.exception))

    def test_unstable_row_order_is_fatal(self):
        recs = _records()
        recs[2]["rows"].reverse()
        with self.assertRaises(self.D.InvariantError) as ctx:
            self.D.assert_schema_invariants(recs)
        self.assertIn("row order is NOT stable", str(ctx.exception))

    def test_anchor_set_differing_across_conditions_is_fatal(self):
        recs = _records()
        for rec in recs:
            for row in rec["rows"]:
                if row["condition"] == "formal" and row["anchor"] == "x":
                    row["anchor"] = "y"
        with self.assertRaises(self.D.InvariantError) as ctx:
            self.D.assert_schema_invariants(recs)
        self.assertIn("anchor sets across conditions", str(ctx.exception))

    def test_raw_affinity_is_fatal(self):
        recs = _records()
        recs[0]["affinity"] = "raw"
        with self.assertRaises(self.D.InvariantError):
            self.D.assert_schema_invariants(recs)

    def test_duplicate_layer_is_fatal(self):
        recs = _records()
        recs[1]["layer"] = 0
        with self.assertRaises(self.D.InvariantError):
            self.D.assert_schema_invariants(recs)


class TestLinguisticLabelPassthrough(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.D = _load_driver()

    def test_absent_label_is_none_not_invented(self):
        rec = {"problem_id": "p", "realizations": {"nl_base": {"text": "t", "anchors": {}}}}
        self.assertIsNone(self.D.linguistic_label(rec, "nl_base", "FINAL"))
        self.assertIsNone(self.D.linguistic_label(rec, "nl_base", "x"))

    def test_per_anchor_label_is_passed_through(self):
        rec = {"problem_id": "p", "realizations": {
            "nl_base": {"linguistic_labels": {"FINAL": "PUNCT", "x": "NUM"}}}}
        self.assertEqual(self.D.linguistic_label(rec, "nl_base", "FINAL"), "PUNCT")
        self.assertEqual(self.D.linguistic_label(rec, "nl_base", "x"), "NUM")

    def test_record_level_label_is_passed_through(self):
        rec = {"problem_id": "p", "realizations": {"nl_base": {}},
               "linguistic_labels": {"nl_base": {"x": "NUM"}}}
        self.assertEqual(self.D.linguistic_label(rec, "nl_base", "x"), "NUM")
        self.assertIsNone(self.D.linguistic_label(rec, "nl_base", "FINAL"))

    def test_real_corpus_record_has_no_label_yet(self):
        path = os.path.join(_REPO, "data", "corpus", "problems.jsonl")
        if not os.path.exists(path):
            self.skipTest("no corpus generated")
        import json
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        # Not an aspiration: if this ever starts returning a label, it is because
        # another agent produced one, and H1 becomes computable. It must never be
        # this script that invents it.
        self.assertIsNone(self.D.linguistic_label(rec, "nl_base", "FINAL"))


if __name__ == "__main__":
    unittest.main()
