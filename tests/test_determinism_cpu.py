"""CPU-only tests for the GPU-side foundation.

What CPU can and cannot check
-----------------------------
The determinism gate itself (G1) is a GPU claim and cannot be tested here — its
whole content is that two CUDA forward passes produce identical bits. What CPU
*can* check, and what this file checks, is everything that would silently
corrupt the pipeline while the gate happily passes:

* the anchor CHARACTER-span -> TOKEN-index mapping (the important one). It is
  the join between the corpus and the model, it is invisible in the outputs, and
  §A.6 / configs/corpus_schema.md depend on it entirely.
* the ordering of anchor columns, which must be identical across the four
  realizations or every cross-condition comparison is comparing different
  quantities.
* that ``assert_bitwise_identical`` actually rejects a difference of 1e-8 —
  i.e. that it is not secretly an ``allclose``.
* that raw activations cannot be checkpointed (§A.1 rule 1).
* import-order safety: CUBLAS_WORKSPACE_CONFIG is in the environment as soon as
  ``ling.determinism`` is imported.

Run with:  python3 -m unittest discover -s tests -v
(stdlib unittest, so it runs on a machine with no torch and no pytest)
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ling import determinism as det          # noqa: E402
from ling import hooks                        # noqa: E402
from ling.hooks import (                      # noqa: E402
    AnchorError, FINAL, LayerCheckpointWriter, anchor_order,
    char_span_to_token_index, check_anchor_keys_consistent,
    resolve_anchor_positions,
)

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

HAVE_NUMPY = np is not None


# ==========================================================================
# A fake fast tokenizer.
#
# It splits on whitespace and on digit/word boundaries, which is enough to
# exercise every branch of the span mapping: multi-token quantities, leading
# spaces folded into a token, a BOS with a zero-width span, and an optional
# trailing special token. It reproduces the offset_mapping contract that the
# real fast tokenizers implement, which is the only thing under test.
# ==========================================================================
class FakeFastTokenizer:
    is_fast = True

    def __init__(self, add_bos=True, add_eos=False, leading_space=True):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.leading_space = leading_space

    def _split(self, text):
        """-> list of (start, end) spans covering the non-whitespace runs."""
        spans = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            j = i
            kind = text[i].isdigit()
            while j < n and not text[j].isspace() and text[j].isdigit() == kind:
                j += 1
            start = i
            # sentencepiece-style: the preceding space belongs to this token
            if self.leading_space and start > 0 and text[start - 1] == " ":
                start -= 1
            spans.append((start, j))
            i = j
        return spans

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True,
                 return_tensors=None, padding=False, truncation=False, **kw):
        spans = self._split(text)
        offsets = list(spans)
        ids = list(range(100, 100 + len(spans)))
        if add_special_tokens and self.add_bos:
            offsets.insert(0, (0, 0))
            ids.insert(0, 1)
        if add_special_tokens and self.add_eos:
            offsets.append((0, 0))
            ids.append(2)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class SlowTokenizer(FakeFastTokenizer):
    is_fast = False


TEXT = "Natalia sold 48 clips in April and 24 in May."
#       0123456789...
#       "Natalia"=0:7  " sold"=7:12  " 48"=12:15  " clips"=15:21 ...


class TestCharSpanToTokenIndex(unittest.TestCase):
    """The join between the corpus and the model. §A.6, corpus_schema.md."""

    def setUp(self):
        self.tok = FakeFastTokenizer()
        enc = self.tok(TEXT, return_offsets_mapping=True)
        self.offsets = enc["offset_mapping"]
        self.ids = enc["input_ids"]

    def test_offsets_line_up_with_the_text(self):
        # sanity on the fixture itself
        words = [TEXT[s:e].strip() for s, e in self.offsets if e > s]
        self.assertEqual(words[0], "Natalia")
        self.assertIn("48", words)

    def test_single_token_quantity(self):
        span = (TEXT.index("48"), TEXT.index("48") + 2)
        i = char_span_to_token_index(self.offsets, span)
        s, e = self.offsets[i]
        self.assertEqual(TEXT[s:e].strip(), "48")

    def test_bos_is_not_matched(self):
        """A zero-width special token must never absorb an anchor at char 0."""
        span = (0, 7)                       # "Natalia"
        i = char_span_to_token_index(self.offsets, span)
        self.assertNotEqual(i, 0)           # index 0 is the BOS
        self.assertEqual(self.offsets[i], (0, 7))

    def test_multi_token_span_last_vs_first(self):
        # a span covering "48 clips" touches two tokens
        s = TEXT.index("48")
        e = TEXT.index("clips") + len("clips")
        i_last = char_span_to_token_index(self.offsets, (s, e), strategy="last")
        i_first = char_span_to_token_index(self.offsets, (s, e), strategy="first")
        self.assertGreater(i_last, i_first)
        self.assertEqual(TEXT[self.offsets[i_first][0]:self.offsets[i_first][1]].strip(), "48")
        self.assertEqual(TEXT[self.offsets[i_last][0]:self.offsets[i_last][1]].strip(), "clips")

    def test_default_strategy_is_last(self):
        s = TEXT.index("48")
        e = TEXT.index("clips") + len("clips")
        self.assertEqual(char_span_to_token_index(self.offsets, (s, e)),
                         char_span_to_token_index(self.offsets, (s, e), strategy="last"))

    def test_final_sentinel(self):
        i = char_span_to_token_index(self.offsets, (-1, -1))
        self.assertEqual(i, len(self.offsets) - 1)

    def test_final_sentinel_rejects_trailing_special(self):
        """A trailing EOS is not a content position and can differ per condition."""
        tok = FakeFastTokenizer(add_eos=True)
        offs = tok(TEXT, return_offsets_mapping=True)["offset_mapping"]
        with self.assertRaises(AnchorError) as cm:
            char_span_to_token_index(offs, (-1, -1))
        self.assertIn("zero-width", str(cm.exception))
        # ... unless explicitly allowed
        self.assertEqual(
            char_span_to_token_index(offs, (-1, -1), allow_trailing_special=True),
            len(offs) - 1)

    def test_unmatched_span_raises(self):
        with self.assertRaises(AnchorError) as cm:
            char_span_to_token_index(self.offsets, (len(TEXT) + 5, len(TEXT) + 9),
                                     name="num_apples")
        msg = str(cm.exception)
        self.assertIn("num_apples", msg)
        self.assertIn("re-tokenizing", msg)   # names the forbidden workaround

    def test_bad_strategy_raises(self):
        with self.assertRaises(AnchorError):
            char_span_to_token_index(self.offsets, (0, 7), strategy="middle")

    def test_zero_width_anchor_is_a_point_lookup(self):
        pos = TEXT.index("48")
        i = char_span_to_token_index(self.offsets, (pos, pos))
        self.assertEqual(TEXT[self.offsets[i][0]:self.offsets[i][1]].strip(), "48")


class TestAnchorOrdering(unittest.TestCase):
    """Column j must mean the same quantity in all four realizations. §A.6."""

    def test_names_sorted_final_last(self):
        self.assertEqual(
            anchor_order({"zebra": [0, 1], "apple": [2, 3], FINAL: [-1, -1]}),
            ["apple", "zebra", FINAL])

    def test_missing_final_raises(self):
        with self.assertRaises(AnchorError):
            anchor_order({"a": [0, 1]})

    def test_order_is_independent_of_textual_position(self):
        """The Python rewrite may introduce quantities in a different order."""
        nl = {"num_clips": [13, 15], "num_days": [35, 37], FINAL: [-1, -1]}
        formal = {"num_days": [4, 6], "num_clips": [20, 22], FINAL: [-1, -1]}
        self.assertEqual(anchor_order(nl), anchor_order(formal))

    def test_check_anchor_keys_consistent(self):
        good = {c: {"anchors": {"a": [0, 1], "b": [2, 3], FINAL: [-1, -1]}}
                for c in ("nl_base", "nl_para", "formal", "nl_break")}
        self.assertEqual(check_anchor_keys_consistent(good), ["a", "b", FINAL])

    def test_check_anchor_keys_inconsistent_raises(self):
        bad = {
            "nl_base": {"anchors": {"a": [0, 1], "b": [2, 3], FINAL: [-1, -1]}},
            "formal": {"anchors": {"a": [0, 1], FINAL: [-1, -1]}},
        }
        with self.assertRaises(AnchorError) as cm:
            check_anchor_keys_consistent(bad)
        self.assertIn("drop", str(cm.exception).lower())


class TestResolveAnchorPositions(unittest.TestCase):
    def test_end_to_end(self):
        tok = FakeFastTokenizer()
        anchors = {
            "num_clips_april": [TEXT.index("48"), TEXT.index("48") + 2],
            "num_clips_may": [TEXT.index("24"), TEXT.index("24") + 2],
            FINAL: [-1, -1],
        }
        res = resolve_anchor_positions(TEXT, anchors, tok)
        self.assertEqual(res.names, ["num_clips_april", "num_clips_may", FINAL])
        self.assertEqual(res.n_positions, 3)
        self.assertEqual(len(res.token_indices), 3)
        self.assertEqual(res.token_indices[2], res.n_tokens - 1)
        # the two quantity anchors landed on the right text
        for name, idx in zip(res.names[:2], res.token_indices[:2]):
            s, e = res.offsets[idx]
            self.assertIn(TEXT[s:e].strip(), ("48", "24"))
        # strictly increasing here, since the anchors appear in text order
        self.assertLess(res.token_indices[0], res.token_indices[1])

    def test_slow_tokenizer_refused(self):
        with self.assertRaises(AnchorError) as cm:
            resolve_anchor_positions(TEXT, {FINAL: [-1, -1]}, SlowTokenizer())
        self.assertIn("fast tokenizer", str(cm.exception))

    def test_duplicate_positions_can_be_rejected(self):
        tok = FakeFastTokenizer()
        pos = TEXT.index("48")
        anchors = {"a": [pos, pos + 2], "b": [pos, pos + 2], FINAL: [-1, -1]}
        resolve_anchor_positions(TEXT, anchors, tok)          # allowed by default
        with self.assertRaises(AnchorError):
            resolve_anchor_positions(TEXT, anchors, tok,
                                     allow_duplicate_positions=False)

    def test_aligned_columns_across_conditions(self):
        """The load-bearing property: same column -> same quantity, all conditions."""
        tok = FakeFastTokenizer()
        nl = "Natalia sold 48 clips over 3 days."
        formal = "days = 3\nclips = 48\ntotal = clips / days"
        a_nl = {"n_clips": [nl.index("48"), nl.index("48") + 2],
                "n_days": [nl.index("3 days"), nl.index("3 days") + 1],
                FINAL: [-1, -1]}
        a_fm = {"n_clips": [formal.index("48"), formal.index("48") + 2],
                "n_days": [formal.index("3"), formal.index("3") + 1],
                FINAL: [-1, -1]}
        r_nl = resolve_anchor_positions(nl, a_nl, tok)
        r_fm = resolve_anchor_positions(formal, a_fm, tok)
        self.assertEqual(r_nl.names, r_fm.names)
        # the quantity appears BEFORE the other one in NL and AFTER it in the
        # formal rewrite; the column order must not follow the text order
        self.assertLess(r_nl.token_indices[0], r_nl.token_indices[1])
        self.assertGreater(r_fm.token_indices[0], r_fm.token_indices[1])
        for name, r, txt in (("n_clips", r_nl, nl), ("n_clips", r_fm, formal)):
            idx = r.token_indices[r.names.index(name)]
            s, e = r.offsets[idx]
            self.assertEqual(txt[s:e].strip(), "48")


class TestAssertBitwiseIdentical(unittest.TestCase):
    """It must be exact equality, not allclose. §A.4."""

    def test_identical_passes(self):
        det.assert_bitwise_identical([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_one_ulp_fails(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0 + 1e-16, 3.0]
        if a == b:                       # 1e-16 is below float64 resolution here
            b = [1.0, 2.0000000000000004, 3.0]
        with self.assertRaises(det.DeterminismError):
            det.assert_bitwise_identical(a, b)

    def test_tiny_difference_fails_and_reports_index_and_magnitude(self):
        a = [0.0] * 10
        b = [0.0] * 10
        b[7] = 1e-8
        with self.assertRaises(det.DeterminismError) as cm:
            det.assert_bitwise_identical(a, b, name="residual")
        msg = str(cm.exception)
        self.assertIn("residual", msg)
        self.assertIn("flat index 7", msg)
        self.assertIn("1e-08", msg)
        self.assertIn("EXACTLY zero", msg)

    def test_length_mismatch(self):
        with self.assertRaises(det.DeterminismError):
            det.assert_bitwise_identical([1.0], [1.0, 2.0])

    def test_nested_sequences(self):
        det.assert_bitwise_identical([[1.0, 2.0], [3.0, 4.0]],
                                     [[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(det.DeterminismError):
            det.assert_bitwise_identical([[1.0, 2.0], [3.0, 4.0]],
                                         [[1.0, 2.0], [3.0, 4.5]])

    def test_max_abs_difference(self):
        self.assertEqual(det.max_abs_difference([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertAlmostEqual(det.max_abs_difference([1.0, 2.0], [1.0, 2.5]), 0.5)

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_numpy_arrays(self):
        a = np.zeros((4, 5), dtype=np.float32)
        b = a.copy()
        det.assert_bitwise_identical(a, b, name="X")
        b[2, 3] = np.float32(1e-7)
        with self.assertRaises(det.DeterminismError) as cm:
            det.assert_bitwise_identical(a, b, name="X")
        self.assertIn("(2, 3)", str(cm.exception))

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_numpy_shape_mismatch_message_mentions_batching(self):
        with self.assertRaises(det.DeterminismError) as cm:
            det.assert_bitwise_identical(np.zeros((2, 3)), np.zeros((2, 4)))
        self.assertIn("shape mismatch", str(cm.exception))

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_nan_matches_nan_but_signed_zero_does_not(self):
        a = np.array([np.nan, 0.0], dtype=np.float64)
        b = np.array([np.nan, 0.0], dtype=np.float64)
        det.assert_bitwise_identical(a, b)          # NaN bits are identical
        c = np.array([np.nan, -0.0], dtype=np.float64)
        with self.assertRaises(det.DeterminismError):
            det.assert_bitwise_identical(a, c)      # different bit patterns


class TestDeterminismEnvironment(unittest.TestCase):
    def test_cublas_workspace_config_set_at_import(self):
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                         det.CUBLAS_WORKSPACE_CONFIG_EFFECTIVE)
        self.assertIn(det.CUBLAS_WORKSPACE_CONFIG_EFFECTIVE, (":4096:8", ":16:8"))

    def test_enforce_determinism_without_torch_is_loud(self):
        if det.torch_available():
            info = det.enforce_determinism(1234)
            self.assertEqual(info["seed"], 1234)
            self.assertTrue(info["use_deterministic_algorithms"])
            self.assertFalse(info["cuda.matmul.allow_tf32"])
            self.assertFalse(info["cudnn.allow_tf32"])
            self.assertTrue(info["cudnn.deterministic"])
            self.assertFalse(info["cudnn.benchmark"])
        else:
            with self.assertRaises(det.DeterminismError):
                det.enforce_determinism(1234)

    def test_modules_import_without_torch_or_cuda(self):
        """Everything must be import-safe on a laptop with no GPU."""
        import importlib
        for name in ("ling", "ling.determinism", "ling.hooks", "ling.model"):
            importlib.import_module(name)


class TestLayerCheckpointWriter(unittest.TestCase):
    """§A.1 rule 3 (checkpoint every layer) and rule 1 (never persist raw acts)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ling_ckpt_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_read_and_resume(self):
        w = LayerCheckpointWriter(self.tmp, "run1", d_model=3840)
        self.assertEqual(w.completed_layers(), [])
        w.write_layer(0, {"assignments": [0, 1, 1, 2], "k": 3, "lambda2": 0.012})
        w.write_layer(2, {"assignments": [0, 0, 1, 1], "k": 2, "lambda2": 0.004})
        self.assertEqual(w.completed_layers(), [0, 2])
        self.assertTrue(w.is_done(0))
        self.assertFalse(w.is_done(1))
        got = w.read_layer(2)
        self.assertEqual(got["layer"], 2)
        self.assertEqual(got["k"], 2)
        # atomic write leaves no partial file behind
        self.assertEqual([f for f in os.listdir(os.path.join(self.tmp, "run1"))
                          if f.endswith(".tmp")], [])

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_refuses_raw_activations(self):
        w = LayerCheckpointWriter(self.tmp, "run2", d_model=8)
        with self.assertRaises(hooks.ExtractionError) as cm:
            w.write_layer(0, {"states": np.zeros((5, 8), dtype=np.float32)})
        self.assertIn("A.1", str(cm.exception))

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_refuses_large_float_blobs(self):
        w = LayerCheckpointWriter(self.tmp, "run3", d_model=8, max_float_elems=100)
        with self.assertRaises(hooks.ExtractionError):
            w.write_layer(0, {"x": np.zeros(101, dtype=np.float64)})
        w.write_layer(1, {"x": np.zeros(99, dtype=np.float64)})   # fine

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_numpy_assignments_are_serialised(self):
        w = LayerCheckpointWriter(self.tmp, "run4", d_model=8)
        w.write_layer(5, {"assignments": np.array([0, 1, 2, 1], dtype=np.int32)})
        self.assertEqual(w.read_layer(5)["assignments"], [0, 1, 2, 1])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestAlignedActivationStore(unittest.TestCase):
    """The prompt-major -> layer-major transpose. §A.1."""

    def _store_roundtrip(self, spill_dir=None):
        L, P, D = 4, 3, 6
        store = hooks.AlignedActivationStore(L, D, spill_dir=spill_dir)
        names = ["a", "b", hooks.FINAL]
        for p in range(2):
            for cond in ("nl_base", "formal"):
                vals = np.arange(L * P * D, dtype=np.float32).reshape(L, P, D)
                vals = vals + 1000 * p + (0 if cond == "nl_base" else 100)
                store.add("prob{}".format(p), cond, names, vals)
        self.assertEqual(store.n_states, 2 * 2 * P)

        seen = []
        for batch in store.iter_layers():
            seen.append(batch.layer)
            self.assertEqual(batch.states.shape, (2 * 2 * P, D))
            self.assertEqual(len(batch.rows), 2 * 2 * P)
            # row provenance is preserved and addressable
            i = batch.index_of("prob1", "formal", hooks.FINAL)
            expected = np.arange(L * P * D, dtype=np.float32).reshape(L, P, D)[
                batch.layer, 2] + 1000 + 100
            np.testing.assert_array_equal(batch.states[i], expected)
        self.assertEqual(seen, [0, 1, 2, 3])
        store.close()
        return store

    def test_ram_backend(self):
        self._store_roundtrip()

    def test_spill_backend_and_cleanup(self):
        tmp = tempfile.mkdtemp(prefix="ling_spill_")
        try:
            spill = os.path.join(tmp, "scratch")
            self._store_roundtrip(spill_dir=spill)
            self.assertFalse(os.path.exists(spill), "spill dir must be deleted")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_refuses_to_spill_into_results_or_data(self):
        for bad in ("results/scratch", "data/scratch"):
            with self.assertRaises(hooks.ExtractionError) as cm:
                hooks.AlignedActivationStore(2, 4, spill_dir=os.path.join(
                    tempfile.gettempdir(), bad))
            self.assertIn("A.1", str(cm.exception))

    def test_release_frees_layers_as_they_are_consumed(self):
        store = hooks.AlignedActivationStore(3, 4)
        store.add("p0", "nl_base", ["a", hooks.FINAL],
                  np.ones((3, 2, 4), dtype=np.float32))
        seen = []
        for batch in store.iter_layers(release=True):
            seen.append(batch.states.shape)
        self.assertEqual(seen, [(2, 4)] * 3)
        # consumed: re-reading a released layer is an error, not silent garbage
        for i in range(3):
            with self.assertRaises(hooks.ExtractionError):
                store.layer(i)
        store.close()

    def test_wrong_shape_rejected(self):
        store = hooks.AlignedActivationStore(4, 6)
        with self.assertRaises(hooks.ExtractionError):
            store.add("p", "nl_base", ["a", hooks.FINAL],
                      np.zeros((4, 3, 6), dtype=np.float32))
        store.close()

    def test_projected_memory_matches_the_planned_corpus(self):
        # 400 prompts x 20 aligned positions x 40 layers x 3840 dims, float32
        b = hooks.AlignedActivationStore.projected_bytes(400, 20, 40, 3840)
        self.assertLess(b / 1e9, 6.0)     # host RAM, not VRAM; documented in A.1

    def test_stream_layers_with_checkpoints_resumes(self):
        tmp = tempfile.mkdtemp(prefix="ling_stream_")
        try:
            store = hooks.AlignedActivationStore(3, 4)
            store.add("p0", "nl_base", ["a", hooks.FINAL],
                      np.zeros((3, 2, 4), dtype=np.float32))
            w = LayerCheckpointWriter(tmp, "run", d_model=4)
            calls = []

            def fn(batch):
                calls.append(batch.layer)
                return {"assignments": [0] * batch.n_states, "k": 1}

            done = hooks.stream_layers_with_checkpoints(store, w, fn)
            self.assertEqual(done, [0, 1, 2])
            self.assertEqual(calls, [0, 1, 2])
            # second call: everything is checkpointed, nothing recomputed
            done2 = hooks.stream_layers_with_checkpoints(store, w, fn)
            self.assertEqual(done2, [])
            self.assertEqual(calls, [0, 1, 2])
            store.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestModelConfig(unittest.TestCase):
    def test_config_loads_and_flags_unverified_checkpoint(self):
        from ling.model import load_config
        cfg = load_config()
        self.assertIn("model_id", cfg)
        self.assertEqual(cfg["attn_implementation"], "eager",
                         "SDPA/flash are not bitwise reproducible across shapes "
                         "(A.4); the G1 gate depends on eager attention")
        self.assertEqual(cfg["dtype"], "bfloat16")     # A.2: no quantization
        self.assertFalse(cfg["thinking"])              # A.2: thinking off
        self.assertIsInstance(cfg["checkpoint_verified"], bool)
        if not cfg["checkpoint_verified"]:
            # this is the expected state until a human checks the Hub (A.2)
            self.assertTrue(cfg["model_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
