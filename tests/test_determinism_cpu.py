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

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import determinism as det          # noqa: E402
from ling import hooks                        # noqa: E402
from ling import model as lmodel              # noqa: E402
from ling.hooks import (                      # noqa: E402
    AnchorDropError, AnchorError, AnchorKeyMismatch, DuplicateAnchorPosition,
    FINAL, FinalTokenError, LayerCheckpointWriter, anchor_order,
    char_span_to_token_index, check_anchor_keys_consistent,
    layer_pair_indices, layer_pair_members, n_layer_pairs,
    normalize_prompt_text, resolve_anchor_positions, validate_final_token,
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
    #: Every real HF tokenizer exposes this; resolve_anchor_positions REQUIRES
    #: it, because the FINAL position is checked against the special-token set.
    all_special_ids = [1, 2]

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


class WhitespaceTokenTokenizer:
    """Fake fast tokenizer that emits whitespace runs as their OWN tokens.

    This is what the real Gemma/Llama BPE tokenizers do with newlines, and it is
    the case :class:`FakeFastTokenizer` cannot express (it drops whitespace
    entirely). A trailing ``"\\n"`` therefore arrives as a token with a REAL,
    non-zero-width offset span — which is exactly why the zero-width
    trailing-special guard does not catch it.
    """

    is_fast = True
    all_special_ids = [1]

    def __init__(self, add_bos=True):
        self.add_bos = add_bos

    @staticmethod
    def spans(text):
        out, i, n = [], 0, len(text)
        while i < n:
            c, j = text[i], i
            if c in ("\n", " ", "\t"):
                while j < n and text[j] == c:
                    j += 1
            else:
                kind = c.isdigit()
                while j < n and not text[j].isspace() and text[j].isdigit() == kind:
                    j += 1
            out.append((i, j))
            i = j
        return out

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True,
                 return_tensors=None, padding=False, truncation=False, **kw):
        offsets = self.spans(text)
        ids = list(range(100, 100 + len(offsets)))
        if add_special_tokens and self.add_bos:
            offsets.insert(0, (0, 0))
            ids.insert(0, 1)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


TEXT = "Natalia sold 48 clips in April and 24 in May."


def _names(*quantities):
    """The shared ordered anchor key list for a problem (§A.6)."""
    return sorted(quantities) + [FINAL]
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
    NAMES = ["num_clips_april", "num_clips_may", FINAL]

    def test_end_to_end(self):
        tok = FakeFastTokenizer()
        anchors = {
            "num_clips_april": [TEXT.index("48"), TEXT.index("48") + 2],
            "num_clips_may": [TEXT.index("24"), TEXT.index("24") + 2],
            FINAL: [-1, -1],
        }
        res = resolve_anchor_positions(TEXT, anchors, tok, expected_names=self.NAMES)
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
            resolve_anchor_positions(TEXT, {FINAL: [-1, -1]}, SlowTokenizer(),
                                     expected_names=[FINAL])
        self.assertIn("fast tokenizer", str(cm.exception))

    def test_duplicate_positions_are_rejected_BY_DEFAULT(self):
        """F2: the default must be reject, not allow.

        Two anchors on one token produce two byte-identical rows, i.e. a
        cosine-affinity-1.0 node pair, which changes the degree distribution and
        therefore lambda2 — the H3 endpoint — and the node count path entropy
        normalises by. It used to default to ALLOW, and extract_aligned never
        overrode it.
        """
        tok = FakeFastTokenizer()
        pos = TEXT.index("48")
        anchors = {"a": [pos, pos + 2], "b": [pos, pos + 2], FINAL: [-1, -1]}
        names = ["a", "b", FINAL]
        with self.assertRaises(DuplicateAnchorPosition) as cm:
            resolve_anchor_positions(TEXT, anchors, tok, expected_names=names)
        exc = cm.exception
        self.assertIsInstance(exc, AnchorDropError)          # countable for G3
        self.assertEqual(exc.reason, "duplicate_anchor_position")
        self.assertEqual(sorted(exc.duplicates), ["a", "b"])
        # ... and it is opt-out only, never the default
        res = resolve_anchor_positions(TEXT, anchors, tok, expected_names=names,
                                       allow_duplicate_positions=True)
        self.assertEqual(res.token_indices[0], res.token_indices[1])

    def test_duplicate_collision_is_asymmetric_between_conditions(self):
        """F2, the reason it is not a wash: `formal` collides, prose does not."""
        tok = WhitespaceTokenTokenizer()
        formal = "num_clips_april = 48\nnum_clips_may = 24"
        prose = "Natalia sold 48 clips in April and 24 in May."
        names = ["num_clips_april", "num_clips_may", FINAL]

        a_prose = {"num_clips_april": [prose.index("48"), prose.index("48") + 2],
                   "num_clips_may": [prose.index("24"), prose.index("24") + 2],
                   FINAL: [-1, -1]}
        r = resolve_anchor_positions(prose, a_prose, tok, expected_names=names,
                                     problem_id="p1", condition="nl_base")
        self.assertEqual(len(set(r.token_indices)), 3)       # prose: no collision

        a_formal = {"num_clips_april": [formal.index("48"), formal.index("48") + 2],
                    "num_clips_may": [formal.rindex("24"), formal.rindex("24") + 2],
                    FINAL: [-1, -1]}
        with self.assertRaises(DuplicateAnchorPosition) as cm:
            resolve_anchor_positions(formal, a_formal, tok, expected_names=names,
                                     problem_id="p1", condition="formal")
        self.assertEqual(cm.exception.problem_id, "p1")
        self.assertEqual(cm.exception.condition, "formal")

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
        names = ["n_clips", "n_days", FINAL]
        r_nl = resolve_anchor_positions(nl, a_nl, tok, expected_names=names)
        r_fm = resolve_anchor_positions(formal, a_fm, tok, expected_names=names)
        self.assertEqual(r_nl.names, r_fm.names)
        # the quantity appears BEFORE the other one in NL and AFTER it in the
        # formal rewrite; the column order must not follow the text order
        self.assertLess(r_nl.token_indices[0], r_nl.token_indices[1])
        self.assertGreater(r_fm.token_indices[0], r_fm.token_indices[1])
        for name, r, txt in (("n_clips", r_nl, nl), ("n_clips", r_fm, formal)):
            idx = r.token_indices[r.names.index(name)]
            s, e = r.offsets[idx]
            self.assertEqual(txt[s:e].strip(), "48")


class TestExpectedNamesInvariantIsUnskippable(unittest.TestCase):
    """F1 — §A.6's cross-condition anchor-key invariant, enforced at the API.

    The invariant used to be defined (``check_anchor_keys_consistent``) and
    tested, but never called from any production path. If ``formal`` is missing
    one quantity it contributes fewer token states per problem than
    ``nl_base``; lambda2 is strongly node-count dependent, so H3's endpoint
    absorbs a pure graph-size difference — in its pre-committed direction.
    """

    def setUp(self):
        self.tok = FakeFastTokenizer()
        self.anchors = {
            "num_clips_april": [TEXT.index("48"), TEXT.index("48") + 2],
            "num_clips_may": [TEXT.index("24"), TEXT.index("24") + 2],
            FINAL: [-1, -1],
        }

    def test_expected_names_is_a_REQUIRED_parameter(self):
        """Not a default that a caller can forget: Python must refuse the call."""
        for fn in (hooks.resolve_anchor_positions, hooks.extract_aligned):
            sig = inspect.signature(fn)
            p = sig.parameters["expected_names"]
            self.assertIs(p.default, inspect.Parameter.empty,
                          "{}: expected_names must have NO default".format(fn.__name__))
            self.assertEqual(p.kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(TypeError):
            resolve_anchor_positions(TEXT, self.anchors, self.tok)

    def test_missing_quantity_in_one_condition_is_a_typed_countable_drop(self):
        shared = ["num_clips_april", "num_clips_may", FINAL]
        short = dict(self.anchors)
        short.pop("num_clips_may")           # `formal` forgot a quantity
        with self.assertRaises(AnchorKeyMismatch) as cm:
            resolve_anchor_positions(TEXT, short, self.tok, expected_names=shared,
                                     problem_id="gsms_0007", condition="formal")
        exc = cm.exception
        self.assertIsInstance(exc, AnchorDropError)
        self.assertEqual(exc.reason, "anchor_key_mismatch")
        self.assertEqual(exc.problem_id, "gsms_0007")
        self.assertEqual(exc.condition, "formal")
        self.assertEqual(exc.missing, ["num_clips_may"])
        self.assertIn("gsms_0007", str(exc))
        self.assertIn("node-count", str(exc))

    def test_extra_quantity_in_one_condition_is_also_a_drop(self):
        shared = ["num_clips_april", FINAL]
        with self.assertRaises(AnchorKeyMismatch) as cm:
            resolve_anchor_positions(TEXT, self.anchors, self.tok,
                                     expected_names=shared, condition="nl_para")
        self.assertEqual(cm.exception.unexpected, ["num_clips_may"])

    def test_expected_names_must_be_in_canonical_order(self):
        """A permuted key list would permute the columns between conditions."""
        with self.assertRaises(AnchorError) as cm:
            resolve_anchor_positions(
                TEXT, self.anchors, self.tok,
                expected_names=[FINAL, "num_clips_april", "num_clips_may"])
        self.assertIn("canonical order", str(cm.exception))

    def test_expected_names_must_contain_FINAL(self):
        with self.assertRaises(AnchorError):
            resolve_anchor_positions(TEXT, self.anchors, self.tok,
                                     expected_names=["num_clips_april"])

    def test_checker_output_is_exactly_what_the_extractor_wants(self):
        """The two ends of the invariant must line up without adaptation."""
        realizations = {
            c: {"anchors": dict(self.anchors)}
            for c in ("nl_base", "nl_para", "formal", "nl_break")
        }
        shared = check_anchor_keys_consistent(realizations, problem_id="gsms_1")
        for cond in realizations:
            res = resolve_anchor_positions(
                TEXT, realizations[cond]["anchors"], self.tok,
                expected_names=shared, problem_id="gsms_1", condition=cond)
            self.assertEqual(res.names, shared)

    def test_extract_problem_is_where_the_invariant_is_actually_called(self):
        """The production path. It reaches the key check before touching torch."""
        rec = {
            "problem_id": "gsms_0007",
            "realizations": {
                "nl_base": {"text": TEXT, "anchors": dict(self.anchors)},
                "nl_para": {"text": TEXT, "anchors": dict(self.anchors)},
                "nl_break": {"text": TEXT, "anchors": dict(self.anchors)},
                # `formal` forgot a quantity
                "formal": {"text": TEXT,
                           "anchors": {"num_clips_april": [0, 7], FINAL: [-1, -1]}},
            },
        }
        with self.assertRaises(AnchorKeyMismatch) as cm:
            hooks.extract_problem(rec, self.tok, object())
        self.assertEqual(cm.exception.problem_id, "gsms_0007")
        self.assertIn("formal", cm.exception.condition)

    def test_checker_raises_the_same_typed_drop(self):
        bad = {
            "nl_base": {"anchors": dict(self.anchors)},
            "formal": {"anchors": {"num_clips_april": [0, 1], FINAL: [-1, -1]}},
        }
        with self.assertRaises(AnchorKeyMismatch) as cm:
            check_anchor_keys_consistent(bad, problem_id="gsms_9")
        self.assertEqual(cm.exception.problem_id, "gsms_9")
        self.assertEqual(cm.exception.reason, "anchor_key_mismatch")


class TestFinalTokenIsRealContent(unittest.TestCase):
    """F3 — FINAL must not land on a trailing newline.

    A trailing ``"\\n"`` is a token of its own with a real, non-zero-width
    offset span, so the trailing-SPECIAL guard passes it. FINAL would then mean
    "the newline" for a Python rewrite and "the sentence-final period" for
    prose: a systematic, condition-correlated difference in one of only ~4-6
    aligned positions, and the one H2's fork layer is read at.
    """

    FORMAL = "num_clips_april = 48\nnum_clips_may = 24\n"
    PROSE = "Natalia sold 48 clips in April and 24 in May."

    def setUp(self):
        self.tok = WhitespaceTokenTokenizer()

    def test_the_fixture_really_does_produce_a_trailing_newline_token(self):
        """Guard the guard: if this stops holding the test below proves nothing."""
        offs = self.tok(self.FORMAL, return_offsets_mapping=True)["offset_mapping"]
        s, e = offs[-1]
        self.assertGreater(e, s, "trailing newline must be a NON-zero-width token")
        self.assertEqual(self.FORMAL[s:e], "\n")

    def test_trailing_newline_never_becomes_FINAL(self):
        anchors = {"num_clips_may": [self.FORMAL.rindex("24"),
                                     self.FORMAL.rindex("24") + 2],
                   FINAL: [-1, -1]}
        res = resolve_anchor_positions(
            self.FORMAL, anchors, self.tok, expected_names=["num_clips_may", FINAL],
            problem_id="p", condition="formal", allow_duplicate_positions=True)
        self.assertTrue(res.text_was_rstripped)
        self.assertNotEqual(res.token_text(FINAL), "\n")
        self.assertEqual(res.token_text(FINAL).strip(), "24")

    def test_whitespace_FINAL_is_rejected_by_a_positive_check(self):
        """The zero-width guard alone does NOT catch this; assert the positive one."""
        offs = self.tok(self.FORMAL, return_offsets_mapping=True)["offset_mapping"]
        ids = self.tok(self.FORMAL)["input_ids"]
        last = len(offs) - 1
        ts, te = offs[last]
        self.assertGreater(te, ts)          # would pass the old zero-width guard
        with self.assertRaises(FinalTokenError) as cm:
            validate_final_token(last, offs, text=self.FORMAL, input_ids=ids,
                                 special_ids=self.tok.all_special_ids,
                                 problem_id="p", condition="formal")
        self.assertIn("whitespace-only", str(cm.exception))
        self.assertIsInstance(cm.exception, AnchorDropError)

    def test_whitespace_FINAL_is_rejected_even_with_allow_trailing_special(self):
        offs = self.tok(self.FORMAL, return_offsets_mapping=True)["offset_mapping"]
        with self.assertRaises(FinalTokenError):
            validate_final_token(len(offs) - 1, offs, text=self.FORMAL,
                                 allow_trailing_special=True)

    def test_special_id_FINAL_is_rejected_even_with_a_real_span(self):
        offs = [(0, 3), (3, 8)]
        with self.assertRaises(FinalTokenError) as cm:
            validate_final_token(1, offs, text="abc defgh", input_ids=[100, 1],
                                 special_ids=[1])
        self.assertIn("all_special_ids", str(cm.exception))

    def test_conditions_end_on_comparable_positions_after_normalisation(self):
        """The point of the fix: FINAL means 'last content token' in BOTH."""
        names = ["num_clips_may", FINAL]
        r_f = resolve_anchor_positions(
            self.FORMAL, {"num_clips_may": [self.FORMAL.rindex("24"),
                                            self.FORMAL.rindex("24") + 2],
                          FINAL: [-1, -1]},
            self.tok, expected_names=names, allow_duplicate_positions=True)
        r_p = resolve_anchor_positions(
            self.PROSE, {"num_clips_may": [self.PROSE.index("24"),
                                           self.PROSE.index("24") + 2],
                         FINAL: [-1, -1]},
            self.tok, expected_names=names)
        for r in (r_f, r_p):
            self.assertNotEqual(r.token_text(FINAL).strip(), "")
            self.assertEqual(r.token_indices[r.names.index(FINAL)], r.n_tokens - 1)

    def test_normalize_prompt_text_is_rstrip_only(self):
        self.assertEqual(normalize_prompt_text("a = 1\n\n  "), "a = 1")
        self.assertEqual(normalize_prompt_text("a = 1"), "a = 1")
        with self.assertRaises(AnchorError):
            normalize_prompt_text("   \n ")            # empty after stripping
        with self.assertRaises(AnchorError) as cm:
            normalize_prompt_text("  leading")          # would shift every span
        self.assertIn("anchor character span", str(cm.exception))

    def test_tokenizer_without_all_special_ids_is_refused(self):
        class NoSpecials(WhitespaceTokenTokenizer):
            def __getattribute__(self, name):
                if name == "all_special_ids":
                    raise AttributeError(name)
                return object.__getattribute__(self, name)

        tok = NoSpecials()
        self.assertFalse(hasattr(tok, "all_special_ids"))
        with self.assertRaises(AnchorError) as cm:
            resolve_anchor_positions(self.PROSE, {FINAL: [-1, -1]}, tok,
                                     expected_names=[FINAL])
        self.assertIn("all_special_ids", str(cm.exception))


class TestGateProbes(unittest.TestCase):
    """The gate's own probes must model what extraction actually sees (F3)."""

    def _gate_module(self):
        path = os.path.join(_REPO, "scripts", "01_determinism_gate.py")
        spec = importlib.util.spec_from_file_location("gate_g1", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # also asserts import-safety w/o torch
        return mod

    def test_no_probe_ends_in_whitespace(self):
        mod = self._gate_module()
        for i, p in enumerate(mod.DEFAULT_PROBES):
            self.assertEqual(p, p.rstrip(),
                             "probe {} ends in whitespace: its final token would "
                             "be a newline, not a content token".format(i))


class TestLayerPairConvention(unittest.TestCase):
    """Amendment 3. An off-by-one here is §A.11 item 2 one level up."""

    def test_L_minus_one_pairs(self):
        self.assertEqual(n_layer_pairs(48), 47)     # gemma-4-12B-it: n for H3
        self.assertEqual(n_layer_pairs(2), 1)
        self.assertEqual(layer_pair_indices(48)[0], 0)
        self.assertEqual(layer_pair_indices(48)[-1], 46)
        self.assertEqual(len(layer_pair_indices(48)), 47)

    def test_pair_p_is_blocks_p_and_p_plus_1(self):
        self.assertEqual(layer_pair_members(0, 48), (0, 1))
        self.assertEqual(layer_pair_members(46, 48), (46, 47))

    def test_there_is_no_pair_for_the_last_block(self):
        with self.assertRaises(hooks.ExtractionError) as cm:
            layer_pair_members(47, 48)
        self.assertIn("no successor", str(cm.exception))
        with self.assertRaises(hooks.ExtractionError):
            layer_pair_members(-1, 48)

    def test_single_block_model_has_no_pairs(self):
        with self.assertRaises(hooks.ExtractionError):
            n_layer_pairs(1)

    def test_convention_is_documented_in_the_module_docstring(self):
        doc = hooks.__doc__
        self.assertIn("LAYER-PAIR INDEX CONVENTION", doc)
        self.assertIn("47 layer-pairs", doc)


class _FakeStates:
    """Stand-in for a numpy [N, d] block, so the store is testable w/o numpy."""

    def __init__(self, n, d, layer):
        self.shape = (n, d)
        self.layer = layer


class FakeStore(hooks.AlignedActivationStore):
    """AlignedActivationStore with the numpy-backed loader replaced.

    Deliberately bypasses ``__init__`` (which requires numpy): the pair-sweep
    logic, the release semantics and the resume skip-set are pure index
    arithmetic and must be testable on a machine with no numpy — which is
    exactly where a layer-pair off-by-one would otherwise go unnoticed.
    """

    def __init__(self, n_layers, n_rows=4, d_model=6):
        self.n_layers = n_layers
        self.d_model = d_model
        self.dtype = None
        self.rows = [hooks.StateRow("p{}".format(i // 2), "nl_base", "a")
                     for i in range(n_rows)]
        self._closed = False
        self._buffers = [[] for _ in range(n_layers)]
        self._spill_dir = None
        self._released = set()
        self.loads = []

    def _load_layer(self, i):
        if i in self._released:
            raise hooks.ExtractionError(
                "layer {} was released after being streamed".format(i))
        self.loads.append(i)
        return _FakeStates(len(self.rows), self.d_model, i)


class TestIterLayerPairs(unittest.TestCase):
    """Friction A: lambda2 is a TWO-layer statistic; give it a two-layer API."""

    def test_full_sweep_yields_L_minus_1_pairs_in_order(self):
        store = FakeStore(48)
        pairs = [(lp.pair, lp.source.layer, lp.target.layer)
                 for lp in store.iter_layer_pairs()]
        self.assertEqual(len(pairs), 47)
        self.assertEqual(pairs[0], (0, 0, 1))
        self.assertEqual(pairs[-1], (46, 46, 47))
        self.assertEqual([p for p, _, _ in pairs], list(range(47)))

    def test_each_layer_is_loaded_once_not_twice(self):
        """The source of pair p is the target of pair p-1; reuse it."""
        store = FakeStore(6)
        list(store.iter_layer_pairs())
        self.assertEqual(store.loads, [0, 1, 2, 3, 4, 5])

    def test_pair_rejects_non_adjacent_or_misordered_batches(self):
        a = hooks.LayerBatch(layer=3, states=_FakeStates(2, 6, 3),
                             rows=[hooks.StateRow("p", "c", "a")] * 2)
        b = hooks.LayerBatch(layer=5, states=_FakeStates(2, 6, 5),
                             rows=[hooks.StateRow("p", "c", "a")] * 2)
        with self.assertRaises(hooks.ExtractionError) as cm:
            hooks.LayerPair(pair=3, source=a, target=b)
        self.assertIn("Amendment 3", str(cm.exception))
        with self.assertRaises(hooks.ExtractionError):
            hooks.LayerPair(pair=4, source=b, target=b)

    def test_release_true_on_iter_layers_cannot_silently_break_a_pair_sweep(self):
        store = FakeStore(4)
        list(store.iter_layers(release=True))
        with self.assertRaises(hooks.ExtractionError) as cm:
            list(store.iter_layer_pairs())
        msg = str(cm.exception)
        self.assertIn("release=True", msg)
        self.assertIn("TWO-layer", msg)
        # and it refuses UP FRONT, before yielding a single pair
        self.assertEqual(store.loads, [0, 1, 2, 3])   # only the iter_layers sweep

    def test_pair_release_frees_a_layer_only_after_its_last_use(self):
        store = FakeStore(4)
        seen = []
        for lp in store.iter_layer_pairs(release=True):
            seen.append(lp.pair)
            # both members are readable while the pair is being consumed
            self.assertEqual(lp.source.states.shape[0], len(store.rows))
            self.assertEqual(lp.target.states.shape[0], len(store.rows))
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(store._released, {0, 1, 2, 3})

    def test_out_of_range_pair_is_refused(self):
        store = FakeStore(4)
        with self.assertRaises(hooks.ExtractionError):
            list(store.iter_layer_pairs([3]))          # pairs are 0..2

    def test_pair_exposes_shared_row_provenance(self):
        store = FakeStore(3, n_rows=6)
        lp = next(iter(store.iter_layer_pairs()))
        self.assertEqual(len(lp.rows), 6)
        self.assertEqual(lp.n_states, 6)
        self.assertEqual(lp.layers(), (0, 1))


class TestStreamingSkipSet(unittest.TestCase):
    """F8: a resume must not materialise the layers it is about to skip."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ling_skip_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_completed_layers_are_never_loaded_on_resume(self):
        store = FakeStore(5)
        w = LayerCheckpointWriter(self.tmp, "run", d_model=6)
        hooks.stream_layers_with_checkpoints(
            store, w, lambda b: {"assignments": [0] * b.n_states})
        self.assertEqual(store.loads, [0, 1, 2, 3, 4])

        store2 = FakeStore(5)
        computed = hooks.stream_layers_with_checkpoints(
            store2, w, lambda b: {"assignments": [0] * b.n_states})
        self.assertEqual(computed, [])
        self.assertEqual(store2.loads, [],
                         "resume loaded layers it then skipped; on the spill "
                         "backend that is 40 layers read off disk and dropped")

    def test_partial_resume_loads_only_the_remaining_layers(self):
        store = FakeStore(5)
        w = LayerCheckpointWriter(self.tmp, "run2", d_model=6)
        w.write_layer(0, {"k": 1})
        w.write_layer(3, {"k": 1})
        computed = hooks.stream_layers_with_checkpoints(
            store, w, lambda b: {"assignments": [0] * b.n_states})
        self.assertEqual(computed, [1, 2, 4])
        self.assertEqual(store.loads, [1, 2, 4])

    def test_release_is_available_but_off_by_default(self):
        sig = inspect.signature(hooks.stream_layers_with_checkpoints)
        self.assertIs(sig.parameters["release"].default, False)
        store = FakeStore(3)
        w = LayerCheckpointWriter(self.tmp, "run3", d_model=6)
        hooks.stream_layers_with_checkpoints(
            store, w, lambda b: {"k": 1}, release=True)
        self.assertEqual(store._released, {0, 1, 2})

    def test_pair_stream_refuses_a_layer_namespace_writer(self):
        store = FakeStore(4)
        w = LayerCheckpointWriter(self.tmp, "run4", d_model=6)   # prefix="layer"
        with self.assertRaises(hooks.ExtractionError) as cm:
            hooks.stream_layer_pairs_with_checkpoints(store, w, lambda lp: {"l2": 0.1})
        self.assertIn("prefix='pair'", str(cm.exception))

    def test_pair_stream_checkpoints_and_resumes_by_PAIR_index(self):
        store = FakeStore(4)
        w = LayerCheckpointWriter(self.tmp, "run5", d_model=6, prefix="pair")
        done = hooks.stream_layer_pairs_with_checkpoints(
            store, w, lambda lp: {"lambda2": 0.1 * lp.pair})
        self.assertEqual(done, [0, 1, 2])            # 4 blocks -> 3 pairs
        rec = w.read_layer(1)
        self.assertEqual(rec["pair"], 1)
        self.assertEqual((rec["source_layer"], rec["target_layer"]), (1, 2))
        store2 = FakeStore(4)
        self.assertEqual(hooks.stream_layer_pairs_with_checkpoints(
            store2, w, lambda lp: {"lambda2": 0.0}), [])
        self.assertEqual(store2.loads, [])
        # pair and layer namespaces do not collide
        wl = LayerCheckpointWriter(self.tmp, "run5", d_model=6)
        self.assertEqual(wl.completed_layers(), [])


class TestModelAttnReadback(unittest.TestCase):
    """F5: check the EFFECTIVE attention implementation, not the requested one."""

    class _Cfg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _Model:
        def __init__(self, config):
            self.config = config

    def test_reads_back_from_the_private_attribute(self):
        m = self._Model(self._Cfg(_attn_implementation="eager"))
        self.assertEqual(lmodel.read_back_attn_implementations(m),
                         {"config": "eager"})
        lmodel.assert_attn_implementation(m, "eager")

    def test_a_silent_fallback_to_sdpa_is_caught(self):
        m = self._Model(self._Cfg(_attn_implementation="sdpa"))
        with self.assertRaises(lmodel.ModelLoadError) as cm:
            lmodel.assert_attn_implementation(m, "eager")
        self.assertIn("sdpa", str(cm.exception))

    def test_a_subconfig_keeping_its_own_implementation_is_caught(self):
        """A wrapped/multimodal checkpoint: top-level says eager, text tower does not."""
        m = self._Model(self._Cfg(
            _attn_implementation="eager",
            text_config=self._Cfg(_attn_implementation="sdpa")))
        with self.assertRaises(lmodel.ModelLoadError) as cm:
            lmodel.assert_attn_implementation(m, "eager")
        self.assertIn("text_config", str(cm.exception))

    def test_a_config_that_exposes_nothing_is_not_assumed_to_be_eager(self):
        m = self._Model(self._Cfg())
        with self.assertRaises(lmodel.ModelLoadError):
            lmodel.assert_attn_implementation(m, "eager")

    def test_load_model_records_the_readback_not_the_request(self):
        src = inspect.getsource(lmodel.load_model)
        self.assertIn("assert_attn_implementation(model, requested_attn)", src)
        self.assertIn("attn_implementation_effective", src)


class TestDeterminismSummaryIsDerived(unittest.TestCase):
    """F4: the methods sentence may not claim a control it did not verify."""

    def _good(self):
        applied = dict(det.REQUIRED_CONTROLS)
        applied.update({
            "torch_version": "2.6.0", "cuda_version": "12.4",
            "cuda_device_name": "NVIDIA A40", "seed": 20260906,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "problems": [],
        })
        return applied

    def test_all_controls_in_force_passes(self):
        status = det.determinism_controls_status(self._good())
        self.assertTrue(status["ok"], status["detail"])
        line = det.determinism_summary_line(self._good())
        self.assertIn("TF32 off for matmul", line)
        self.assertIn("bf16 reduced-precision reductions off", line)
        self.assertNotIn("NOT", line)

    def test_a_control_absent_from_the_torch_build_is_not_claimed(self):
        applied = self._good()
        applied["cuda.matmul.allow_bf16_reduced_precision_reduction"] = det.UNAVAILABLE
        line = det.determinism_summary_line(applied)
        self.assertIn(det.UNAVAILABLE, line)
        self.assertNotIn("bf16 reduced-precision reductions off,", line)
        status = det.determinism_controls_status(applied)
        self.assertFalse(status["ok"])
        self.assertIn("cuda.matmul.allow_bf16_reduced_precision_reduction",
                      status["unavailable"])

    def test_a_control_never_applied_is_not_claimed(self):
        applied = self._good()
        del applied["sdpa_backend"]
        line = det.determinism_summary_line(applied)
        self.assertIn("SDPA backend NOT APPLIED", line)
        self.assertFalse(det.determinism_controls_status(applied)["ok"])

    def test_a_control_set_to_the_wrong_value_is_reported_as_such(self):
        applied = self._good()
        applied["cuda.matmul.allow_tf32"] = True
        line = det.determinism_summary_line(applied)
        self.assertIn("TF32 ON for matmul", line)
        status = det.determinism_controls_status(applied)
        self.assertFalse(status["ok"])
        self.assertTrue(any(k == "cuda.matmul.allow_tf32" for k, _, _ in status["wrong"]))

    def test_a_swallowed_sdpa_exception_is_visible(self):
        applied = self._good()
        applied["sdpa_backend"] = "FAILED (RuntimeError: nope)"
        self.assertFalse(det.determinism_controls_status(applied)["ok"])
        self.assertIn("SDPA backend NOT restricted",
                      det.determinism_summary_line(applied))

    def test_eager_attention_is_not_a_string_literal(self):
        """The old sentence said 'eager attention' whatever the model loaded."""
        line = det.determinism_summary_line(self._good())
        self.assertNotIn("eager", line)
        line2 = det.determinism_summary_line(
            self._good(), {"dtype": "torch.bfloat16",
                           "attn_implementation_effective": "sdpa",
                           "attn_implementation_requested": "eager"})
        self.assertIn("sdpa", line2)

    def test_an_illegal_cublas_config_fails_the_status(self):
        applied = self._good()
        applied["CUBLAS_WORKSPACE_CONFIG"] = ""
        self.assertFalse(det.determinism_controls_status(applied)["ok"])

    def test_reported_preconditions_fail_the_status(self):
        applied = self._good()
        applied["problems"] = ["CUDA was already initialised"]
        self.assertFalse(det.determinism_controls_status(applied)["ok"])

    def test_the_gate_uses_the_status_in_its_pass_predicate(self):
        path = os.path.join(_REPO, "scripts", "01_determinism_gate.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("determinism_controls_status(det)", src)
        self.assertIn('controls["ok"]', src)
        self.assertIn("attn_implementation_effective", src)


class TestCublasWorkspaceValidation(unittest.TestCase):
    """F6: an empty CUBLAS_WORKSPACE_CONFIG is not a value, it is a typo."""

    def _run(self, value):
        env = dict(os.environ)
        if value is None:
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            env["CUBLAS_WORKSPACE_CONFIG"] = value
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from ling import determinism as d;"
            "print(repr(d.CUBLAS_WORKSPACE_CONFIG_EFFECTIVE), d._CUBLAS_SET_BY_US)"
            % os.path.join(_REPO, "src")
        )
        return subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True)

    def test_unset_is_filled_in_and_recorded_as_ours(self):
        out = self._run(None)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("':4096:8' True", out.stdout)

    def test_empty_string_is_treated_as_unset_not_as_a_value(self):
        """`export CUBLAS_WORKSPACE_CONFIG=` used to disable BOTH the pinning
        and the CUDA-ordering precondition check that would have noticed."""
        out = self._run("")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("':4096:8' True", out.stdout)

    def test_whitespace_only_is_treated_as_unset(self):
        out = self._run("   ")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("':4096:8' True", out.stdout)

    def test_an_illegal_value_raises_at_import(self):
        out = self._run(":8:4")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("DeterminismError", out.stderr)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", out.stderr)

    def test_the_other_legal_value_is_accepted_and_not_ours(self):
        out = self._run(":16:8")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("':16:8' False", out.stdout)


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

    # ---- F7: the guard had three holes -------------------------------------
    def test_d_model_is_REQUIRED(self):
        """It used to default to None, which disabled the strongest guard."""
        sig = inspect.signature(LayerCheckpointWriter.__init__)
        self.assertIs(sig.parameters["d_model"].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            LayerCheckpointWriter(self.tmp, "run_nodm")
        with self.assertRaises(hooks.ExtractionError):
            LayerCheckpointWriter(self.tmp, "run_nodm", d_model=None)

    def test_refuses_a_tolist_ed_activation_matrix(self):
        """`.tolist()` is still a raw activation; json.dump does not care."""
        w = LayerCheckpointWriter(self.tmp, "run_list", d_model=8)
        acts = [[0.1 * j for j in range(8)] for _ in range(4)]
        with self.assertRaises(hooks.ExtractionError) as cm:
            w.write_layer(0, {"states": acts})
        self.assertIn("A.1", str(cm.exception))
        self.assertIn("tolist", str(cm.exception))

    def test_refuses_a_single_activation_vector_as_a_flat_list(self):
        w = LayerCheckpointWriter(self.tmp, "run_vec", d_model=8)
        with self.assertRaises(hooks.ExtractionError):
            w.write_layer(0, {"centroid": [0.5] * 8})

    def test_the_guard_recurses_into_nested_dicts(self):
        w = LayerCheckpointWriter(self.tmp, "run_nest", d_model=8)
        payload = {"diagnostics": {"layer": {"states": [[0.0] * 8, [1.0] * 8]}}}
        with self.assertRaises(hooks.ExtractionError) as cm:
            w.write_layer(0, payload)
        self.assertIn("diagnostics.layer.states", str(cm.exception))

    def test_large_nested_float_lists_hit_the_element_guard(self):
        w = LayerCheckpointWriter(self.tmp, "run_big", d_model=8, max_float_elems=50)
        with self.assertRaises(hooks.ExtractionError):
            w.write_layer(0, {"x": [[0.0] * 10 for _ in range(6)]})
        w.write_layer(1, {"x": [[0.0] * 5 for _ in range(4)]})     # fine

    def test_integer_assignments_are_never_mistaken_for_activations(self):
        """A d_model-length list of cluster ids is legitimate; do not over-reject."""
        w = LayerCheckpointWriter(self.tmp, "run_ints", d_model=8)
        w.write_layer(0, {"assignments": [0, 1, 2, 3, 1, 0, 2, 1]})
        self.assertEqual(len(w.read_layer(0)["assignments"]), 8)

    def test_nested_dicts_and_lists_survive_serialisation(self):
        w = LayerCheckpointWriter(self.tmp, "run_ser", d_model=3840)
        w.write_layer(2, {"per_condition": {"formal": {"lambda2": 0.01, "k": 9}},
                          "eigengaps": [0.4, 0.2, 0.1]})
        got = w.read_layer(2)
        self.assertEqual(got["per_condition"]["formal"]["k"], 9)
        self.assertEqual(got["eigengaps"], [0.4, 0.2, 0.1])

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_refuses_complex_arrays(self):
        """np.linalg.eig on a non-symmetric transition matrix returns complex128,
        whose dtype.kind is 'c' — the old guard only looked at 'f'."""
        w = LayerCheckpointWriter(self.tmp, "run_cx", d_model=8, max_float_elems=100)
        with self.assertRaises(hooks.ExtractionError):
            w.write_layer(0, {"eigvals": np.zeros(200, dtype=np.complex128)})

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_refuses_a_nested_ndarray(self):
        w = LayerCheckpointWriter(self.tmp, "run_nda", d_model=8)
        with self.assertRaises(hooks.ExtractionError):
            w.write_layer(0, {"d": {"states": np.zeros((3, 8), dtype=np.float32)}})


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

    def test_problem_and_condition_indices(self):
        """Resampling unit is the PROBLEM (§A.9.2, prereg decision 3)."""
        store = hooks.AlignedActivationStore(2, 4)
        names = ["a", hooks.FINAL]
        for p in ("p0", "p1"):
            for cond in ("nl_base", "nl_para", "formal", "nl_break"):
                store.add(p, cond, names, np.zeros((2, 2, 4), dtype=np.float32))
        batch = store.layer(0)
        pidx = batch.problem_index()
        self.assertEqual(sorted(pidx), ["p0", "p1"])
        # each problem contributes 4 conditions x 2 anchors = 8 rows
        self.assertEqual([len(v) for v in pidx.values()], [8, 8])
        # and a resampled problem carries all four conditions with it
        conds = {batch.rows[i].condition for i in pidx["p0"]}
        self.assertEqual(conds, {"nl_base", "nl_para", "formal", "nl_break"})
        cidx = batch.condition_index()
        self.assertEqual(sorted(cidx), ["formal", "nl_base", "nl_break", "nl_para"])
        self.assertEqual([len(v) for v in cidx.values()], [4, 4, 4, 4])
        store.close()

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
