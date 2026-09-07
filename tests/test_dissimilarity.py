"""Tests for src/ling/dissimilarity.py and scripts/025_covariates.py.

Three things are checked, and they are the three that can silently corrupt H2:

  1. the distances are the distances -- hand-computed Levenshtein values and Jaccard on
     toy sets, so a rewrite of the DP loop cannot quietly change what "lexical
     dissimilarity" means;
  2. no lexical proxy is ever promoted to a syntactic covariate, and nothing is imputed
     when no parser is present (DESIGN.md A.7: the syntactic-vs-lexical contrast IS the
     contribution over Han et al.);
  3. the emitted JSON survives `scripts/04_analyze.py`'s covariate handling, which takes
     the UNION of keys across pairs and then `float()`s every key on every pair -- a null
     or a key present on only some pairs is a SystemExit in the middle of the primary
     analysis.

Run with:  python3 -m unittest discover -s tests -v
(stdlib unittest; the syntactic tests skip themselves when spaCy is absent)
"""

import importlib.util
import json
import os
import sys
import unittest
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import dissimilarity as D  # noqa: E402


def _load_script():
    path = os.path.join(_REPO, "scripts", "025_covariates.py")
    spec = importlib.util.spec_from_file_location("covariates_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


COV = _load_script()


class _Unavailable(D.SyntacticAnalyzer):
    """Stands in for a machine with no spaCy, without uninstalling anything."""

    def __init__(self):
        D.SyntacticAnalyzer.__init__(self, reason="stubbed out for the test")


class Levenshtein(unittest.TestCase):

    def test_hand_computed_character_distances(self):
        cases = [
            ("kitten", "sitting", 3),   # k->s, e->i, +g
            ("flaw", "lawn", 2),        # -f, +n
            ("saturday", "sunday", 3),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("abc", "", 3),
            ("", "", 0),
            ("a", "b", 1),
            ("200-foot dolphin", "100-foot dolphin", 1),  # the nl_break perturbation
        ]
        for a, b, want in cases:
            self.assertEqual(D.levenshtein(a, b), want, (a, b))
            self.assertEqual(D.levenshtein(b, a), want, "symmetry: %r %r" % (a, b))

    def test_normalization_divides_by_the_longer_string(self):
        self.assertAlmostEqual(D.normalized_levenshtein("kitten", "sitting"), 3 / 7.0)
        self.assertAlmostEqual(D.normalized_levenshtein("abc", ""), 1.0)
        self.assertAlmostEqual(D.normalized_levenshtein("", ""), 0.0)
        self.assertAlmostEqual(D.normalized_levenshtein("abc", "abc"), 0.0)

    def test_works_on_sequences_not_just_strings(self):
        # this is the code path the POS-sequence distance uses
        self.assertEqual(D.levenshtein(["NOUN", "VERB", "NUM"],
                                       ["NOUN", "NUM"]), 1)
        self.assertAlmostEqual(
            D.normalized_levenshtein(["NOUN", "VERB", "NUM"], ["NOUN", "NUM"]), 1 / 3.0)

    def test_triangle_inequality_on_a_few_triples(self):
        for a, b, c in [("kitten", "sitting", "mitten"), ("abc", "abd", "xyz")]:
            self.assertLessEqual(D.levenshtein(a, c),
                                 D.levenshtein(a, b) + D.levenshtein(b, c))


class Jaccard(unittest.TestCase):

    def test_toy_sets(self):
        self.assertAlmostEqual(D.jaccard_distance({"a", "b", "c"}, {"b", "c", "d"}), 0.5)
        self.assertAlmostEqual(D.jaccard_distance({"a"}, {"a"}), 0.0)
        self.assertAlmostEqual(D.jaccard_distance({"a"}, {"b"}), 1.0)
        self.assertAlmostEqual(D.jaccard_distance(set(), set()), 0.0)
        self.assertAlmostEqual(D.jaccard_distance({"a", "b"}, set()), 1.0)
        # types, not tokens: repeats must not change the answer
        self.assertAlmostEqual(D.jaccard_distance(["a", "a", "b"], ["a", "b"]), 0.0)

    def test_tokenizer_keeps_numerals_and_lowercases(self):
        self.assertEqual(D.tokenize("Amara saw a 200-foot dolphin."),
                         ["amara", "saw", "a", "200", "foot", "dolphin"])
        self.assertEqual(D.tokenize("x = 200\nk = 8"), ["x", "200", "k", "8"])

    def test_numerals_are_what_makes_nl_break_visible(self):
        # if the tokenizer dropped numerals these two would be Jaccard-identical and the
        # lexical control would have zero variance on exactly the H2 workhorse condition
        a = "Amara saw a 200-foot dolphin."
        b = "Amara saw a 100-foot dolphin."
        self.assertGreater(D.token_jaccard_distance(a, b), 0.0)
        self.assertAlmostEqual(D.token_jaccard_distance(a, b), 2 / 7.0)


class BagDistance(unittest.TestCase):

    def test_total_variation_on_toy_bags(self):
        f = D._bag_tv_distance
        self.assertAlmostEqual(f(Counter({"a": 1, "b": 1}), Counter({"a": 1})), 0.5)
        self.assertAlmostEqual(f(Counter({"a": 2}), Counter({"a": 5})), 0.0)  # normalized
        self.assertAlmostEqual(f(Counter({"a": 1}), Counter({"b": 1})), 1.0)
        self.assertAlmostEqual(f(Counter(), Counter()), 0.0)
        self.assertAlmostEqual(f(Counter({"a": 1}), Counter()), 1.0)

    def test_length_invariance(self):
        # a 30-token and a 300-token text with the same label proportions are distance 0;
        # otherwise `formal` (short) would score distant from `nl_base` (long) for a
        # reason that has nothing to do with syntax
        self.assertAlmostEqual(
            D._bag_tv_distance(Counter({"nsubj": 1, "dobj": 1}),
                               Counter({"nsubj": 50, "dobj": 50})), 0.0)


class NoProxySubstitution(unittest.TestCase):
    """The load-bearing honesty test: absent parser => null, never a lexical stand-in."""

    def test_unavailable_analyzer_reports_itself(self):
        a = _Unavailable()
        self.assertFalse(a.available)
        self.assertEqual(a.method, "unavailable")
        self.assertRaises(RuntimeError, a.parse, "anything")

    def test_syntactic_keys_are_explicitly_null_when_no_parser(self):
        m = D.pair_measures("Amara saw a 200-foot dolphin.",
                            "Amara saw a 100-foot dolphin.", _Unavailable())
        for key in ("syn_dep_bag", "syn_dep_triple", "syn_pos_seq"):
            self.assertIn(key, m)
            self.assertIsNone(m[key], key)
        self.assertGreater(m["lex_edit"], 0.0)

    def test_null_syntax_never_reaches_the_model_ready_file(self):
        m = D.pair_measures("a b c", "a b d", _Unavailable())
        self.assertIsNone(D.select(m, D.PRIMARY_SOURCE))
        self.assertIsNone(D.select(m, D.ROBUSTNESS_SOURCE))
        # ... and with only the lexical key requested it still projects
        picked = D.select(m, {"lex_dissim": D.PRIMARY_SOURCE["lex_dissim"]})
        self.assertEqual(sorted(picked), ["lex_dissim"])

    def test_primary_keys_draw_from_different_measure_families(self):
        self.assertEqual(D.PRIMARY_SOURCE["lex_dissim"], "lex_edit")
        self.assertTrue(D.PRIMARY_SOURCE["syn_dissim"].startswith("syn_"))
        self.assertTrue(D.ROBUSTNESS_SOURCE["syn_dissim"].startswith("syn_"))
        self.assertFalse(D.PRIMARY_SOURCE["syn_dissim"].startswith("lex_"))


_ANALYZER = D.load_syntactic_analyzer()


@unittest.skipUnless(_ANALYZER.available, "spaCy parser not installed")
class Syntactic(unittest.TestCase):

    def test_identical_texts_are_distance_zero(self):
        t = "Natalia sold 48 clips to her friends in April."
        p = _ANALYZER.parse(t)
        d = _ANALYZER.dissimilarity(p, p)
        self.assertEqual(set(d), {"syn_dep_bag", "syn_dep_triple", "syn_pos_seq"})
        for v in d.values():
            self.assertAlmostEqual(v, 0.0)

    def test_lexical_swap_with_identical_structure_is_syntactically_closer(self):
        # same parse shape, different words: syntactic distance must be far below the
        # lexical one. This is the contrast H2 is built on.
        a = "Natalia sold 48 clips to her friends in April."
        b = "Beatriz bought 92 crates from her neighbours in October."
        syn = _ANALYZER.dissimilarity(_ANALYZER.parse(a), _ANALYZER.parse(b))
        lex = D.normalized_levenshtein(a, b)
        self.assertLess(syn["syn_dep_bag"], lex)

    def test_bounds(self):
        pairs = [("The cat sat.", "x = 3"),
                 ("What percentage of the body length is that?", "answer = 12")]
        for a, b in pairs:
            for v in _ANALYZER.dissimilarity(_ANALYZER.parse(a),
                                             _ANALYZER.parse(b)).values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_anchor_label_is_aligned_by_character_offset(self):
        t = "Amara saw a 200-foot dolphin with 8 remoras."
        p = _ANALYZER.parse(t)
        self.assertEqual(t[12:15], "200")
        self.assertEqual(_ANALYZER.anchor_label(p, [12, 15]), "nummod")
        self.assertEqual(t[34:35], "8")
        self.assertEqual(_ANALYZER.anchor_label(p, [34, 35]), "nummod")
        self.assertEqual(_ANALYZER.anchor_label(p, [-1, -1]), "punct")  # FINAL sentinel
        # an offset inside a word resolves to that word's token, not to a re-tokenized
        # substring: "olphin" is not a token, but offset 21 lands inside "dolphin"
        self.assertEqual(p.token_at_char(21, 27)[2], "dolphin")


class OutputSchema(unittest.TestCase):
    """The emitted JSON must survive scripts/04_analyze.py's covariate handling."""

    CORPUS = [{
        "problem_id": "gsms_0000",
        "realizations": {
            "nl_base": {"text": "Amara saw a 200-foot dolphin with 8 remoras.",
                        "anchors": {"FINAL": [-1, -1], "x": [12, 15], "k": [34, 35]}},
            "nl_break": {"text": "Amara saw a 100-foot dolphin with 8 remoras.",
                         "anchors": {"FINAL": [-1, -1], "x": [12, 15], "k": [34, 35]}},
            "formal": {"text": "x = 200\nk = 8\nanswer = x * k",
                       "anchors": {"FINAL": [-1, -1], "x": [4, 7], "k": [12, 13]}},
            "nl_para": {"text": "", "status": "pending",
                        "anchors": {}},
        },
    }]

    @staticmethod
    def replay_04_analyze(cov):
        """Verbatim replay of scripts/04_analyze.py's `analyze_h2` covariate block."""
        cov = {k: v for k, v in cov.items() if k != D.SOURCE_KEY}  # main() pops it
        cov_names, seen = [], set()
        for per_cond in cov.values():
            for vals in per_cond.values():
                seen.update(vals)
        cov_names = sorted(seen)
        rows = []
        for pid, per_cond in cov.items():
            for cond in per_cond:
                rec = {}
                vals = (cov.get(pid, {}) or {}).get(cond, {})
                for name in cov_names:
                    if name not in vals:
                        raise AssertionError(
                            "covariates file is missing %r for (%s, %s)"
                            % (name, pid, cond))
                    rec[name] = float(vals[name])   # a null dies here
                rows.append(rec)
        return cov_names, rows

    def test_emitted_shape_is_problem_condition_covariate(self):
        measures, labels, skipped = COV.build(
            self.CORPUS, _ANALYZER, "nl_base", ["nl_para", "formal", "nl_break"])
        keys = list(D.PRIMARY_KEYS) if _ANALYZER.available else ["lex_dissim"]
        primary, _ = COV.project(measures, D.PRIMARY_SOURCE, keys)
        self.assertEqual(sorted(primary), ["gsms_0000"])
        self.assertEqual(sorted(primary["gsms_0000"]), ["formal", "nl_break"])
        for vals in primary["gsms_0000"].values():
            self.assertEqual(sorted(vals), sorted(keys))
            for v in vals.values():
                self.assertIsInstance(v, float)
        self.assertEqual([s["condition"] for s in skipped], ["nl_para"])

    def test_placeholder_realizations_are_skipped_not_zero_filled(self):
        measures, _, skipped = COV.build(
            self.CORPUS, _Unavailable(), "nl_base", ["nl_para", "formal", "nl_break"])
        self.assertNotIn("nl_para", measures["gsms_0000"])
        self.assertIn("empty placeholder", skipped[0]["reason"])

    def test_consumer_replay_accepts_the_emitted_file(self):
        measures, _, _ = COV.build(self.CORPUS, _ANALYZER, "nl_base",
                                   ["nl_para", "formal", "nl_break"])
        keys = list(D.PRIMARY_KEYS) if _ANALYZER.available else ["lex_dissim"]
        for family, source in ((D.SOURCE_PRIMARY, D.PRIMARY_SOURCE),
                               (D.SOURCE_ROBUSTNESS, D.ROBUSTNESS_SOURCE)):
            cov, _ = COV.project(measures, source, keys)
            cov[D.SOURCE_KEY] = D.source_tag(family, keys)   # as main() does
            self.assertEqual(COV.check_consumer_shape(cov, keys), [])
            names, rows = self.replay_04_analyze(cov)
            self.assertEqual(names, sorted(keys))
            self.assertEqual(len(rows), 2)

    def test_consumer_replay_rejects_a_null_or_ragged_file(self):
        # a null does not merely degrade the fit -- it is a TypeError inside
        # 04_analyze.py's `float(vals[name])`, i.e. the primary analysis dies mid-run
        self.assertRaises(TypeError, self.replay_04_analyze,
                          {"p": {"formal": {"lex_dissim": 0.3, "syn_dissim": None}}})
        self.assertRaises(AssertionError, self.replay_04_analyze,
                          {"p": {"formal": {"lex_dissim": 0.3, "syn_dissim": 0.1},
                                 "nl_break": {"lex_dissim": 0.01}}})
        self.assertTrue(COV.check_consumer_shape(
            {"p": {"formal": {"lex_dissim": 0.3, "syn_dissim": None}}},
            ["lex_dissim", "syn_dissim"]))

    def test_meta_never_leaks_into_the_model_ready_file(self):
        # covariates_measures.json carries `_meta`; 04_analyze.py would read that key as
        # a problem_id and its string values as covariate names. It must live only there.
        path = os.path.join(_REPO, "data/corpus/covariates.json")
        if not os.path.exists(path):
            self.skipTest("covariates.json not built yet")
        with open(path, "r", encoding="utf-8") as fh:
            cov = json.load(fh)
        self.assertNotIn("_meta", cov)
        names, rows = self.replay_04_analyze(cov)
        self.assertTrue(rows)
        self.assertTrue(set(names) <= set(D.PRIMARY_KEYS))
        # `_source` is the ONE reserved key, and it must name the primary family --
        # the robustness file is shape-identical, so this tag is the only thing that
        # stops it being fitted and reported as H2's pre-registered primary.
        self.assertEqual(cov[D.SOURCE_KEY]["family"], D.SOURCE_PRIMARY)
        self.assertEqual(cov[D.SOURCE_KEY]["measures"],
                         {k: D.PRIMARY_SOURCE[k] for k in names})
        for pid, per_cond in cov.items():
            if pid == D.SOURCE_KEY:
                continue
            self.assertTrue(pid.startswith("gsms_"), pid)
            self.assertNotIn(D.REFERENCE_CONDITION, per_cond,
                             "the reference condition has no covariate row of its own")


if __name__ == "__main__":
    unittest.main(verbosity=2)
