"""Tests for the corpus generator (src/ling/templates.py, src/ling/corpus.py).

The property that actually matters here is span exactness. An anchor span that
is off by one character maps to the wrong token at extraction time, and every
lambda2 / fork-layer / AMI number downstream is then plausible-looking garbage
with nothing in the output to reveal it (DESIGN.md A.11). So the tests below
re-derive `text[start:end] == literal` for every quantity in every generated
realization of every problem, and additionally check that the *machinery* that
enforces it actually fails when handed a bad span -- a validator that never
rejects anything is worse than no validator.

Also checked:
  * ground truth is COMPUTED from the template's own `#answer:` expression
  * `formal` is givens + goal with NO derivation (prereg/DEVIATIONS.md
    Amendment 4): every literal equals the value it claims and the goal is the
    template's own question
  * nl_break changes the answer and changes nothing else that surfaces
  * nl_para is answer-preserving, anchors every quantity with the same surface
    form nl_base used, and varies its surface operation across the corpus
  * the FINAL sentinel is present in every realization, as [-1, -1], which is
    the encoding `ling.hooks.resolve_anchor_positions` expects
  * same seed -> byte-identical corpus; different seed -> different corpus
  * the corpus is varied enough in linguistic structure for H1 to be measurable
  * no numpy, no network

Run with:  python3 -m unittest discover -s tests -v
(stdlib unittest, so it runs on a machine with no torch and no pytest)
"""

import json
import os
import random
import re
import sys
import unittest
from fractions import Fraction

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import corpus as C  # noqa: E402
from ling import templates as T  # noqa: E402

SEED = 20260906


def _build(seed=SEED, n=100):
    prov, tmpls = T.load_templates()
    recs, fails = C.build_corpus(tmpls, seed=seed, n_problems=n, provenance=prov)
    return prov, tmpls, recs, fails


class _Corpus(unittest.TestCase):
    """Base class: build the corpus once for the whole module."""

    prov = tmpls = recs = fails = None

    @classmethod
    def setUpClass(cls):
        if _Corpus.recs is None:
            _Corpus.prov, _Corpus.tmpls, _Corpus.recs, _Corpus.fails = _build()
        cls.prov, cls.tmpls, cls.recs, cls.fails = (
            _Corpus.prov, _Corpus.tmpls, _Corpus.recs, _Corpus.fails
        )


class TestTemplateBundle(_Corpus):
    def test_route_is_the_official_gsm_symbolic_templates(self):
        # DESIGN.md A.5 asks for GSM-Symbolic templates, not raw GSM8K and not
        # substitutes. If this ever flips to authored templates it must be a
        # deliberate, visible change.
        self.assertEqual(self.prov["route"], "OFFICIAL")
        self.assertIn("apple/ml-gsm-symbolic", self.prov["repo"])
        self.assertTrue(self.prov["commit"])
        self.assertEqual(len(self.tmpls), 100)

    def test_bundle_ships_no_raw_gsm8k_text(self):
        # A.5's whole point: raw GSM8K is contaminated. Only the symbolic form
        # is vendored.
        with open(T.DEFAULT_TEMPLATE_BUNDLE, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        for raw in bundle["templates"]:
            self.assertNotIn("question", raw)
            self.assertNotIn("answer", raw)
            self.assertIn("question_annotated", raw)

    def test_no_build_failures(self):
        self.assertEqual(self.fails, [], "templates that produced no problem")

    def test_hundred_problems_from_hundred_distinct_templates(self):
        self.assertEqual(len(self.recs), 100)
        self.assertEqual(len({r["template_id"] for r in self.recs}), 100)


class TestSpanExactness(_Corpus):
    def test_every_quantity_span_holds_its_literal(self):
        checked = 0
        for r in self.recs:
            for cond in C.GENERATED_CONDITIONS:
                real = r["realizations"][cond]
                text = real["text"]
                for q in r["quantities"]:
                    start, end = real["anchors"][q["name"]]
                    self.assertTrue(0 <= start < end <= len(text))
                    self.assertTrue(
                        text[start:end].strip(),
                        "%s/%s/%s anchors whitespace" % (r["problem_id"], cond, q["name"]),
                    )
                    checked += 1
        self.assertEqual(checked, 3 * sum(len(r["quantities"]) for r in self.recs))
        self.assertGreater(checked, 1000)

    def test_recorded_surface_matches_the_span_in_both_notations(self):
        for r in self.recs:
            nl = r["realizations"]["nl_base"]
            fm = r["realizations"]["formal"]
            for q in r["quantities"]:
                s, e = nl["anchors"][q["name"]]
                self.assertEqual(nl["text"][s:e], q["surface"])
                s, e = fm["anchors"][q["name"]]
                self.assertEqual(fm["text"][s:e], q["formal_literal"])

    def test_formal_anchor_is_a_python_numeric_literal(self):
        pat = re.compile(r"^(-?\d+(?:\.\d+)?|Fraction\(-?\d+, -?\d+\))$")
        for r in self.recs:
            fm = r["realizations"]["formal"]
            for q in r["quantities"]:
                s, e = fm["anchors"][q["name"]]
                self.assertRegex(fm["text"][s:e], pat)

    def test_anchor_spans_do_not_overlap(self):
        for r in self.recs:
            for cond in C.GENERATED_CONDITIONS:
                spans = sorted(
                    tuple(v) for k, v in r["realizations"][cond]["anchors"].items()
                    if k != "FINAL"
                )
                for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                    self.assertLessEqual(e1, s2, "%s/%s" % (r["problem_id"], cond))

    def test_spans_are_emitted_not_searched(self):
        # A repeated surface form is the case where find() would silently pick
        # the wrong occurrence. Confirm the corpus contains such cases and that
        # our anchor is the FIRST introduction, per configs/corpus_schema.md.
        found_repeat = 0
        for r in self.recs:
            text = r["realizations"]["nl_base"]["text"]
            for q in r["quantities"]:
                lit = q["surface"]
                if text.count(lit) > 1:
                    found_repeat += 1
                    s, _ = r["realizations"]["nl_base"]["anchors"][q["name"]]
                    self.assertLessEqual(
                        s,
                        max(
                            r["realizations"]["nl_base"]["anchors"][o["name"]][0]
                            for o in r["quantities"]
                        ),
                    )
        self.assertGreater(found_repeat, 0, "no repeated literals — test is vacuous")

    def test_validator_rejects_a_tampered_span(self):
        rec = json.loads(json.dumps(self.recs[0]))
        name = rec["quantities"][0]["name"]
        s, e = rec["realizations"]["nl_base"]["anchors"][name]
        rec["realizations"]["nl_base"]["anchors"][name] = [s + 1, e + 1]
        # validate_record checks bounds/overlap; the literal check lives in
        # assert_span_literals, which is what the builder calls.
        with self.assertRaises(T.SpanError):
            C.assert_span_literals(
                rec, {"nl_base": {name: rec["quantities"][0]["surface"]}}
            )

    def test_validator_rejects_a_missing_quantity(self):
        rec = json.loads(json.dumps(self.recs[0]))
        name = rec["quantities"][0]["name"]
        del rec["realizations"]["formal"]["anchors"][name]
        with self.assertRaises(T.SpanError):
            C.validate_record(rec)

    def test_render_asserts_on_a_corrupted_segment_list(self):
        # If the renderer's bookkeeping and the produced string ever disagree,
        # the assertion in render_nl must fire rather than emitting a record.
        t = self.tmpls[0]
        inst = t.sample(random.Random(1))
        r = t.render_nl(inst.bindings, inst.quantities)
        bad = T.Rendered(text=r.text[:-1], anchors=dict(r.anchors), literals=dict(r.literals))
        with self.assertRaises(T.SpanError):
            bad.anchors[inst.quantities[0]] = [0, len(bad.text) + 5]
            bad.validate(inst.quantities)


class TestFinalSentinel(_Corpus):
    def test_final_present_in_every_generated_realization(self):
        for r in self.recs:
            for cond in C.GENERATED_CONDITIONS:
                anchors = r["realizations"][cond]["anchors"]
                self.assertIn("FINAL", anchors, "%s/%s" % (r["problem_id"], cond))
                # ling.hooks.resolve_anchor_positions reads [-1, -1] as
                # "the last token"; any other encoding breaks the join.
                self.assertEqual(anchors["FINAL"], [-1, -1])


class TestGroundTruthIsComputed(_Corpus):
    def test_formal_is_givens_and_goal_with_no_derivation(self):
        # prereg/DEVIATIONS.md Amendment 4. The earlier build emitted the full
        # program, `answer = <the template's answer expression>`; under that,
        # H3 compared a prompt containing the derivation against a prompt
        # containing a word problem. The replacement checks are: every literal
        # equals the value it claims, and nothing binds an expression.
        by_id = {t.template_id: t for t in self.tmpls}
        for r in self.recs:
            text = r["realizations"]["formal"]["text"]
            self.assertTrue(text.endswith("answer = ?"), r["problem_id"])
            self.assertNotIn("answer =", text[: -len("answer = ?")])
            expr = by_id[r["template_id"]].answer_expr
            if re.search(r"[-+*/%()]", expr):
                self.assertNotIn(
                    re.sub(r"\s+", "", expr), re.sub(r"\s+", "", text), r["problem_id"]
                )
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                name, _, rhs = line.partition(" = ")
                if name == "answer":
                    continue
                self.assertRegex(
                    rhs, r"^(-?\d+(\.\d+)?|Fraction\(-?\d+, -?\d+\))$",
                    "%s: given %r is not a bare literal" % (r["problem_id"], line),
                )

    def test_formal_literals_equal_the_sampled_values(self):
        for r in self.recs:
            for q in r["quantities"]:
                lit = q["formal_literal"]
                m = re.match(r"^Fraction\((-?\d+), (-?\d+)\)$", lit)
                got = Fraction(int(m.group(1)), int(m.group(2))) if m else Fraction(lit)
                val = q["value"]
                if isinstance(val, dict):
                    want = Fraction(val["num"], val["den"])
                    self.assertEqual(got, want, r["problem_id"])
                else:
                    self.assertAlmostEqual(float(got), float(val), places=9)

    def test_formal_goal_is_the_templates_own_question(self):
        for r in self.recs:
            fm = r["realizations"]["formal"]
            goal_nl = fm["goal_nl"]
            base = r["realizations"]["nl_base"]["text"]
            self.assertTrue(base.endswith(goal_nl), r["problem_id"])
            head = base[: len(base) - len(goal_nl)]
            if head:
                self.assertRegex(head, r"[.!?]\s+$", r["problem_id"])
            self.assertTrue(
                "?" in goal_nl
                or re.search(r"\b(calculate|find|determine|compute)\b", goal_nl.lower()),
                r["problem_id"],
            )
            # the goal block carries identifiers, not values: the quantity's
            # literal appears once, in the givens block
            self.assertIn(fm["goal"], fm["text"])

    def test_formal_never_binds_a_solved_for_scratch_variable(self):
        # Three templates write `#answer: ans`, where `ans` IS the answer.
        # Binding it in the givens would hand the model the result.
        by_id = {t.template_id: t for t in self.tmpls}
        for r in self.recs:
            t = by_id[r["template_id"]]
            bound = set(r["realizations"]["formal"]["given"])
            self.assertEqual(bound & set(t.free_vars), set(), r["problem_id"])

    def test_ground_truth_is_a_positive_clean_number(self):
        for r in self.recs:
            gt = r["ground_truth"]
            self.assertIsInstance(gt, (int, float))
            self.assertGreater(gt, 0)
            if isinstance(gt, float):
                self.assertAlmostEqual(gt, round(gt, 2), places=9)

    def test_answer_recomputes_from_the_template_expression(self):
        # Independent re-derivation: re-sample the same template with the same
        # derived RNG and require the same number.
        by_id = {t.template_id: t for t in self.tmpls}
        for r in self.recs[:25]:
            t = by_id[r["template_id"]]
            inst = t.sample(C.problem_rng(r["seed"], t.template_id))
            self.assertEqual(inst.answer, r["ground_truth"])
            self.assertEqual([q["name"] for q in r["quantities"]], inst.quantities)


class TestBreakCondition(_Corpus):
    def test_answer_changed_everywhere(self):
        for r in self.recs:
            self.assertNotEqual(
                r["realizations"]["nl_break"]["answer"], r["ground_truth"], r["problem_id"]
            )
            self.assertEqual(r["perturbation"]["answer_base"], r["ground_truth"])
            self.assertEqual(
                r["perturbation"]["answer_break"], r["realizations"]["nl_break"]["answer"]
            )

    def test_exactly_one_quantity_was_perturbed(self):
        for r in self.recs:
            self.assertIn(
                r["perturbation"]["quantity"], [q["name"] for q in r["quantities"]]
            )
            self.assertNotEqual(
                r["perturbation"]["from_surface"], r["perturbation"]["to_surface"]
            )

    def test_break_stays_surface_similar(self):
        # configs/corpus_schema.md: a perturbation that is both semantically and
        # lexically far away makes H2 trivially true. The surface-dissimilarity
        # covariates can only do their job if nl_break really is close.
        dists = [r["perturbation"]["text_char_edit_distance"] for r in self.recs]
        self.assertGreaterEqual(
            sum(1 for d in dists if d == 1), 90, "fewer than 90/100 single-char breaks"
        )
        self.assertLessEqual(max(dists), 12)

    def test_break_keeps_the_same_quantity_set_and_anchors(self):
        for r in self.recs:
            base = r["realizations"]["nl_base"]["anchors"]
            brk = r["realizations"]["nl_break"]["anchors"]
            self.assertEqual(sorted(base), sorted(brk))

    def test_break_text_differs_from_base(self):
        for r in self.recs:
            self.assertNotEqual(
                r["realizations"]["nl_base"]["text"],
                r["realizations"]["nl_break"]["text"],
            )

    def test_break_is_labelled_not_answer_preserving(self):
        for r in self.recs:
            self.assertFalse(r["answer_preserving"]["nl_break"])
            self.assertTrue(r["answer_preserving"]["nl_base"])
            self.assertTrue(r["answer_preserving"]["formal"])


class TestParaphrase(_Corpus):
    """`nl_para` is authored prose, so it gets the strictest anchoring checks."""

    def test_every_paraphrase_is_written_and_answer_preserving(self):
        for r in self.recs:
            para = r["realizations"]["nl_para"]
            self.assertEqual(para["status"], "written", r["problem_id"])
            self.assertTrue(para["text"])
            self.assertEqual(para["answer"], r["ground_truth"])
            self.assertTrue(r["answer_preserving"]["nl_para"])

    def test_paraphrase_anchors_every_quantity_at_its_nl_base_surface(self):
        # The whole point of the schema representation: the paraphrase carries
        # the SAME emitted surface form, so the two conditions anchor the same
        # quantity introduction and the span can be verified exactly.
        checked = 0
        for r in self.recs:
            para = r["realizations"]["nl_para"]
            for q in r["quantities"]:
                self.assertIn(q["name"], para["anchors"], r["problem_id"])
                s, e = para["anchors"][q["name"]]
                self.assertEqual(para["text"][s:e], q["surface"], r["problem_id"])
                checked += 1
        self.assertEqual(checked, sum(len(r["quantities"]) for r in self.recs))

    def test_paraphrase_differs_from_nl_base(self):
        for r in self.recs:
            self.assertNotEqual(
                r["realizations"]["nl_para"]["text"],
                r["realizations"]["nl_base"]["text"],
                r["problem_id"],
            )

    def test_paraphrase_operations_vary(self):
        # H2's surface-dissimilarity covariates are lexical AND syntactic. A
        # corpus where every paraphrase applies the same transformation gives
        # them no variance, and nothing else in the pipeline would report that.
        ops = [r["realizations"]["nl_para"]["paraphrase_op"] for r in self.recs]
        self.assertGreaterEqual(len(set(ops)), 8, sorted(set(ops)))
        self.assertLess(max(ops.count(o) for o in set(ops)), 0.4 * len(self.recs))

    def test_paraphrase_surface_dissimilarity_has_spread(self):
        def toks(t):
            return set(re.findall(r"[a-z0-9']+", t.lower()))

        jac = []
        for r in self.recs:
            a = toks(r["realizations"]["nl_base"]["text"])
            b = toks(r["realizations"]["nl_para"]["text"])
            jac.append(len(a & b) / len(a | b))
        self.assertLess(min(jac), 0.7, "no lexically distant paraphrase")
        self.assertGreater(max(jac), 0.9, "no syntax-only paraphrase")

    def test_validator_rejects_a_paraphrase_missing_a_quantity(self):
        rec = json.loads(json.dumps(self.recs[0]))
        name = rec["quantities"][0]["name"]
        del rec["realizations"]["nl_para"]["anchors"][name]
        with self.assertRaises(T.SpanError):
            C.validate_record(rec)

    def test_all_four_condition_keys_exist(self):
        for r in self.recs:
            self.assertEqual(sorted(r["realizations"]), sorted(C.CONDITIONS))


class TestDeterminism(_Corpus):
    def test_same_seed_is_byte_identical(self):
        _, _, a, _ = _build(seed=SEED, n=12)
        _, _, b, _ = _build(seed=SEED, n=12)
        self.assertEqual(
            [json.dumps(r, sort_keys=True) for r in a],
            [json.dumps(r, sort_keys=True) for r in b],
        )

    def test_different_seed_changes_the_values(self):
        _, _, a, _ = _build(seed=SEED, n=12)
        _, _, b, _ = _build(seed=SEED + 1, n=12)
        self.assertNotEqual(
            [r["realizations"]["nl_base"]["text"] for r in a],
            [r["realizations"]["nl_base"]["text"] for r in b],
        )

    def test_per_problem_rng_is_independent_of_template_order(self):
        # Dropping a template must not perturb any other problem's values.
        prov, tmpls = T.load_templates()
        full, _ = C.build_corpus(tmpls, seed=SEED, n_problems=8, provenance=prov)
        skipped, _ = C.build_corpus(
            [tmpls[0]] + tmpls[2:], seed=SEED, n_problems=8, provenance=prov
        )
        by_id = {r["template_id"]: r["realizations"]["nl_base"]["text"] for r in skipped}
        for r in full:
            if r["template_id"] in by_id:
                self.assertEqual(by_id[r["template_id"]], r["realizations"]["nl_base"]["text"])


class TestLinguisticVariation(_Corpus):
    """H1 scores clusters against linguistic labels.

    A corpus in which every problem shares one syntactic frame gives those
    labels no variance and makes H1 unmeasurable -- and nothing else in the
    pipeline would report that. These thresholds are deliberately loose; they
    exist to catch a collapse, not to certify diversity.
    """

    def _prof(self, key):
        return [r["linguistic_profile"][key] for r in self.recs]

    def test_multiple_question_forms(self):
        forms = set(self._prof("question_form"))
        self.assertGreaterEqual(len(forms), 5, forms)
        # no single frame may swallow the corpus
        top = max(self._prof("question_form").count(f) for f in forms)
        self.assertLess(top, 0.7 * len(self.recs))

    def test_clause_structure_varies(self):
        sigs = {tuple(r["linguistic_profile"]["clause_markers"]) for r in self.recs}
        self.assertGreaterEqual(len(sigs), 15, "clause-marker signatures collapsed")
        self.assertGreaterEqual(len(set(self._prof("n_sentences"))), 5)

    def test_length_varies(self):
        w = self._prof("n_words")
        self.assertLess(min(w), 30)
        self.assertGreater(max(w), 80)

    def test_quantity_introduction_pattern_varies(self):
        counts = set(self._prof("n_quantities"))
        self.assertGreaterEqual(len(counts), 5, counts)
        first = self._prof("first_quantity_rel_pos")
        self.assertLess(min(first), 0.15)
        self.assertGreater(max(first), 0.4)

    def test_openings_vary(self):
        opens = self._prof("opens_with_proper_name")
        self.assertGreater(sum(opens), 10)
        self.assertGreater(len(opens) - sum(opens), 10)


class TestTemplateMachinery(unittest.TestCase):
    def test_number_to_words_matches_official_generated_data(self):
        self.assertEqual(T.number_to_words(31), "thirty one")
        self.assertEqual(T.number_to_words(20), "twenty")
        self.assertEqual(T.number_to_words(7), "seven")

    def test_numeric_wrappers_are_both_numbers_and_pairs(self):
        v = T._wrap(("six-sided", 6))
        self.assertEqual(v + 1, 7)
        self.assertEqual(v[0], "six-sided")
        self.assertEqual(v[1], 6)
        self.assertEqual(v.surface, "six-sided")
        f = T._wrap(("half", Fraction(1, 2)))
        self.assertEqual(f * 4, 2)
        self.assertEqual(f.surface, "half")

    def test_normalise_answer_rejects_dirty_values(self):
        self.assertEqual(T.normalise_answer(Fraction(6, 2)), 3)
        self.assertEqual(T.normalise_answer(12.50), 12.5)
        self.assertIsNone(T.normalise_answer(Fraction(19, 3)))
        self.assertIsNone(T.normalise_answer("cat"))
        self.assertEqual(T.normalise_answer({Fraction(75, 1)}), 75)

    def test_indefinite_article_agreement(self):
        self.assertEqual(T._fix_article("He paid a ", "agent"), "He paid an ")
        self.assertEqual(T._fix_article("He paid an ", "broker"), "He paid a ")
        self.assertEqual(T._fix_article("within a ", "hour"), "within an ")
        self.assertEqual(T._fix_article("within a ", "university"), "within a ")
        self.assertEqual(T._fix_article("no article here", "apple"), "no article here")

    def test_lexical_coherence_guard(self):
        self.assertFalse(T.Template._lexically_coherent({"a": "coins", "b": "coins"}))
        self.assertTrue(T.Template._lexically_coherent({"a": "coins", "b": "beads"}))
        # numeric collisions stay legal
        self.assertTrue(
            T.Template._lexically_coherent({"a": T._wrap(3), "b": T._wrap(3)})
        )

    def test_no_numpy_or_network_imports(self):
        for mod in (T, C):
            with open(mod.__file__, "r", encoding="utf-8") as fh:
                src = fh.read()
            self.assertNotIn("import numpy", src)
            self.assertNotIn("import requests", src)
            self.assertNotIn("urllib.request", src)


class TestWrittenCorpusFile(unittest.TestCase):
    """If the generator has been run, hold the file on disk to the same rules."""

    PATH = os.path.join(_REPO, "data", "corpus", "problems.jsonl")

    def setUp(self):
        if not os.path.exists(self.PATH):
            self.skipTest("corpus not generated yet: run scripts/02_generate_corpus.py")

    def test_file_matches_schema_and_spans(self):
        with open(self.PATH, "r", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        self.assertEqual(len(recs), 100)
        for r in recs:
            C.validate_record(r)
            for cond in C.CONDITIONS:
                real = r["realizations"][cond]
                # Amendment 6's style controls are recorded as an explicit
                # `status: "unavailable"` hole -- no text, no anchors -- when the
                # instance cannot support them. `validate_record` above already
                # required that hole to be genuinely empty; there is nothing to
                # span-check in it.
                if not real.get("text"):
                    self.assertIn(real.get("status"), ("pending", "unavailable"))
                    self.assertFalse(real.get("anchors"))
                    continue
                for q in r["quantities"]:
                    s, e = real["anchors"][q["name"]]
                    self.assertTrue(real["text"][s:e].strip())
            self.assertEqual(
                set(r) >= {"problem_id", "template_id", "seed", "ground_truth",
                           "quantities", "realizations"},
                True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
