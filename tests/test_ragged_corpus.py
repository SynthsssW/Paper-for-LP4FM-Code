"""A declared hole must skip ONE condition, never drop the whole problem.

Regression guard for the highest-severity bug found before the GPU run: with
`nl_minimal_lex` absent for 30 problems, the all-or-none extraction path dropped
every one of those problems across ALL SIX conditions. The corpus would have
silently collapsed from 100 problems to ~70 for everything, including H3's
primary endpoint, and assignments.jsonl would have looked perfectly well-formed.
"""
import json, pathlib, unittest, importlib.util

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "problems.jsonl"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "extract_driver", ROOT / "scripts" / "03_extract_and_cluster.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDeclaredHolesSkipOneConditionOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CORPUS.exists():
            raise unittest.SkipTest("corpus not built")
        cls.m = _driver()
        cls.records = [json.loads(l) for l in CORPUS.open()]

    def test_no_problem_is_lost_to_a_declared_hole(self):
        """The bug: one unavailable condition dropped all six."""
        for rec in self.records:
            available, _absent, _holes = self.m.partition_conditions(
                rec, self.m.CONDITIONS)
            self.assertIn("nl_base", available,
                          f"{rec['problem_id']}: reference condition lost")
            self.assertIn("formal", available,
                          f"{rec['problem_id']}: H3's primary condition lost")
            self.assertIn("nl_break", available,
                          f"{rec['problem_id']}: H2's primary condition lost")

    def test_holes_match_the_corpus_and_are_not_absences(self):
        holes_by_cond, absent_total, available_total = {}, 0, 0
        for rec in self.records:
            available, absent, holes = self.m.partition_conditions(
                rec, self.m.CONDITIONS)
            available_total += len(available)
            absent_total += len(absent)
            for c in holes:
                holes_by_cond[c] = holes_by_cond.get(c, 0) + 1

        self.assertEqual(len(self.records), 100)
        self.assertEqual(absent_total, 0,
                         "a hole must be DECLARED, never a missing key")
        self.assertEqual(holes_by_cond.get("nl_minimal_lex"), 30,
                         "30 problems contain no proper name to swap")
        self.assertEqual(holes_by_cond.get("nl_minimal_orth"), 1,
                         "1 problem has no digit-rendered integer quantity")
        self.assertEqual(available_total, 569, "100*6 - 31 declared holes")

    def test_a_hole_is_not_counted_as_a_G3_drop(self):
        """Holes are expected and per-condition; drops are errors and whole-problem.
        Summing them would misreport the drop rate the paper commits to."""
        self.assertTrue(hasattr(self.m, "is_declared_hole"))
        rec = {"realizations": {"x": {"text": "", "anchors": {},
                                      "status": "unavailable"}}}
        self.assertTrue(self.m.is_declared_hole(rec["realizations"]["x"]))
        self.assertFalse(self.m.is_declared_hole(
            {"text": "real prompt", "anchors": {"q": [0, 1]}}))


if __name__ == "__main__":
    unittest.main()
