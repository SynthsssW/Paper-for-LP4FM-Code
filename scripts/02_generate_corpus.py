#!/usr/bin/env python3
"""Generate the corpus. DESIGN.md §A.5 / §A.6; configs/corpus_schema.md.

    "Build NL-base from GSM-Symbolic templates (Mirzadeh et al., ICLR 2025)
     with freshly sampled instantiations."   -- DESIGN.md §A.5

Route taken: **OFFICIAL templates.** The 100 templates are Apple's own
GSM-Symbolic `templates/symbolic/*.json`, vendored into
`data/templates/gsm_symbolic_templates.json` (see the header of
`src/ling/templates.py` for exactly which parts are Apple's and which parts are
our reconstruction). Nothing is fetched at run time.

Emits all four realizations per problem:

    nl_base    NL rendering; values freshly sampled; ground truth COMPUTED by
               executing the template's own `#answer:` expression
    nl_para    answer-preserving paraphrase, rendered from the authored schemas
               in `data/corpus/paraphrases.json` against the SAME bindings as
               nl_base, so every quantity keeps its nl_base surface form and its
               span is emitted rather than searched for
    formal     GIVENS + GOAL rewrite of the SAME instance (DESIGN.md §B.8 and
               prereg/DEVIATIONS.md Amendment 4: input notation, NO derivation
               and no answer expression, never executed)
    nl_break   exactly ONE quantity perturbed so the computed answer changes,
               kept as surface-similar to nl_base as the template allows

Determinism: the same `--seed` reproduces byte-identical output. Each problem
draws from its own RNG derived by SHA-256 from (seed, template_id), so adding,
dropping or reordering templates cannot perturb any other problem's values.
`--verify-reproducible` rebuilds the whole corpus a second time and compares the
serialised bytes.

Usage:
    python scripts/02_generate_corpus.py
    python scripts/02_generate_corpus.py --seed 20260906 --verify-reproducible
    python scripts/02_generate_corpus.py --out data/corpus/problems.jsonl -n 100
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling.corpus import (  # noqa: E402
    CONDITIONS,
    GENERATED_CONDITIONS,
    build_corpus,
    load_paraphrases,
    write_jsonl,
)
from ling.templates import load_templates  # noqa: E402

DEFAULT_OUT = os.path.join(_REPO, "data", "corpus", "problems.jsonl")
DEFAULT_SUMMARY = os.path.join(_REPO, "data", "corpus", "corpus_summary.json")
MODEL_YAML = os.path.join(_REPO, "configs", "model.yaml")
FALLBACK_SEED = 20260906


def seed_from_model_yaml(path: str = MODEL_YAML) -> "tuple[int, str]":
    """Read the master seed out of configs/model.yaml without a YAML dependency.

    One master seed for the whole run (configs/model.yaml `seed:`), so the corpus
    and the GPU side cannot silently drift apart.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^seed:\s*(\d+)\s*(?:#.*)?$", line)
                if m:
                    return int(m.group(1)), "configs/model.yaml"
    except OSError:
        pass
    return FALLBACK_SEED, "built-in fallback"


def serialise(records) -> bytes:
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records
    ).encode("utf-8")


def _tokens(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _char_edit(a, b):
    """Levenshtein on characters (the corpus is small; this is fast enough)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def paraphrase_stats(records) -> dict:
    """How far each nl_para moved from its nl_base, and by which operation.

    H2's control is a surface-dissimilarity covariate, lexical AND syntactic.
    If every paraphrase applied the same transformation those covariates would
    have no variance to work with, so the spread reported here is a property the
    corpus has to demonstrate, not an incidental statistic.
    """
    if not records:
        return {"n": 0}
    jac, edit, dsent, dword = [], [], [], []
    for r in records:
        base = r["realizations"]["nl_base"]["text"]
        para = r["realizations"]["nl_para"]["text"]
        tb, tp = _tokens(base), _tokens(para)
        jac.append(len(tb & tp) / max(len(tb | tp), 1))
        edit.append(_char_edit(base, para) / max(len(base), 1))
        dsent.append(
            len(re.findall(r"[.!?](?:\s|$)", para))
            - len(re.findall(r"[.!?](?:\s|$)", base))
        )
        dword.append(len(para.split()) - len(base.split()))
    return {
        "n": len(records),
        "operations": dict(
            sorted(
                collections.Counter(
                    r["realizations"]["nl_para"]["paraphrase_op"] for r in records
                ).items()
            )
        ),
        "n_distinct_operations": len(
            {r["realizations"]["nl_para"]["paraphrase_op"] for r in records}
        ),
        "lexical_overlap_jaccard": {
            "min": round(min(jac), 3),
            "median": round(statistics.median(jac), 3),
            "max": round(max(jac), 3),
        },
        "char_edit_distance_normalised": {
            "min": round(min(edit), 3),
            "median": round(statistics.median(edit), 3),
            "max": round(max(edit), 3),
        },
        "sentence_count_delta": dict(sorted(collections.Counter(dsent).items())),
        "word_count_delta": {
            "min": min(dword),
            "median": statistics.median(dword),
            "max": max(dword),
        },
        "identical_to_nl_base": sum(
            1
            for r in records
            if r["realizations"]["nl_para"]["text"] == r["realizations"]["nl_base"]["text"]
        ),
    }


def summarise(records) -> dict:
    prof = [r["linguistic_profile"] for r in records]
    pert = [r["perturbation"] for r in records]

    def counter(key, source=prof):
        return dict(sorted(collections.Counter(x[key] for x in source).items(), key=str))

    words = [p["n_words"] for p in prof]
    quantities = [len(r["quantities"]) for r in records]
    marker_sigs = {tuple(p["clause_markers"]) for p in prof}
    marker_counts = collections.Counter()
    for p in prof:
        marker_counts.update(p["clause_markers"])

    lens = {
        cond: [len(r["realizations"][cond]["text"]) for r in records]
        for cond in CONDITIONS
        if all(r["realizations"][cond].get("text") for r in records)
    }
    written = [
        r for r in records if r["realizations"]["nl_para"].get("status") == "written"
    ]

    return {
        "n_problems": len(records),
        "n_realizations_generated": len(records) * len(GENERATED_CONDITIONS),
        "n_realizations_paraphrased": len(written),
        "n_realizations_pending": len(records) - len(written),
        "conditions": list(CONDITIONS),
        "paraphrase": paraphrase_stats(written),
        "n_distinct_templates": len({r["template_id"] for r in records}),
        "quantities": {
            "total_anchors": sum(quantities),
            "per_problem": dict(sorted(collections.Counter(quantities).items())),
            "min": min(quantities),
            "max": max(quantities),
        },
        "answers": {
            "min": min(r["ground_truth"] for r in records),
            "max": max(r["ground_truth"] for r in records),
            "n_integer": sum(1 for r in records if isinstance(r["ground_truth"], int)),
        },
        "perturbation": {
            "template_conditions_still_satisfied": sum(
                1 for p in pert if p["template_conditions_still_satisfied"]
            ),
            "single_char_edit": sum(1 for p in pert if p["text_char_edit_distance"] == 1),
            "text_char_edit_distance": dict(
                sorted(collections.Counter(p["text_char_edit_distance"] for p in pert).items())
            ),
            "occurrences_rewritten": dict(
                sorted(collections.Counter(p["occurrences_rewritten"] for p in pert).items())
            ),
            "median_text_char_edit_distance": statistics.median(
                p["text_char_edit_distance"] for p in pert
            ),
        },
        "linguistic_variation": {
            "question_form": counter("question_form"),
            "n_sentences": counter("n_sentences"),
            "n_clause_markers": counter("n_clause_markers"),
            "question_is_final_sentence": counter("question_is_final_sentence"),
            "opens_with_proper_name": counter("opens_with_proper_name"),
            "distinct_clause_marker_signatures": len(marker_sigs),
            "clause_marker_frequency": dict(marker_counts.most_common()),
            "n_words": {
                "min": min(words),
                "median": statistics.median(words),
                "max": max(words),
            },
            "first_quantity_rel_pos_quartiles": [
                round(v, 4)
                for v in statistics.quantiles(
                    sorted(p["first_quantity_rel_pos"] for p in prof), n=4
                )
            ],
            "has_enumeration": sum(1 for p in prof if p["has_enumeration"]),
            "has_currency": sum(1 for p in prof if p["has_currency"]),
            "has_word_numeral": sum(1 for p in prof if p["has_word_numeral"]),
        },
        "text_length_chars": {
            cond: {
                "min": min(v),
                "median": statistics.median(v),
                "max": max(v),
                "mean": round(statistics.mean(v), 1),
            }
            for cond, v in lens.items()
        },
    }


def main(argv=None) -> int:
    default_seed, seed_src = seed_from_model_yaml()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=default_seed,
                    help="master seed (default %d, from %s)" % (default_seed, seed_src))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("-n", "--n-problems", type=int, default=100)
    ap.add_argument("--templates", default=None, help="template bundle path")
    ap.add_argument("--paraphrases", default=None,
                    help="nl_para schema bundle path "
                         "(default data/corpus/paraphrases.json)")
    ap.add_argument("--verify-reproducible", action="store_true",
                    help="rebuild once more and require byte-identical output")
    args = ap.parse_args(argv)

    prov, templates = load_templates(args.templates) if args.templates else load_templates()
    para_prov, paraphrases = (
        load_paraphrases(args.paraphrases) if args.paraphrases else load_paraphrases()
    )

    print("corpus generation — DESIGN.md §A.5")
    print("  template route     : %s" % prov.get("route"))
    print("  template source    : %s @ %s" % (prov.get("repo"), (prov.get("commit") or "")[:12]))
    print("  templates loaded   : %d" % len(templates))
    print("  paraphrases loaded : %d  (%s)" % (len(paraphrases), para_prov.get("route", "none")))
    print("  master seed        : %d  (%s)" % (args.seed, seed_src))
    print()

    records, failures = build_corpus(
        templates,
        seed=args.seed,
        n_problems=args.n_problems,
        provenance=prov,
        paraphrases=paraphrases,
    )

    if failures:
        print("BUILD FAILURES (%d) — these templates produced no valid problem:" % len(failures))
        for tid, why in failures:
            print("   %-14s %s" % (tid, why))
        print()

    payload = serialise(records)

    if args.verify_reproducible:
        again, _ = build_corpus(
            templates,
            seed=args.seed,
            n_problems=args.n_problems,
            provenance=prov,
            paraphrases=paraphrases,
        )
        if serialise(again) != payload:
            print("FATAL: rebuild with the same seed is not byte-identical.")
            return 1
        print("reproducibility  : OK — rebuild at seed %d is byte-identical" % args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(records, args.out)
    digest = hashlib.sha256(payload).hexdigest()

    summary = summarise(records)
    summary["seed"] = args.seed
    summary["sha256"] = digest
    summary["template_route"] = prov.get("route")
    summary["template_commit"] = prov.get("commit")
    summary["build_failures"] = [{"template_id": t, "reason": w} for t, w in failures]
    summary["paraphrase_route"] = para_prov.get("route")
    with open(args.summary, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    n = summary["n_problems"]
    print("wrote %s" % os.path.relpath(args.out, _REPO))
    print("      %s" % os.path.relpath(args.summary, _REPO))
    print("  sha256           : %s" % digest)
    print()
    print("problems           : %d  (%d distinct templates)"
          % (n, summary["n_distinct_templates"]))
    print("realizations       : %d generated (nl_base, formal, nl_break)"
          % summary["n_realizations_generated"])
    print("                     %d paraphrased (nl_para, authored schemas)"
          % summary["n_realizations_paraphrased"])
    print("                     %d pending" % summary["n_realizations_pending"])
    print("build drop rate    : %d/%d templates (%.1f%%)"
          % (len(failures), len(templates), 100.0 * len(failures) / max(len(templates), 1)))
    print()
    print("anchored quantities: %d total, %d–%d per problem"
          % (summary["quantities"]["total_anchors"],
             summary["quantities"]["min"], summary["quantities"]["max"]))
    print("                     every quantity anchored in all %d realizations"
          % len(summary["conditions"]))
    print("                     FINAL sentinel present in all %d"
          % len(summary["conditions"]))
    print()
    p = summary["perturbation"]
    print("nl_break           : %d/%d differ from nl_base by ONE character"
          % (p["single_char_edit"], n))
    print("                     %d/%d keep the template's #conditions satisfied"
          % (p["template_conditions_still_satisfied"], n))
    print("                     answer changed in %d/%d (asserted)" % (n, n))
    print()
    pp = summary["paraphrase"]
    if pp["n"]:
        print("nl_para            : %d/%d written, %d distinct surface operations"
              % (pp["n"], n, pp["n_distinct_operations"]))
        print("                     lexical overlap (Jaccard) median %.2f (%.2f–%.2f)"
              % (pp["lexical_overlap_jaccard"]["median"],
                 pp["lexical_overlap_jaccard"]["min"],
                 pp["lexical_overlap_jaccard"]["max"]))
        print("                     char edit / len   median %.2f (%.2f–%.2f)"
              % (pp["char_edit_distance_normalised"]["median"],
                 pp["char_edit_distance_normalised"]["min"],
                 pp["char_edit_distance_normalised"]["max"]))
        print("                     sentence-count delta: %s" % pp["sentence_count_delta"])
        print("                     answer unchanged in %d/%d (asserted)" % (pp["n"], pp["n"]))
        print()

    lv = summary["linguistic_variation"]
    print("linguistic variation (nl_base):")
    print("  question form    : %s" % ", ".join("%s=%d" % kv for kv in lv["question_form"].items()))
    print("  sentences/problem: %s" % ", ".join("%s=%d" % kv for kv in lv["n_sentences"].items()))
    print("  words/problem    : min %d, median %d, max %d"
          % (lv["n_words"]["min"], lv["n_words"]["median"], lv["n_words"]["max"]))
    print("  clause-marker signatures: %d distinct" % lv["distinct_clause_marker_signatures"])
    print("  question final?  : %s" % lv["question_is_final_sentence"])
    print("  opens w/ name?   : %s" % lv["opens_with_proper_name"])
    print()
    print("text length (chars): " + "; ".join(
        "%s median %d (%d–%d)" % (c, v["median"], v["min"], v["max"])
        for c, v in summary["text_length_chars"].items()
    ))
    print()
    print("NOTE: this is the mechanical build. Gate G3 (round-trip verification "
          "by a DIFFERENT model than the generator, prereg/PRIMARY_ENDPOINTS.md) "
          "is a separate step and its drop rate is reported separately.")
    # A few build failures are logged, not fatal (A.5 wants the drop rate). A
    # corpus that came up short of the requested size is a different matter.
    if len(records) < args.n_problems:
        print("\nFATAL: asked for %d problems, built %d." % (args.n_problems, len(records)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
