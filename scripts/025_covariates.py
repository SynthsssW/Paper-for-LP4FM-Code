#!/usr/bin/env python3
"""Build H2's surface-dissimilarity covariates (and H1's anchor labels) from the corpus.

Runs on the laptop, before or after the pod. No torch, no GPU, no activations -- it reads
`data/corpus/problems.jsonl` and writes JSON that `scripts/04_analyze.py --covariates`
consumes.

WHY: without this file `04_analyze.py` fits H2 UNCONTROLLED and stamps
`"primary_endpoint_complete": false`. The pre-registered H2 endpoint is the coefficient on
answer-survival AFTER surface dissimilarity is controlled; the uncontrolled coefficient is
not that endpoint. DESIGN.md A.7 makes the *syntactic-vs-lexical* contrast the part Han et
al. do not have, so the two families are computed separately and never conflated.

OUTPUTS (all under --out-dir, default data/corpus/)

  covariates.json             model-ready, exactly the two pre-registered controls:
                                {problem_id: {condition: {lex_dissim, syn_dissim}}}
                              lex_dissim = normalized character Levenshtein
                              syn_dissim = dependency-label bag TV distance
                              `04_analyze.py` enters EVERY key it finds as a covariate
                              (see its `cov_names` union), so this file carries the two
                              controls and nothing else -- extra keys would silently add
                              collinear terms to the primary model.

  covariates_robustness.json  same shape and same key names, different measures:
                              lex_dissim = token-type Jaccard, syn_dissim = POS-sequence
                              normalized Levenshtein. Drops into `--covariates` unchanged
                              as the robustness refit.

  covariates_measures.json    the honest complete record: every measure for every pair,
                              plus `_meta` (methods, versions, what was skipped). NOT a
                              `--covariates` input; `_meta` and any null would be read as
                              a covariate name / `float(None)` by the consumer.

  linguistic_labels.json      {problem_id: {condition: {anchor_name: dep_label}}}, the
                              dependency relation of the token at each anchor's character
                              span. H1's descriptive AMI note needs this. Written ONLY
                              when a real parser loaded.

NO PARSER, NO SYNTAX. If spaCy is absent the syntactic measures are written as explicit
nulls in covariates_measures.json with `syn_dissim_method: "unavailable"`,
covariates.json falls back to the lexical control alone, linguistic_labels.json is not
written, and the script says so on stderr. A fabricated syntactic covariate would leave
H2 claiming a control it does not have; a missing one only weakens it honestly.

Usage:
    python scripts/025_covariates.py
    python scripts/025_covariates.py --corpus data/corpus/problems.jsonl --out-dir /tmp/x
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import dissimilarity as D  # noqa: E402


def _log(msg=""):
    print(msg, file=sys.stderr)


def _load_corpus(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build(records, analyzer, reference, conditions):
    """-> (measures, labels, skipped) with measures[pid][cond] = full measure dict."""
    measures, labels, skipped = {}, {}, []
    for rec in records:
        pid = rec["problem_id"]
        reals = rec.get("realizations", {})
        ref = reals.get(reference)
        if ref is None or not ref.get("text", "").strip():
            skipped.append({"problem_id": pid, "condition": reference,
                            "reason": "reference realization has no text"})
            continue
        ref_text = ref["text"]
        ref_parse = analyzer.parse(ref_text) if analyzer.available else None

        if analyzer.available:
            labels.setdefault(pid, {})[reference] = {
                name: analyzer.anchor_label(ref_parse, span)
                for name, span in sorted(ref.get("anchors", {}).items())
            }

        for cond in conditions:
            real = reals.get(cond)
            if real is None:
                skipped.append({"problem_id": pid, "condition": cond,
                                "reason": "condition absent from record"})
                continue
            text = real.get("text", "")
            if not text.strip():
                skipped.append({"problem_id": pid, "condition": cond,
                                "reason": "realization is an empty placeholder"
                                          + (" (status=%s)" % real["status"]
                                             if real.get("status") else "")})
                continue
            parse = analyzer.parse(text) if analyzer.available else None
            measures.setdefault(pid, {})[cond] = D.pair_measures(
                ref_text, text, analyzer, parse_ref=ref_parse, parse_other=parse)
            if analyzer.available:
                labels.setdefault(pid, {})[cond] = {
                    name: analyzer.anchor_label(parse, span)
                    for name, span in sorted(real.get("anchors", {}).items())
                }
    return measures, labels, skipped


def project(measures, source, keys):
    """Full measures -> model-ready {pid: {cond: {key: float}}}, dropping null pairs."""
    out, dropped = {}, []
    for pid, per_cond in measures.items():
        for cond, vals in per_cond.items():
            picked = D.select(vals, {k: source[k] for k in keys})
            if picked is None:
                dropped.append((pid, cond))
                continue
            out.setdefault(pid, {})[cond] = picked
    return out, dropped


def check_consumer_shape(cov, keys):
    """Replay `scripts/04_analyze.py`'s covariate handling and fail here, not there.

    That script takes the UNION of keys across all pairs, then requires every key on
    every pair and calls float() on it -- a key present for some pairs and absent for
    others is a hard SystemExit mid-analysis. Better to catch it at write time.
    """
    seen = set()
    for per_cond in cov.values():
        for vals in per_cond.values():
            seen.update(vals)
    problems = []
    if seen != set(keys):
        problems.append("key union is %s, expected %s" % (sorted(seen), sorted(keys)))
    for pid, per_cond in cov.items():
        for cond, vals in per_cond.items():
            for name in sorted(seen):
                if name not in vals:
                    problems.append("(%s, %s) is missing %r" % (pid, cond, name))
                    continue
                try:
                    float(vals[name])
                except (TypeError, ValueError):
                    problems.append("(%s, %s)[%r] is not float()-able: %r"
                                    % (pid, cond, name, vals[name]))
    return problems


def _summary(measures, key):
    per_cond = {}
    for per in measures.values():
        for cond, vals in per.items():
            v = vals.get(key)
            if v is not None:
                per_cond.setdefault(cond, []).append(float(v))
    out = {}
    for cond, vals in per_cond.items():
        vals = sorted(vals)
        n = len(vals)
        out[cond] = {
            "n": n,
            "mean": sum(vals) / n,
            "median": vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0,
            "min": vals[0],
            "max": vals[-1],
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data/corpus/problems.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "data/corpus"))
    ap.add_argument("--reference-condition", default=D.REFERENCE_CONDITION)
    ap.add_argument("--conditions", nargs="+", default=list(D.COMPARED_CONDITIONS),
                    help="conditions compared against the reference")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--require-syntactic", action="store_true",
                    help="exit non-zero instead of degrading to a lexical-only control")
    args = ap.parse_args(argv)

    records = _load_corpus(args.corpus)
    _log("corpus: %d problems from %s" % (len(records), args.corpus))

    analyzer = D.load_syntactic_analyzer(args.spacy_model)
    if analyzer.available:
        _log("parser: spacy %s / %s %s" % (analyzer.spacy_version, analyzer.model_name,
                                           analyzer.model_version))
    else:
        _log("")
        _log("!" * 86)
        _log("!! NO SYNTACTIC PARSER. syn_dissim is NULL, method 'unavailable'.")
        _log("!! reason: %s" % analyzer.reason)
        _log("!! H2 will be controlled for LEXICAL dissimilarity ONLY. The syntactic-vs-")
        _log("!! lexical contrast that DESIGN.md A.7 names as our contribution over Han")
        _log("!! et al. CANNOT be reported from this run. No lexical proxy is being")
        _log("!! substituted -- a fake syntactic control is worse than a missing one.")
        _log("!! linguistic_labels.json is NOT written, so H1's descriptive note is")
        _log("!! unavailable too. Fix: pip install spacy && python -m spacy download %s"
             % args.spacy_model)
        _log("!" * 86)
        _log("")
        if args.require_syntactic:
            return 2

    measures, labels, skipped = build(records, analyzer, args.reference_condition,
                                      args.conditions)

    keys = list(D.PRIMARY_KEYS) if analyzer.available else ["lex_dissim"]
    primary, dropped_p = project(measures, D.PRIMARY_SOURCE, keys)
    robust, dropped_r = project(measures, D.ROBUSTNESS_SOURCE, keys)

    problems = check_consumer_shape(primary, keys) + check_consumer_shape(robust, keys)
    if problems:
        _log("REFUSING TO WRITE -- output would break scripts/04_analyze.py:")
        for p in problems[:20]:
            _log("  " + p)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": os.path.relpath(args.corpus, _REPO),
        "reference_condition": args.reference_condition,
        "conditions": list(args.conditions),
        "n_problems_in_corpus": len(records),
        "n_pairs_with_covariates": sum(len(v) for v in primary.values()),
        "lex_dissim_method": "normalized character Levenshtein distance, "
                             "dist / max(len_a, len_b)",
        "lex_dissim_robustness_method": "token-type Jaccard distance, lowercased "
                                        "[A-Za-z0-9_]+ runs, numerals retained",
        "syn_dissim_method": ("dependency-label bag total-variation distance from a "
                              "spacy dependency parse" if analyzer.available
                              else "unavailable"),
        "syn_dissim_robustness_method": ("POS-sequence normalized Levenshtein"
                                         if analyzer.available else "unavailable"),
        "syn_tree_edit_distance": "NOT computed (see ling.dissimilarity."
                                  "SyntacticAnalyzer docstring)",
        "parser": ({"spacy": analyzer.spacy_version, "model": analyzer.model_name,
                    "model_version": analyzer.model_version}
                   if analyzer.available else None),
        "parser_unavailable_reason": None if analyzer.available else analyzer.reason,
        "primary_file_keys": {k: D.PRIMARY_SOURCE[k] for k in keys},
        "robustness_file_keys": {k: D.ROBUSTNESS_SOURCE[k] for k in keys},
        "skipped_pairs": skipped,
        "summary": {m: _summary(measures, m) for m in
                    ("lex_edit", "lex_jaccard", "char_edit_distance",
                     "syn_dep_bag", "syn_dep_triple", "syn_pos_seq")},
        "caveats": [
            "formal is Python source; an English dependency parser applied to it "
            "produces labels that are consistent but not linguistically meaningful. "
            "Its syn_dissim is a real parse distance, not a claim about English syntax.",
        ],
    }
    if not analyzer.available:
        meta["caveats"].append(
            "syn_dissim is null everywhere; covariates.json carries the lexical control "
            "only and H2's syntactic-vs-lexical contrast cannot be reported.")

    paths = {
        "covariates.json": primary,
        "covariates_robustness.json": robust,
        "covariates_measures.json": dict(measures, _meta=meta),
    }
    if analyzer.available:
        paths["linguistic_labels.json"] = labels
    for name, obj in paths.items():
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
        _log("wrote %s" % os.path.join(args.out_dir, name))

    # ---- sanity expectations (reported, never enforced) -------------------------------
    _log("")
    _log("pairs with covariates: %d (%d problems)"
         % (meta["n_pairs_with_covariates"], len(primary)))
    if skipped:
        by_cond = {}
        for s in skipped:
            by_cond[s["condition"]] = by_cond.get(s["condition"], 0) + 1
        _log("SKIPPED pairs by condition: %s -- these conditions have NO covariate rows, "
             "and 04_analyze.py raises SystemExit if a fork row exists for them."
             % by_cond)
    for m in ("lex_edit", "lex_jaccard", "syn_dep_bag", "syn_pos_seq"):
        s = meta["summary"].get(m) or {}
        if s:
            _log("%-16s %s" % (m, "  ".join(
                "%s: mean=%.4f median=%.4f max=%.4f" % (c, v["mean"], v["median"],
                                                        v["max"])
                for c, v in sorted(s.items()))))
    brk = meta["summary"].get("char_edit_distance", {}).get("nl_break")
    if brk:
        n_one = sum(1 for per in measures.values()
                    if per.get("nl_break", {}).get("char_edit_distance") == 1)
        _log("nl_break char edit distance: median=%.1f max=%.1f; %d/%d differ from "
             "nl_base by exactly ONE character" % (brk["median"], brk["max"], n_one,
                                                   brk["n"]))
        if brk["median"] > 5:
            _log("*** LOUD WARNING: nl_break is NOT lexically close to nl_base "
                 "(median char edit distance %.1f). configs/corpus_schema.md requires "
                 "nl_break to be surface-SIMILAR but answer-breaking. If it is lexically "
                 "far, H2's surface-dissimilarity control is confounded with the "
                 "answer-breaking manipulation and the residual is not interpretable."
                 % brk["median"])
        lex = meta["summary"]["lex_edit"]
        if "formal" in lex and lex["formal"]["mean"] <= lex["nl_break"]["mean"]:
            _log("*** LOUD WARNING: formal is not lexically farther than nl_break "
                 "(mean lex_edit formal=%.4f, nl_break=%.4f). Expected formal to be far "
                 "on both measures." % (lex["formal"]["mean"], lex["nl_break"]["mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
