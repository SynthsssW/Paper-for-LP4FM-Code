#!/usr/bin/env python3
"""Laptop-side analysis: the two primary endpoints, and nothing else unlabelled.

Input:  results/<run_id>/assignments.jsonl  (configs/assignments_schema.md) -- the ONLY
        artifact the GPU pod produces, and the only thing this script reads. No torch,
        no activations, no GPU.
Output: results/analysis.json

WHAT IS PRIMARY (prereg/PRIMARY_ENDPOINTS.md + prereg/DEVIATIONS.md). Exactly TWO
endpoints, because Amendment 1 demoted H1:

  H2  the coefficient on answer-survival status in a discrete-time survival model for
      fork layer, complementary log-log link, person-period expansion, with surface
      dissimilarity entered as controls. Pre-committed direction: NEGATIVE (answer-
      breaking pairs have the higher hazard, i.e. fork earlier).

  H3  the mean paired difference in per-layer-pair lambda2, formal minus nl_base,
      averaged across LAYER-PAIRS (n = 47, Amendment 3), tested by Wilcoxon signed-rank
      paired across layer-pairs plus a permutation null that resamples PROBLEMS
      (Amendment 2). Pre-committed direction: NEGATIVE.

  H1  has NO primary endpoint and NO directional pre-commitment (Amendment 1). AMI vs
      K-means is computed and reported as DESCRIPTIVE, with a bootstrap CI resampling
      problems and no p-value. It is a methods note about the clustering front end. This
      script will not emit a test for it.

Everything else in the output carries "status": "EXPLORATORY".

Optional inputs (never fabricated when absent):
  --covariates PATH   JSON: {problem_id: {condition: {"lex_dissim": x, "syn_dissim": y}}}
                      the surface-dissimilarity controls H2's endpoint is defined with.
                      Absent -> H2 is fitted WITHOUT controls and flagged
                      "primary_endpoint_complete": false. The uncontrolled coefficient is
                      NOT the pre-registered endpoint and is labelled as such.
  H1 needs per-row "linguistic_label" and "kmeans_cluster" fields at the max-eigengap
  layer. Absent -> H1 reports status "unavailable" with the reason. It is never faked.

Usage:
    python scripts/04_analyze.py --run-id 20260906_gemma4
    python scripts/04_analyze.py --assignments results/x/assignments.jsonl --n-boot 500
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ling import metrics as M  # noqa: E402
from ling import stats as S  # noqa: E402


# --------------------------------------------------------------------------------------
# H1 helpers: AMI, pure python (no sklearn dependency)
# --------------------------------------------------------------------------------------


def _contingency(a, b):
    tab = {}
    for x, y in zip(a, b):
        tab[(x, y)] = tab.get((x, y), 0) + 1
    return tab


def mutual_information(a, b):
    n = len(a)
    tab = _contingency(a, b)
    ra, cb = {}, {}
    for (x, y), c in tab.items():
        ra[x] = ra.get(x, 0) + c
        cb[y] = cb.get(y, 0) + c
    mi = 0.0
    for (x, y), c in tab.items():
        mi += (c / n) * math.log((c * n) / float(ra[x] * cb[y]))
    return mi


def _entropy(labels):
    n = len(labels)
    counts = {}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    return -sum((c / n) * math.log(c / float(n)) for c in counts.values())


def expected_mutual_information(a, b):
    """E[MI] under the permutation model (Vinh, Epps & Bailey 2010), via lgamma."""
    n = len(a)
    ra, cb = {}, {}
    for v in a:
        ra[v] = ra.get(v, 0) + 1
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    lg = math.lgamma
    log_n = math.log(n)
    emi = 0.0
    for ai in ra.values():
        for bj in cb.values():
            lo = max(1, ai + bj - n)
            hi = min(ai, bj)
            base = (lg(ai + 1) + lg(bj + 1) + lg(n - ai + 1) + lg(n - bj + 1) - lg(n + 1))
            for nij in range(lo, hi + 1):
                logterm = base - (lg(nij + 1) + lg(ai - nij + 1) + lg(bj - nij + 1)
                                  + lg(n - ai - bj + nij + 1))
                emi += (nij / float(n)) * (math.log(nij) + log_n - math.log(ai) - math.log(bj)) \
                    * math.exp(logterm)
    return emi


def adjusted_mutual_information(a, b):
    """AMI with the arithmetic-mean normalisation (sklearn's default)."""
    if len(a) != len(b):
        raise ValueError("label vectors must be the same length")
    mi = mutual_information(a, b)
    emi = expected_mutual_information(a, b)
    ha, hb = _entropy(a), _entropy(b)
    denom = 0.5 * (ha + hb) - emi
    if abs(denom) < 1e-12:
        return 0.0
    return (mi - emi) / denom


# --------------------------------------------------------------------------------------
# H1 -- DESCRIPTIVE ONLY (Amendment 1)
# --------------------------------------------------------------------------------------


def analyze_h1(a, args):
    """AMI(spectral, linguistic label) vs AMI(kmeans, linguistic label) at the
    max-eigengap layer. DESCRIPTIVE: no p-value, no direction, no primary endpoint."""
    out = {
        "status": "DESCRIPTIVE (Amendment 1: H1 has no primary endpoint and no "
                  "pre-committed direction; this is a methods note on the clustering "
                  "front end, not a test)",
        "is_test": False,
    }
    try:
        layer = a.max_eigengap_layer()
    except M.InvariantViolation as exc:
        out["available"] = False
        out["reason"] = "no eigengap field on any layer: %s" % exc
        return out
    out["layer"] = layer
    out["eigengap"] = a.eigengap_by_layer.get(layer)

    rec = a.record(layer)
    rows = rec["rows"]
    idx = [j for j, (pid, cond, _) in enumerate(a.keys) if cond == args.reference_condition]
    if not idx:
        out["available"] = False
        out["reason"] = "no rows for reference condition %r" % args.reference_condition
        return out
    missing = [f for f in ("linguistic_label", "kmeans_cluster") if f not in rows[idx[0]]]
    if missing:
        out["available"] = False
        out["reason"] = (
            "assignments.jsonl carries no %s field. H1's AMI comparison needs the "
            "linguistic label and the K-means comparator written alongside the spectral "
            "cluster at extraction time; this script consumes assignments.jsonl and "
            "nothing else, and will not fabricate them." % " or ".join(missing))
        return out

    spectral = [rows[j]["cluster"] for j in idx]
    kmeans = [rows[j]["kmeans_cluster"] for j in idx]
    labels = [rows[j]["linguistic_label"] for j in idx]
    problems = [a.keys[j][0] for j in idx]

    ami_s = adjusted_mutual_information(spectral, labels)
    ami_k = adjusted_mutual_information(kmeans, labels)

    by_problem = {}
    for pos, pid in enumerate(problems):
        by_problem.setdefault(pid, []).append(pos)

    def diff(problem_multiset):
        sel = [pos for pid in problem_multiset for pos in by_problem[pid]]
        s = [spectral[i] for i in sel]
        k = [kmeans[i] for i in sel]
        l = [labels[i] for i in sel]
        return adjusted_mutual_information(s, l) - adjusted_mutual_information(k, l)

    ci = None
    if args.n_boot > 0:
        ci = S.bootstrap_ci(sorted(by_problem), diff, n_boot=args.n_boot,
                            alpha=args.alpha, seed=args.seed)
    out.update({
        "available": True,
        "condition": args.reference_condition,
        "k": a.k_by_layer[layer],
        "n_token_states": len(idx),
        "ami_spectral": ami_s,
        "ami_kmeans": ami_k,
        "ami_difference": ami_s - ami_k,
        "bootstrap_ci_difference": ci,
        "note": (
            "DESIGN.md §A.7 / Amendment 1: the descriptive report is still stated at "
            "lambda = 0 (purely geometric affinity). Reporting it at lambda > 0 would be "
            "circular. Demotion does not change that."),
    })
    return out


# --------------------------------------------------------------------------------------
# H2 -- PRIMARY
# --------------------------------------------------------------------------------------


def analyze_h2(a, args, covariates):
    breaking = set(args.breaking_conditions)
    ref = args.reference_condition
    others = [c for c in a.conditions if c != ref]
    if not others:
        return {"status": "PRIMARY", "available": False,
                "reason": "only one condition present"}

    cov_names = []
    if covariates:
        seen = set()
        for per_cond in covariates.values():
            for vals in per_cond.values():
                seen.update(vals)
        cov_names = sorted(seen)

    records = []
    censored = 0
    for cond in others:
        for f in M.fork_layers(a, (ref, cond)):
            rec = {
                "problem_id": f.problem_id,
                "condition": cond,
                "fork": f,
                # answer_survival = 1 when the realization PRESERVES the answer.
                "answer_survival": 0.0 if cond in breaking else 1.0,
            }
            if cov_names:
                vals = (covariates.get(f.problem_id, {}) or {}).get(cond, {})
                for name in cov_names:
                    if name not in vals:
                        raise SystemExit(
                            "covariates file is missing %r for (%s, %s); a control that "
                            "is present for some pairs and absent for others cannot be "
                            "entered as a covariate" % (name, f.problem_id, cond))
                    rec[name] = float(vals[name])
            records.append(rec)
            censored += 1 if f.censored else 0

    fit = S.discrete_time_survival(
        records,
        layers=a.layers,
        survival_key="answer_survival",
        covariates=cov_names,
        baseline=args.baseline_hazard,
        cluster_by_problem=True,
    )
    complete = bool(cov_names)
    return {
        "status": "PRIMARY",
        "hypothesis": "H2",
        "endpoint": ("coefficient on answer-survival status in a discrete-time survival "
                     "model for fork layer, cloglog link, person-period expansion"),
        "available": True,
        "primary_endpoint_complete": complete,
        "covariates_entered": cov_names,
        "condition_pairs": [[ref, c] for c in others],
        "answer_breaking_conditions": sorted(breaking),
        "n_censored_pairs": censored,
        "censoring_handling": ("right-censored pairs contribute all non-fork rows; none "
                               "dropped, no fabricated event layer (DESIGN.md §A.9.1)"),
        "fit": fit,
        "caveat": None if complete else (
            "NO surface-dissimilarity controls were available (--covariates not given). "
            "The pre-registered H2 endpoint is the coefficient on answer-survival AFTER "
            "lexical and syntactic dissimilarity controls. The coefficient reported here "
            "is the UNCONTROLLED one and is NOT the pre-registered primary endpoint."),
    }


# --------------------------------------------------------------------------------------
# H3 -- PRIMARY
# --------------------------------------------------------------------------------------


def _lambda2_vector(a, pairs, cond_formal, cond_base, problems=None):
    out = []
    for p in pairs:
        lf = M.lambda2(a, p, cond_formal, problems=problems)
        lb = M.lambda2(a, p, cond_base, problems=problems)
        out.append(lf - lb)
    return out


def analyze_h3(a, args):
    ref, formal = args.reference_condition, args.formal_condition
    if formal not in a.conditions:
        return {"status": "PRIMARY", "hypothesis": "H3", "available": False,
                "reason": "condition %r is not present in the assignments" % formal}
    pairs = a.layer_pairs()
    if len(pairs) < 2:
        return {"status": "PRIMARY", "hypothesis": "H3", "available": False,
                "reason": "only %d layer-pairs available" % len(pairs)}

    l2_formal = [M.lambda2(a, p, formal) for p in pairs]
    l2_base = [M.lambda2(a, p, ref) for p in pairs]
    diffs = [f - b for f, b in zip(l2_formal, l2_base)]
    mean_diff = sum(diffs) / len(diffs)

    wil = S.wilcoxon_paired(l2_formal, l2_base, alternative="less")

    problems = list(a.problems)
    boot = None
    if args.n_boot > 0:
        def stat(problem_multiset):
            v = _lambda2_vector(a, pairs, formal, ref, problems=problem_multiset)
            return sum(v) / len(v)
        boot = S.bootstrap_ci(problems, stat, n_boot=args.n_boot, alpha=args.alpha,
                              seed=args.seed)

    perm = None
    per_layer = None
    if args.n_perm > 0:
        perm_vectors = []

        def perm_stat(swapped):
            cond_f = {p: (ref if p in swapped else formal) for p in problems}
            cond_b = {p: (formal if p in swapped else ref) for p in problems}
            v = _lambda2_vector(a, pairs, cond_f, cond_b)
            perm_vectors.append(v)
            return sum(v) / len(v)

        perm = S.permutation_null(problems, perm_stat, n_perm=args.n_perm,
                                 alternative="less", seed=args.seed + 1)
        # The same permutation draws give the layer-wise family for free.
        draws = perm_vectors[1:] if len(perm_vectors) > args.n_perm else perm_vectors
        if draws:
            pvals = []
            for i in range(len(pairs)):
                col = [v[i] for v in draws]
                extreme = sum(1 for v in col if v <= diffs[i])
                pvals.append((1.0 + extreme) / (1.0 + len(col)))
            bh = S.benjamini_hochberg(pvals, q=args.q)
            per_layer = {
                "status": "EXPLORATORY",
                "note": ("per-layer-pair permutation p-values (one-sided, less), "
                         "Benjamini-Hochberg corrected across the layer-wise family "
                         "(prereg decision 4). The PRIMARY H3 endpoint is the mean "
                         "across layer-pairs, not any single layer-pair."),
                "layer_pairs": list(pairs),
                "p": pvals,
                "bh": bh,
            }

    logs = S.log_lambda2(l2_formal + l2_base, zero_policy="mask")
    return {
        "status": "PRIMARY",
        "hypothesis": "H3",
        "endpoint": ("mean paired difference in per-layer-pair lambda2, formal minus "
                     "nl_base, averaged across layer-pairs (Amendment 2)"),
        "available": True,
        "direction_precommitted": "NEGATIVE (formal rewriting decreases lambda2; Cheeger)",
        "laplacian": "symmetric normalized (I - D^-1/2 W D^-1/2)",
        "pairing_unit": "layer-pair",
        "n_layer_pairs": len(pairs),
        "n_problems": len(problems),
        "mean_paired_difference": mean_diff,
        "consistent_with_precommitted_direction": mean_diff < 0,
        "wilcoxon": wil,
        "bootstrap_ci": boot,
        "permutation_null": perm,
        "per_layer_pair": per_layer,
        "lambda2_zero_count": logs["n_zero"],
        "lambda2_zero_note": (
            "lambda2 = 0 means the pooled transition graph is disconnected at that "
            "layer-pair for that condition. It is a real value, not a missing one; "
            "prereg decision 6's log-transform masks those %d values wherever a "
            "regression is fitted." % logs["n_zero"]),
        "layer_pairs": list(pairs),
        "lambda2_formal": l2_formal,
        "lambda2_base": l2_base,
        "paired_differences": diffs,
    }


# --------------------------------------------------------------------------------------
# Exploratory block
# --------------------------------------------------------------------------------------


def analyze_exploratory(a, args):
    pairs = a.layer_pairs()
    out = {"status": "EXPLORATORY"}
    out["path_entropy_normalized"] = {
        "status": "EXPLORATORY",
        "note": "conditional entropy of the transition chain / log k, in [0, 1]",
        "layer_pairs": list(pairs),
        "by_condition": {
            c: [M.path_entropy(a, p, c, normalized=True) for p in pairs]
            for c in a.conditions
        },
    }
    out["lambda2_by_condition"] = {
        "status": "EXPLORATORY",
        "layer_pairs": list(pairs),
        "by_condition": {c: [M.lambda2(a, p, c) for p in pairs] for c in a.conditions},
    }
    corr = {}
    for c in a.conditions:
        if c == args.reference_condition:
            continue
        try:
            corr[c] = M.metric_correlation_matrix(a, c, args.reference_condition)
        except Exception as exc:  # a degenerate condition should not kill the run
            corr[c] = {"status": "EXPLORATORY", "error": str(exc)}
    out["metric_correlation_matrix"] = {
        "status": "EXPLORATORY",
        "note": ("DESIGN.md §A.8: correlation across the three retained metrics "
                 "(lambda2, path entropy, fork layer), common index = layer-pair."),
        "by_condition": corr,
    }
    fork = {}
    for c in a.conditions:
        if c == args.reference_condition:
            continue
        fs = M.fork_layers(a, (args.reference_condition, c))
        observed = [f.layer for f in fs if not f.censored]
        fork[c] = {
            "n_pairs": len(fs),
            "n_forked": len(observed),
            "n_censored": sum(1 for f in fs if f.censored),
            "median_fork_layer_observed_only": (
                sorted(observed)[len(observed) // 2] if observed else None),
            "warning": ("median over observed-only forks is biased by censoring and is "
                        "descriptive only; H2's survival model is the unbiased summary"),
        }
    out["fork_layer_summary"] = {"status": "EXPLORATORY", "by_condition": fork}
    return out


# --------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None,
                    help="reads results/<run_id>/assignments.jsonl")
    ap.add_argument("--assignments", default=None,
                    help="explicit path to assignments.jsonl (overrides --run-id)")
    ap.add_argument("--out", default=os.path.join(_REPO, "results", "analysis.json"))
    ap.add_argument("--covariates", default=None,
                    help="JSON {problem_id: {condition: {lex_dissim: .., syn_dissim: ..}}}")
    ap.add_argument("--reference-condition", default="nl_base")
    ap.add_argument("--formal-condition", default="formal")
    ap.add_argument("--breaking-conditions", nargs="*", default=["nl_break"],
                    help="conditions that BREAK the answer (answer_survival = 0)")
    ap.add_argument("--baseline-hazard", default="linear",
                    choices=["linear", "log", "dummy", "none"])
    ap.add_argument("--n-boot", type=int, default=200,
                    help="bootstrap resamples (of PROBLEMS); 0 to skip")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutation draws (of PROBLEM condition labels); 0 to skip")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=0.05, help="BH FDR level, layer-wise family")
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args(argv)

    path = args.assignments
    if path is None:
        if args.run_id is None:
            ap.error("one of --run-id or --assignments is required")
        path = os.path.join(_REPO, "results", args.run_id, "assignments.jsonl")
    a = M.load_assignments(path)

    covariates = None
    if args.covariates:
        with open(args.covariates, "r", encoding="utf-8") as fh:
            covariates = json.load(fh)

    affinities = sorted({v for v in a.affinity_by_layer.values() if v is not None})
    result = {
        "meta": {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "assignments": os.path.abspath(path),
            "n_layers": len(a.layers),
            "layers": [a.layers[0], a.layers[-1]],
            "n_layer_pairs": len(a.layer_pairs()),
            "n_problems": len(a.problems),
            "n_rows_per_layer": len(a.keys),
            "conditions": list(a.conditions),
            "affinity": affinities,
            "laplacian": "symmetric normalized",
            "resampling_unit": "problem",
            "seed": args.seed,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "numpy": M.HAVE_NUMPY,
            "prereg": "prereg/PRIMARY_ENDPOINTS.md + prereg/DEVIATIONS.md (4 amendments)",
        },
        "primary_endpoints": {
            "count": 2,
            "note": ("Two primary endpoints only. Amendment 1 demoted H1; it has no "
                     "primary endpoint and no directional pre-commitment."),
            "h2": analyze_h2(a, args, covariates),
            "h3": analyze_h3(a, args),
        },
        "h1_descriptive": analyze_h1(a, args),
        "exploratory": analyze_exploratory(a, args),
        "negative_result_commitment": (
            "If the primary endpoints do not reach significance the result is reported as "
            "a bounded negative result, not reframed around whichever exploratory "
            "endpoint reached p < 0.05 (prereg/PRIMARY_ENDPOINTS.md)."),
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")

    h3 = result["primary_endpoints"]["h3"]
    h2 = result["primary_endpoints"]["h2"]
    print("wrote %s" % args.out)
    if h2.get("available"):
        pr = h2["fit"]["primary"]
        print("H2 PRIMARY  coef(answer_survival) = %+.4f  se %.4f  p %.4g  %s"
              % (pr["coef"], pr["se"], pr["p"],
                 "controlled" if h2["primary_endpoint_complete"] else "UNCONTROLLED"))
    if h3.get("available"):
        print("H3 PRIMARY  mean paired d(lambda2) = %+.6f over %d layer-pairs, "
              "Wilcoxon(less) p = %.4g"
              % (h3["mean_paired_difference"], h3["n_layer_pairs"], h3["wilcoxon"]["p"]))
    print("H1 is DESCRIPTIVE only (Amendment 1): no test, no direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
