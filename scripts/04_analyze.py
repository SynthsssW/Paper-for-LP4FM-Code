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

SECONDARY endpoints (Amendment 6), under the top-level "secondary_endpoints" key. They
are reported ALONGSIDE the two primaries and never in place of them:

  H2 secondary, PRE-REGISTERED: hazard ratio for fork layer, nl_break vs
      nl_minimal_orth, in the same cloglog model. Both arms are one-token edits at the
      same quantity anchor and differ only in whether the value changes, so the
      contrast is not confounded with surface dissimilarity. Pre-committed direction:
      nl_break forks EARLIER (hazard ratio > 1).

  SND, INTERPRETIVE SECONDARY: style-normalized displacement of lambda2 and path
      entropy against BOTH style floors, reported as a RANGE. The raw difference with a
      problem-level bootstrap CI stays PRIMARY. SND is emitted only where the
      denominator's bootstrap CI excludes zero; elsewhere it is null with a reason.

Everything else in the output carries "status": "EXPLORATORY".

Optional inputs (never fabricated when absent):
  --covariates PATH   JSON: {problem_id: {condition: {"lex_dissim": x, "syn_dissim": y}}}
                      the surface-dissimilarity controls H2's endpoint is defined with.
                      Absent -> H2 is fitted WITHOUT controls and flagged
                      "primary_endpoint_complete": false. The uncontrolled coefficient is
                      NOT the pre-registered endpoint and is labelled as such.
                      A wholly absent (problem, condition) entry is an Amendment 6
                      unavailable cell: it is EXCLUDED upstream of the model and counted
                      under h2.availability, never imputed. An entry that exists but is
                      missing one control is an inconsistency and still stops the run.
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
    all_null = [f for f in ("linguistic_label", "kmeans_cluster")
                if all(rows[j][f] is None for j in idx)]
    if all_null:
        out["available"] = False
        out["reason"] = (
            "assignments.jsonl's %s field is present but every value is null for "
            "condition %r. H1's AMI comparison needs real values, not an all-null "
            "column; an AMI computed against an all-null field is a fabricated 0.0, "
            "not a result." % (" and ".join(all_null), args.reference_condition))
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


def _count_fork_events(records):
    """Uncensored (i.e. actually-forked) pairs. Censoring is always explicit."""
    n = 0
    for r in records:
        f = r.get("fork")
        if f is None:
            censored = r.get("censored")
        elif isinstance(f, dict):
            censored = f.get("censored")
        else:
            censored = getattr(f, "censored", None)
        if censored is False:
            n += 1
    return n


def _fit_survival_or_report(records, layers, **kw):
    """Fit the cloglog survival model, or return an UNFITTABLE REPORT instead of dying.

    IRLS needs events. A dataset in which no pair ever forks (or in which the design is
    singular, e.g. perfect separation) has no identified coefficient, and
    `ling.stats.discrete_time_survival` correctly refuses it. What it must NOT do is
    take the whole analysis down with it: this script runs after the pod is torn down,
    and losing H3, the SND block and the exploratory tables because one model had zero
    events would mean re-renting a GPU to recover numbers that were already computed.

    So the refusal is caught HERE and turned into a report carrying the reason and the
    event count, and the caller continues. The report has no "primary" key and
    "fitted": False, so nothing downstream can mistake it for a fitted coefficient.
    """
    n_events = _count_fork_events(records)
    try:
        return S.discrete_time_survival(records, layers=layers, **kw)
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return {
            "fitted": False,
            "link": "cloglog",
            "model": "discrete-time survival, person-period expansion",
            "unfittable_reason": str(exc),
            "n_pairs": len(records),
            "n_events": n_events,
            "n_censored": len(records) - n_events,
            "note": ("The model could not be fitted. No coefficient is reported and none "
                     "is imputed. The remaining analyses in this file were computed and "
                     "are unaffected."),
        }


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
    excluded = {}
    for cond in others:
        # Amendment 6: the style-control conditions are ragged (nl_minimal_orth 99/100,
        # nl_minimal_lex 70/100, 31 unavailable cells in all). A cell with no
        # realization has no prompt, no activation and no fork layer, so it is excluded
        # HERE, upstream of the model, rather than entering the person-period expansion
        # as a row with a missing covariate. The counts are reported below.
        avail = M.fork_layers(a, (ref, cond))
        no_rows = [p for p in a.problems if (p, cond) not in a.index_by_pc]
        no_covariate = []
        for f in avail:
            vals = (covariates.get(f.problem_id, {}) or {}).get(cond) if cov_names else {}
            if cov_names and not vals:
                # The covariates builder skips unavailable realizations the same way, so
                # an absent ENTRY is the same hole, not an inconsistency.
                no_covariate.append(f.problem_id)
                continue
            rec = {
                "problem_id": f.problem_id,
                "condition": cond,
                "fork": f,
                # answer_survival = 1 when the realization PRESERVES the answer.
                "answer_survival": 0.0 if cond in breaking else 1.0,
            }
            for name in cov_names:
                if name not in vals:
                    raise SystemExit(
                        "covariates file has an entry for (%s, %s) but it is missing %r; "
                        "a control that is present for some pairs and absent for others "
                        "cannot be entered as a covariate. (A wholly absent entry is an "
                        "Amendment 6 unavailable cell and is excluded instead.)"
                        % (f.problem_id, cond, name))
                rec[name] = float(vals[name])
            records.append(rec)
            censored += 1 if f.censored else 0
        excluded[cond] = {
            "n_problems": len(a.problems),
            "n_entered": sum(1 for r in records if r["condition"] == cond),
            "n_excluded": len(no_rows) + len(no_covariate),
            "n_excluded_no_realization": len(no_rows),
            "n_excluded_no_covariate_entry": len(no_covariate),
            "excluded_problems": sorted(set(no_rows) | set(no_covariate)),
        }
    if not records:
        return {"status": "PRIMARY", "hypothesis": "H2", "available": False,
                "reason": "every problem x condition cell was excluded as unavailable"}

    fit = _fit_survival_or_report(
        records,
        a.layers,
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
        "n_pairs_entered": len(records),
        "availability": {
            "note": ("Amendment 6: unavailable style-control cells are genuine holes -- "
                     "no realization was built, so no fork layer exists. They are "
                     "excluded upstream of the model, never imputed, and counted here. "
                     "Right-CENSORING is a different thing and is handled inside the "
                     "model; nothing below was censored by exclusion."),
            "total_cells_excluded": sum(v["n_excluded"] for v in excluded.values()),
            "by_condition": excluded,
        },
        "fit": fit,
        "model_fitted": bool(fit.get("fitted")),
        "secondary_endpoints_note": (
            "A pre-registered SECONDARY contrast for H2 (nl_break vs nl_minimal_orth, "
            "Amendment 6) is reported under the top-level `secondary_endpoints` key. It "
            "does NOT replace the endpoint in this block, whichever of the two looks "
            "better."),
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
# SECONDARY endpoints (DEVIATIONS Amendment 6) -- pre-registered, and NOT substitutes
# --------------------------------------------------------------------------------------


SECONDARY_STATUS = (
    "PRE-REGISTERED SECONDARY (DEVIATIONS Amendment 6). This does NOT replace H2's "
    "primary endpoint -- the coefficient on answer-survival with the surface-"
    "dissimilarity covariates, reported under primary_endpoints.h2. BOTH are reported, "
    "and the primary is the primary whichever one comes out looking better. Promoting "
    "this contrast because the primary disappointed is the exact failure mode the "
    "pre-registration exists to prevent."
)


def analyze_h2_secondary(a, args):
    """Hazard ratio for fork layer, nl_break vs nl_minimal_orth (Amendment 6).

    The two arms are matched on edit SITE -- both edit the quantity anchor, 95 of 99 the
    exact same slot -- and differ in whether the value, and hence the answer, changes.
    They are NOT matched on edit magnitude: Amendment 6's correction of
    2026-09-07T00:45 measured nl_minimal_orth at ~8x nl_break's edit size (lex_edit
    means 0.0486 vs 0.0050), because digits-to-words is a multi-character rewrite. The
    phrase "matched magnitude" must not be used for this contrast.

    That asymmetry makes the pre-committed direction CONSERVATIVE rather than fragile:
    nl_minimal_orth is the LARGER surface edit, so any surface-driven forking pushes it
    to fork earlier -- against the prediction that nl_break forks earlier. A confirmed
    result is therefore not explicable by edit size.

    No dissimilarity covariates are entered. The design is a two-arm contrast at a
    matched site, the arms' surface distances are what the contrast is about rather than
    a nuisance measured alongside it, and with one indicator and a dissimilarity control
    that separates the arms almost perfectly the design would be near-singular.

    Same model form as H2's primary: cloglog discrete-time survival, person-period
    expansion, cluster-robust SE on problem. Both arms come from the same problems, so
    the problem cluster carries the pairing.
    """
    brk, floor = args.break_condition, args.style_floor_upper
    out = {
        "status": SECONDARY_STATUS,
        "hypothesis": "H2",
        "endpoint": ("hazard ratio for fork layer, %s vs %s, in the cloglog discrete-"
                     "time survival model" % (brk, floor)),
        "direction_precommitted": (
            "%s forks EARLIER than the matched-SITE style edit %s: coefficient POSITIVE, "
            "hazard ratio > 1" % (brk, floor)),
        "matching": ("matched on edit SITE (the quantity anchor), NOT on edit magnitude: "
                     "%s is ~8x the edit size of %s (lex_edit means 0.0486 vs 0.0050, "
                     "Amendment 6 correction). The pre-committed direction is therefore "
                     "CONSERVATIVE -- the larger surface edit is the one predicted to "
                     "fork LATER." % (floor, brk)),
        "replaces_primary": False,
        "covariates_entered": [],
        "covariates_note": (
            "None, deliberately: the arms are matched on edit site by construction "
            "(Amendment 6), and a dissimilarity control that separates the two arms "
            "almost perfectly would only make the design near-singular."),
    }
    missing = [c for c in (args.reference_condition, brk, floor) if c not in a.conditions]
    if missing:
        out["available"] = False
        out["reason"] = ("conditions %r are not present in the assignments (present: %r). "
                         "This contrast needs the Amendment 6 style controls in the "
                         "corpus." % (missing, list(a.conditions)))
        return out

    ref = args.reference_condition
    # Both arms on the SAME problems: a problem that has no nl_minimal_orth realization
    # (1 of 100 under Amendment 6) contributes to neither arm, so the contrast stays
    # within-problem and the excluded count is reported rather than absorbed.
    common = a.problems_for(ref, brk, floor)
    excluded = [p for p in a.problems if p not in set(common)]
    if len(common) < 2:
        out["available"] = False
        out["reason"] = ("only %d problem(s) have all of %r, %r and %r"
                         % (len(common), ref, brk, floor))
        return out
    records = []
    for cond, is_break in ((brk, 1.0), (floor, 0.0)):
        for f in M.fork_layers(a, (ref, cond), problems=common):
            records.append({
                "problem_id": f.problem_id,
                "condition": cond,
                "fork": f,
                "is_answer_breaking": is_break,
            })
    fit = _fit_survival_or_report(
        records,
        a.layers,
        survival_key="is_answer_breaking",
        covariates=(),
        baseline=args.baseline_hazard,
        cluster_by_problem=True,
        expected_direction="positive",
        direction_note=out["direction_precommitted"],
    )
    out.update({
        "available": True,
        "model_fitted": bool(fit.get("fitted")),
        "contrast": {"arm_1": brk, "arm_0": floor, "coded": "is_answer_breaking"},
        "reference_condition": ref,
        "n_pairs": len(records),
        "n_censored_pairs": len(records) - _count_fork_events(records),
        "availability": {
            "n_problems_in_both_arms": len(common),
            "n_problems": len(a.problems),
            "n_problems_excluded_unavailable": len(excluded),
            "excluded_problems": excluded,
            "note": ("Amendment 6 unavailable cells are excluded upstream of the model, "
                     "not entered as missing rows; both arms use the same problems."),
        },
        "fit": fit,
    })
    if fit.get("fitted"):
        prim = fit["primary"]
        out["hazard_ratio"] = prim["hazard_ratio"]
        out["hazard_ratio_ci95"] = prim["hazard_ratio_ci95"]
        out["p"] = prim["p"]
        out["consistent_with_precommitted_direction"] = \
            prim["consistent_with_precommitted_direction"]
    return out


def analyze_snd(a, args):
    """Style-normalized displacement for lambda2 and path entropy, both floors, as a
    RANGE (Amendment 6). INTERPRETIVE SECONDARY; the raw difference stays primary."""
    out = {
        "status": M.SND_STATUS,
        "formula": "SND(c, m) = [m(c) - m(nl_base)] / [m(nl_minimal, m) - m(nl_base)]",
        "floors": {"upper_bracket": args.style_floor_upper,
                   "lower_bracket": args.style_floor_lower},
        "guard": ("SND is reported ONLY where the denominator's problem-level bootstrap "
                  "CI excludes zero; otherwise it is null with a reason. The guard is "
                  "inside ling.metrics.style_normalized_displacement, so there is no "
                  "code path that emits a ratio through a near-zero denominator."),
    }
    missing = [c for c in (args.reference_condition, args.style_floor_upper,
                           args.style_floor_lower) if c not in a.conditions]
    if missing:
        out["available"] = False
        out["reason"] = ("conditions %r are not present in the assignments (present: %r)"
                         % (missing, list(a.conditions)))
        return out
    floors = {args.style_floor_upper, args.style_floor_lower, args.reference_condition}
    targets = [c for c in a.conditions if c not in floors]
    cache = {}
    by_metric = {}
    for metric in M.SND_METRICS:
        by_condition = {}
        for cond in targets:
            pre = ("abs_gt_1" if (metric == "lambda2" and cond == args.formal_condition)
                   else None)
            by_condition[cond] = M.snd_range(
                a, metric, cond,
                upper_floor=args.style_floor_upper,
                lower_floor=args.style_floor_lower,
                base_condition=args.reference_condition,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
                precommitted_upper=pre, cache=cache)
        by_metric[metric] = by_condition
    out["available"] = True
    out["conditions"] = targets
    out["availability_note"] = (
        "Each (condition, floor) SND is computed on the problems where the base, the "
        "condition and that floor all exist (Amendment 6), so the two floors of one "
        "range rest on different problem subsets; each carries its own n_problems and "
        "n_problems_excluded_unavailable.")
    out["by_metric"] = by_metric
    out["n_undefined"] = sum(
        1 for m in by_metric.values() for c in m.values()
        for b in c["by_floor"].values() if b["snd"] is None)
    return out


# --------------------------------------------------------------------------------------
# Exploratory block
# --------------------------------------------------------------------------------------


def analyze_exploratory(a, args):
    pairs = a.layer_pairs()
    out = {"status": "EXPLORATORY"}
    ragged = ("Each condition is pooled over the problems where that condition EXISTS "
              "(Amendment 6: nl_minimal_orth and nl_minimal_lex are ragged), so the "
              "n_problems below differ by condition and the columns are not pooled over "
              "an identical corpus. Every cross-condition endpoint elsewhere in this "
              "file is computed on a common problem set instead; this table is "
              "descriptive.")
    n_by_cond = {c: len(a.problems_for(c)) for c in a.conditions}
    out["path_entropy_normalized"] = {
        "status": "EXPLORATORY",
        "note": "conditional entropy of the transition chain / log k, in [0, 1]. " + ragged,
        "layer_pairs": list(pairs),
        "n_problems_by_condition": n_by_cond,
        "by_condition": {
            c: [M.path_entropy(a, p, c, normalized=True) for p in pairs]
            for c in a.conditions
        },
    }
    out["lambda2_by_condition"] = {
        "status": "EXPLORATORY",
        "note": ragged,
        "layer_pairs": list(pairs),
        "n_problems_by_condition": n_by_cond,
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
                 "(lambda2, path entropy, fork layer), common index = layer-pair. "
                 "Computed for EVERY condition present in the assignments -- all six "
                 "under Amendment 6, the two style floors included -- against the "
                 "reference condition; the reference itself has no fork-layer pair with "
                 "itself and so is not a row."),
        "conditions_covered": sorted(corr),
        "n_conditions_covered": len(corr),
        "reference_condition": args.reference_condition,
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
            "n_problems_excluded_unavailable": len(a.problems) - len(fs),
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
    ap.add_argument("--break-condition", default="nl_break",
                    help="the answer-breaking arm of the Amendment 6 secondary contrast")
    ap.add_argument("--style-floor-upper", default="nl_minimal_orth",
                    help="Amendment 6 style floor, UPPER bracket (on-anchor orthographic "
                         "re-rendering); also the comparison arm of the H2 secondary")
    ap.add_argument("--style-floor-lower", default="nl_minimal_lex",
                    help="Amendment 6 style floor, LOWER bracket (off-anchor name swap)")
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
            "prereg": "prereg/PRIMARY_ENDPOINTS.md + prereg/DEVIATIONS.md (6 amendments)",
            "n_conditions": len(a.conditions),
            "availability": {
                "note": ("Amendment 6: the style-control conditions are ragged. An "
                         "unavailable problem x condition cell is a genuine hole -- no "
                         "realization, no prompt, no activation -- and is excluded "
                         "upstream of every model rather than imputed. Counted here so "
                         "that availability is a reported number."),
                "by_condition": {c: a.availability(c) for c in a.conditions},
                "total_cells_unavailable": sum(
                    len(a.problems) - len(a.problems_for(c)) for c in a.conditions),
            },
        },
        "primary_endpoints": {
            "count": 2,
            "note": ("Two primary endpoints only. Amendment 1 demoted H1; it has no "
                     "primary endpoint and no directional pre-commitment."),
            "h2": analyze_h2(a, args, covariates),
            "h3": analyze_h3(a, args),
        },
        "secondary_endpoints": {
            "note": ("Amendment 6. Reported ALONGSIDE the primary endpoints, never in "
                     "place of them. Neither entry below is a primary endpoint and "
                     "neither becomes one by outperforming the primary."),
            "h2_fork_hazard_break_vs_style_floor": analyze_h2_secondary(a, args),
            "style_normalized_displacement": analyze_snd(a, args),
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
    if h2.get("available") and h2["fit"].get("fitted"):
        pr = h2["fit"]["primary"]
        print("H2 PRIMARY  coef(answer_survival) = %+.4f  se %.4f  p %.4g  %s"
              % (pr["coef"], pr["se"], pr["p"],
                 "controlled" if h2["primary_endpoint_complete"] else "UNCONTROLLED"))
    elif h2.get("available"):
        print("H2 PRIMARY  UNFITTABLE (%d fork events of %d pairs): %s"
              % (h2["fit"].get("n_events", 0), h2["fit"].get("n_pairs", 0),
                 h2["fit"].get("unfittable_reason")))
        print("            the remaining analyses were still computed and written.")
    if h3.get("available"):
        print("H3 PRIMARY  mean paired d(lambda2) = %+.6f over %d layer-pairs, "
              "Wilcoxon(less) p = %.4g"
              % (h3["mean_paired_difference"], h3["n_layer_pairs"], h3["wilcoxon"]["p"]))
    sec = result["secondary_endpoints"]["h2_fork_hazard_break_vs_style_floor"]
    if sec.get("available") and sec.get("model_fitted"):
        print("H2 SECONDARY (pre-registered, does NOT replace the primary above)  "
              "HR(%s vs %s) = %.4f  p %.4g"
              % (args.break_condition, args.style_floor_upper, sec["hazard_ratio"],
                 sec["p"]))
    elif sec.get("available"):
        print("H2 SECONDARY  UNFITTABLE (%d fork events): %s"
              % (sec["fit"].get("n_events", 0), sec["fit"].get("unfittable_reason")))
    else:
        print("H2 SECONDARY  unavailable: %s" % sec.get("reason"))
    snd = result["secondary_endpoints"]["style_normalized_displacement"]
    if snd.get("available"):
        print("SND is INTERPRETIVE SECONDARY, reported as a range over both style "
              "floors; %d of its floor-wise values are null (denominator CI includes "
              "zero)." % snd["n_undefined"])
    print("H1 is DESCRIPTIVE only (Amendment 1): no test, no direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
