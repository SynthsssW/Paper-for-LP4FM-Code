"""Statistics for the two primary endpoints, laptop-side, stdlib-only.

Covers DESIGN.md §A.9 (drop the Gaussian conditional), §A.9.1 (fork layer is censored
survival data), §A.9.3 (Wilcoxon, log-transformed lambda2, permutation nulls) and §A.10
(Benjamini-Hochberg across the layer-wise family).

Nothing here imports torch. numpy and scipy are used only if importable and only as a
speed/accuracy convenience; every routine has a pure-Python implementation so the whole
analysis runs on a laptop with neither installed.

THE RESAMPLING UNIT IS THE PROBLEM (prereg decision 3, DEVIATIONS Amendment 2).
`bootstrap_ci` and `permutation_null` therefore take a list of PROBLEM IDS and a callable
that REBUILDS the statistic from a problem multiset. They cannot be handed rows or token
states: passing anything that is not a flat sequence of unique scalar problem ids raises.
Resampling token states within a problem gives intervals that are far too narrow, and it
is the single easiest way to make this project produce confident nonsense.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Optional, Sequence

__all__ = [
    "person_period_expand",
    "discrete_time_survival",
    "wilcoxon_paired",
    "bootstrap_ci",
    "permutation_null",
    "benjamini_hochberg",
    "log_lambda2",
    "normal_sf",
]


# --------------------------------------------------------------------------------------
# Small numeric helpers (pure python; no numpy required)
# --------------------------------------------------------------------------------------


def normal_sf(z: float) -> float:
    """Upper-tail standard normal survival function, via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _matmul_xtwx(X, w):
    n, p = len(X), len(X[0])
    out = [[0.0] * p for _ in range(p)]
    for i in range(n):
        xi, wi = X[i], w[i]
        if wi == 0.0:
            continue
        for a in range(p):
            xa = xi[a] * wi
            if xa == 0.0:
                continue
            row = out[a]
            for b in range(a, p):
                row[b] += xa * xi[b]
    for a in range(p):
        for b in range(a):
            out[a][b] = out[b][a]
    return out


def _invert(A):
    """Gauss-Jordan inverse with partial pivoting. Raises on singularity."""
    n = len(A)
    M = [list(map(float, A[i])) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError(
                "design matrix is singular at column %d: a predictor is collinear or "
                "constant (check for a covariate with no variation, or perfect "
                "separation in the survival model)" % col)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f == 0.0:
                continue
            M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


def _matvec(A, v):
    return [sum(a * b for a, b in zip(row, v)) for row in A]


# --------------------------------------------------------------------------------------
# H2: discrete-time survival for fork layer (DESIGN.md §A.9.1)
# --------------------------------------------------------------------------------------


def _fork_fields(rec, layers):
    """Extract (layer, censored, layer_sequence) from a record or a ForkResult."""
    fork = rec.get("fork")
    if fork is not None:
        layer = getattr(fork, "layer", None) if not isinstance(fork, dict) else fork.get("layer")
        cens = getattr(fork, "censored", None) if not isinstance(fork, dict) else fork.get("censored")
        seq = getattr(fork, "layers", None) if not isinstance(fork, dict) else fork.get("layers")
    else:
        if "censored" not in rec:
            raise ValueError(
                "record for problem %r has neither a `fork` ForkResult nor an explicit "
                "`censored` flag. Censoring must be explicit: there is no sentinel layer."
                % rec.get("problem_id"))
        layer = rec.get("fork_layer")
        cens = rec["censored"]
        seq = rec.get("layers")
    if seq is None:
        seq = layers
    if seq is None:
        raise ValueError("no layer sequence available for problem %r" % rec.get("problem_id"))
    seq = tuple(seq)
    if cens:
        if layer is not None:
            raise ValueError(
                "problem %r is marked censored but carries fork_layer=%r. A censored pair "
                "has no event layer." % (rec.get("problem_id"), layer))
    else:
        if layer is None:
            raise ValueError(
                "problem %r is marked uncensored but has fork_layer=None"
                % rec.get("problem_id"))
        if layer not in seq:
            raise ValueError(
                "problem %r: fork layer %r is not in the observed layer sequence"
                % (rec.get("problem_id"), layer))
    return layer, bool(cens), seq


def person_period_expand(records, layers=None):
    """Expand pair-level fork outcomes into person-period rows (DESIGN.md §A.9.1).

    One row per (pair, layer), from the first observed layer up to and including the
    fork layer for an event, or ALL observed layers for a right-censored pair. The
    censored pair contributes every one of its non-fork rows and no fabricated event
    row -- that is the entire reason for using this format rather than regressing on a
    fork-layer number.

    `records` is a sequence of dicts, each with:
        problem_id : str        (the clustering unit for the robust variance)
        fork       : a ling.metrics.ForkResult      -- or --
        fork_layer : int|None  and  censored : bool
        ... any number of covariate fields, referenced by name later.

    Returns a list of dicts: the original record's fields, plus
        layer   : the layer this row is about
        t       : 1-based discrete time index (position in the layer sequence)
        event   : 1 on the fork layer of an uncensored pair, else 0
    """
    out = []
    for i, rec in enumerate(records):
        layer, cens, seq = _fork_fields(rec, layers)
        for t, L in enumerate(seq, start=1):
            row = dict(rec)
            row.pop("fork", None)
            row["pair_index"] = i
            row["layer"] = L
            row["t"] = t
            row["event"] = 1 if (not cens and L == layer) else 0
            out.append(row)
            if not cens and L == layer:
                break
    return out


def discrete_time_survival(
    records,
    layers=None,
    survival_key: str = "answer_survival",
    covariates: Sequence[str] = (),
    baseline: str = "linear",
    cluster_by_problem: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    expected_direction: str = "negative",
    direction_note: Optional[str] = None,
) -> dict:
    """H2's PRIMARY endpoint: the coefficient on answer-survival status.

    Person-period expansion + a binary GLM with a **complementary log-log** link
    (DESIGN.md §A.9.1, prereg/PRIMARY_ENDPOINTS.md H2):

        P(fork at layer t | not yet forked) = 1 - exp(-exp(eta))
        eta = b0 + f(t) + b_surv * answer_survival + b' * surface_dissimilarity

    Fitted by IRLS. Right-censored pairs contribute all their non-fork rows and no
    event row; nothing is dropped and no fork layer is fabricated.

    `survival_key` names the pre-registered answer-survival covariate. `covariates` are
    the surface-dissimilarity CONTROLS (lexical + syntactic). The pre-registered
    endpoint is the coefficient on `survival_key` AFTER those controls; a fit without
    them is a different quantity and callers must say so.

    Pre-committed direction: answer-BREAKING pairs fork earlier, i.e. have the higher
    hazard, so the coefficient on answer-SURVIVAL is expected NEGATIVE. That default is
    H2's PRIMARY endpoint. `expected_direction` ("negative" | "positive") exists because
    DEVIATIONS Amendment 6 adds a pre-registered SECONDARY contrast fitted in this same
    model form -- `nl_break` vs `nl_minimal_orth`, indicator coded 1 for `nl_break`,
    pre-committed POSITIVE (nl_break forks earlier). Setting it changes only which sign
    counts as consistent with the pre-commitment; it does not change the model, and it
    does not make the secondary a primary. `direction_note` overrides the reported text.

    `baseline` is the discrete-time baseline hazard in the layer index t:
        "linear" (default; t), "log" (log t), "dummy" (t indicators, first dropped),
        "none".

    RANDOM INTERCEPT FOR PROBLEM (§A.9.2) IS NOT FITTED. §A.9.3 explicitly says not to
    over-engineer a GLMM at n=100. Instead, with `cluster_by_problem=True` (default),
    standard errors are the cluster-robust sandwich clustered on `problem_id`, which
    addresses the same repeated-measures dependence -- the four realizations of a
    problem are one cluster -- without the convergence risk. Model-based SEs are
    returned alongside as `se_model`. This substitution is a judgment call and must be
    stated in the paper; the Part-B GLMM is where the random intercept belongs.
    """
    if expected_direction not in ("negative", "positive"):
        raise ValueError(
            "expected_direction must be 'negative' or 'positive', got %r" % expected_direction)
    rows = person_period_expand(records, layers)
    if not rows:
        raise ValueError("no person-period rows: `records` was empty")

    terms = ["(intercept)"]
    if baseline == "linear":
        terms.append("t")
    elif baseline == "log":
        terms.append("log_t")
    elif baseline == "dummy":
        ts = sorted({r["t"] for r in rows})
        terms.extend("t=%d" % t for t in ts[1:])
    elif baseline != "none":
        raise ValueError("baseline must be 'linear', 'log', 'dummy' or 'none', got %r" % baseline)
    terms.append(survival_key)
    terms.extend(covariates)

    ts_all = sorted({r["t"] for r in rows}) if baseline == "dummy" else ()

    def design_row(r):
        x = [1.0]
        if baseline == "linear":
            x.append(float(r["t"]))
        elif baseline == "log":
            x.append(math.log(float(r["t"])))
        elif baseline == "dummy":
            x.extend(1.0 if r["t"] == t else 0.0 for t in ts_all[1:])
        for name in [survival_key] + list(covariates):
            if name not in r:
                raise ValueError(
                    "person-period row for problem %r is missing covariate %r"
                    % (r.get("problem_id"), name))
            x.append(float(r[name]))
        return x

    X = [design_row(r) for r in rows]
    y = [float(r["event"]) for r in rows]
    groups = [r.get("problem_id", r.get("pair_index")) for r in rows]
    n, p = len(X), len(X[0])
    if len(terms) != p:
        raise AssertionError("term/column mismatch: %d names, %d columns" % (len(terms), p))
    n_events = int(sum(y))
    if n_events == 0:
        raise ValueError(
            "no fork events at all: every pair is right-censored. The survival "
            "coefficient is not identified.")

    beta = [0.0] * p
    # A sane start for the intercept: overall hazard on the cloglog scale.
    h0 = min(max(n_events / float(n), 1e-6), 1 - 1e-6)
    beta[0] = math.log(-math.log(1.0 - h0))

    converged = False
    dev_prev = None
    deviance = float("nan")
    for _ in range(max_iter):
        eta = [sum(a * b for a, b in zip(xi, beta)) for xi in X]
        w, z, mu_all, dmu_all = [], [], [], []
        for i in range(n):
            e = max(min(eta[i], 20.0), -30.0)
            mu = 1.0 - math.exp(-math.exp(e))
            mu = min(max(mu, 1e-10), 1 - 1e-10)
            dmu = math.exp(e - math.exp(e))
            dmu = max(dmu, 1e-10)
            v = mu * (1.0 - mu)
            wi = dmu * dmu / v
            w.append(wi)
            z.append(e + (y[i] - mu) / dmu)
            mu_all.append(mu)
            dmu_all.append(dmu)
        XtWX = _matmul_xtwx(X, w)
        XtWz = [0.0] * p
        for i in range(n):
            wi, zi, xi = w[i], z[i], X[i]
            for a in range(p):
                XtWz[a] += wi * xi[a] * zi
        beta_new = _matvec(_invert(XtWX), XtWz)
        deviance = -2.0 * sum(
            (y[i] * math.log(mu_all[i]) + (1 - y[i]) * math.log(1 - mu_all[i]))
            for i in range(n))
        beta = beta_new
        if dev_prev is not None and abs(dev_prev - deviance) < tol * (abs(deviance) + 0.1):
            converged = True
            break
        dev_prev = deviance

    # Final quantities at the fitted beta.
    eta = [sum(a * b for a, b in zip(xi, beta)) for xi in X]
    w, score = [], []
    for i in range(n):
        e = max(min(eta[i], 20.0), -30.0)
        mu = min(max(1.0 - math.exp(-math.exp(e)), 1e-10), 1 - 1e-10)
        dmu = max(math.exp(e - math.exp(e)), 1e-10)
        wi = dmu * dmu / (mu * (1.0 - mu))
        w.append(wi)
        s = (y[i] - mu) * dmu / (mu * (1.0 - mu))
        score.append([s * v for v in X[i]])
    bread = _invert(_matmul_xtwx(X, w))
    se_model = [math.sqrt(max(bread[a][a], 0.0)) for a in range(p)]

    if cluster_by_problem:
        acc = {}
        for i in range(n):
            g = groups[i]
            cur = acc.setdefault(g, [0.0] * p)
            for a in range(p):
                cur[a] += score[i][a]
        meat = [[0.0] * p for _ in range(p)]
        for u in acc.values():
            for a in range(p):
                for b in range(p):
                    meat[a][b] += u[a] * u[b]
        G = len(acc)
        scale = (G / float(G - 1)) if G > 1 else 1.0
        mid = [[meat[a][b] * scale for b in range(p)] for a in range(p)]
        tmp = [[sum(bread[a][c] * mid[c][b] for c in range(p)) for b in range(p)] for a in range(p)]
        V = [[sum(tmp[a][c] * bread[c][b] for c in range(p)) for b in range(p)] for a in range(p)]
        se = [math.sqrt(max(V[a][a], 0.0)) for a in range(p)]
        n_clusters = G
    else:
        se = list(se_model)
        n_clusters = None

    coefs = {}
    for a, name in enumerate(terms):
        zst = beta[a] / se[a] if se[a] > 0 else float("nan")
        coefs[name] = {
            "coef": beta[a],
            "se": se[a],
            "se_model": se_model[a],
            "z": zst,
            "p": 2.0 * normal_sf(abs(zst)) if se[a] > 0 else float("nan"),
        }

    prim = dict(coefs[survival_key])
    prim["term"] = survival_key
    prim["hazard_ratio"] = math.exp(beta[terms.index(survival_key)])
    prim["direction_precommitted"] = direction_note or (
        "negative (answer-breaking pairs fork earlier)" if expected_direction == "negative"
        else "positive (higher hazard, i.e. forks earlier)")
    prim["expected_direction"] = expected_direction
    prim["consistent_with_precommitted_direction"] = (
        prim["coef"] < 0 if expected_direction == "negative" else prim["coef"] > 0)
    hr_lo = math.exp(prim["coef"] - 1.959963985 * prim["se"]) if prim["se"] > 0 else float("nan")
    hr_hi = math.exp(prim["coef"] + 1.959963985 * prim["se"]) if prim["se"] > 0 else float("nan")
    prim["hazard_ratio_ci95"] = [hr_lo, hr_hi]

    return {
        "fitted": True,
        "link": "cloglog",
        "model": "discrete-time survival, person-period expansion",
        "terms": terms,
        "coefficients": coefs,
        "primary": prim,
        "n_rows": n,
        "n_pairs": len(records),
        "n_events": n_events,
        "n_censored": len(records) - n_events,
        "n_problem_clusters": n_clusters,
        "se_type": "cluster-robust (problem)" if cluster_by_problem else "model-based",
        "random_intercept_for_problem": False,
        "baseline_hazard": baseline,
        "converged": converged,
        "deviance": deviance,
    }


# --------------------------------------------------------------------------------------
# H3: Wilcoxon signed-rank, paired across LAYER-PAIRS
# --------------------------------------------------------------------------------------


def _signed_rank_null_cdf(n):
    """Exact distribution of W+ under H0 for n untied nonzero differences."""
    counts = [1] + [0] * (n * (n + 1) // 2)
    for r in range(1, n + 1):
        for s in range(len(counts) - 1, r - 1, -1):
            counts[s] += counts[s - r]
    total = float(2 ** n)
    return counts, total


def wilcoxon_paired(x, y=None, alternative: str = "two-sided", exact_max_n: int = 25) -> dict:
    """Wilcoxon signed-rank test on paired observations.

    For H3 the pairing unit is the LAYER-PAIR (n = 47 for L = 48), not the problem
    (DEVIATIONS Amendment 2): lambda2 is a pooled statistic with one value per condition
    per layer-pair, so there is no per-problem lambda2 to pair. Callers pass the 47
    per-layer-pair lambda2 values for each condition; nothing here knows or cares what
    the rows mean, so the caller is responsible for not passing problems.

    Zero differences are dropped (the standard Wilcoxon reduction), and the count is
    reported as `n_zero`. Exact null for n <= exact_max_n and no ties; otherwise the
    normal approximation with tie and continuity correction.

    `alternative`: "two-sided", "less" (median difference < 0 -- H3's pre-committed
    direction for formal minus nl_base), or "greater".
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    if y is None:
        d = [float(v) for v in x]
    else:
        x = list(x)
        y = list(y)
        if len(x) != len(y):
            raise ValueError("paired samples must have equal length (%d vs %d)" % (len(x), len(y)))
        d = [float(a) - float(b) for a, b in zip(x, y)]
    n_total = len(d)
    nz = [v for v in d if v != 0.0]
    n = len(nz)
    n_zero = n_total - n
    if n == 0:
        return {"statistic": 0.0, "p": 1.0, "n": 0, "n_zero": n_zero,
                "method": "degenerate (all differences zero)",
                "alternative": alternative, "mean_difference": 0.0,
                "median_difference": 0.0}

    absd = [abs(v) for v in nz]
    order = sorted(range(n), key=lambda i: absd[i])
    ranks = [0.0] * n
    i = 0
    has_ties = False
    tie_groups = []
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        if j > i:
            has_ties = True
        tie_groups.append(j - i + 1)
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)

    if n <= exact_max_n and not has_ties:
        counts, total = _signed_rank_null_cdf(n)
        cum = [0.0] * (len(counts) + 1)
        acc = 0.0
        for s, c in enumerate(counts):
            acc += c
            cum[s] = acc
        w = int(round(w_plus))
        p_le = cum[w] / total
        p_ge = 1.0 - (cum[w - 1] / total if w > 0 else 0.0)
        if alternative == "two-sided":
            pval = min(1.0, 2.0 * min(p_le, p_ge))
        elif alternative == "less":
            # median difference < 0  <=>  W+ small
            pval = p_le
        else:
            pval = p_ge
        method = "exact"
    else:
        mu = n * (n + 1) / 4.0
        tie_corr = sum(t ** 3 - t for t in tie_groups)
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr / 48.0
        sigma = math.sqrt(sigma2) if sigma2 > 0 else 0.0
        if sigma == 0.0:
            pval = 1.0
        else:
            if alternative == "two-sided":
                z = (abs(w_plus - mu) - 0.5) / sigma
                pval = min(1.0, 2.0 * normal_sf(max(z, 0.0)))
            elif alternative == "less":
                # median difference < 0  <=>  W+ small  <=>  P(W+ <= w) = Phi(z)
                z = (w_plus - mu + 0.5) / sigma
                pval = normal_sf(-z)
            else:
                z = (w_plus - mu - 0.5) / sigma
                pval = normal_sf(z)
        method = "normal approximation (tie + continuity corrected)"

    sd = sorted(d)
    med = sd[n_total // 2] if n_total % 2 else 0.5 * (sd[n_total // 2 - 1] + sd[n_total // 2])
    return {
        "statistic": min(w_plus, w_minus),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p": pval,
        "n": n,
        "n_zero": n_zero,
        "n_total": n_total,
        "method": method,
        "alternative": alternative,
        "mean_difference": sum(d) / n_total,
        "median_difference": med,
    }


# --------------------------------------------------------------------------------------
# Resampling -- PROBLEMS ONLY
# --------------------------------------------------------------------------------------


class ResamplingUnitError(TypeError):
    """The caller tried to resample something other than problems.

    prereg decision 3 / DESIGN.md §A.9.2: token states within a problem are dependent,
    so resampling them produces confidence intervals that are far too narrow. This API
    accepts a flat sequence of unique scalar problem ids and nothing else, so that
    handing it rows, token states, or (problem, anchor) pairs fails immediately instead
    of silently producing a tighter interval than the data support.
    """


def _validate_problem_ids(problem_ids, arg="problem_ids"):
    if isinstance(problem_ids, (str, bytes, dict)):
        raise ResamplingUnitError(
            "%s must be a sequence of problem ids, not %s" % (arg, type(problem_ids).__name__))
    ids = list(problem_ids)
    for v in ids:
        if not isinstance(v, (str, int)) or isinstance(v, bool):
            raise ResamplingUnitError(
                "%s must contain scalar problem ids (str/int); got %r. Rows, token "
                "states and (problem, anchor) tuples are NOT resampling units "
                "(prereg decision 3)." % (arg, type(v).__name__))
    if len(set(ids)) != len(ids):
        raise ResamplingUnitError(
            "%s contains duplicates. Pass the DISTINCT problems in the study; the "
            "bootstrap draws the multiset itself." % arg)
    if len(ids) < 2:
        raise ValueError("%s needs at least 2 problems, got %d" % (arg, len(ids)))
    return ids


def bootstrap_ci(
    problem_ids: Sequence,
    statistic: Callable[[Sequence], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI, resampling PROBLEMS with replacement.

    `statistic` is a callable that REBUILDS the quantity from a problem multiset:

        statistic(problems: Sequence[problem_id]) -> float

    For lambda2 this means rebuilding the pooled cluster-transition graph from exactly
    those problem draws and re-running the eigensolve (DEVIATIONS Amendment 2), e.g.

        lambda(ps): metrics.lambda2(a, p, "formal", problems=ps)
                    - metrics.lambda2(a, p, "nl_base", problems=ps)

    The callable is handed a LIST WITH REPEATS. That is deliberate: a problem drawn
    twice must contribute its token states twice to the rebuilt graph. Any statistic
    that ignores repeats -- e.g. one that deduplicates, or that closes over a
    precomputed pooled value -- is not bootstrapping anything, and the resulting
    interval is meaningless.

    Draws that raise are counted in `n_failed` and skipped (a resample can be degenerate,
    e.g. every drawn problem visiting one cluster); if more than half fail, this raises.
    """
    ids = _validate_problem_ids(problem_ids)
    if not callable(statistic):
        raise TypeError("statistic must be a callable taking a problem multiset")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = random.Random(seed)
    n = len(ids)
    point = float(statistic(list(ids)))
    draws = []
    n_failed = 0
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(float(statistic(sample)))
        except Exception:
            n_failed += 1
    if len(draws) < n_boot / 2.0:
        raise RuntimeError(
            "%d of %d bootstrap resamples failed; the statistic is not computable on "
            "resampled problem sets" % (n_failed, n_boot))
    draws.sort()
    m = len(draws)

    def q(pr):
        if m == 1:
            return draws[0]
        pos = pr * (m - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, m - 1)
        return draws[lo] + (pos - lo) * (draws[hi] - draws[lo])

    mean = sum(draws) / m
    return {
        "point": point,
        "ci_low": q(alpha / 2.0),
        "ci_high": q(1.0 - alpha / 2.0),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_effective": m,
        "n_failed": n_failed,
        "n_problems": n,
        "boot_mean": mean,
        "boot_bias": mean - point,
        "resampling_unit": "problem",
        "seed": seed,
    }


def permutation_null(
    problem_ids: Sequence,
    statistic: Callable[[frozenset], float],
    n_perm: int = 1000,
    alternative: str = "two-sided",
    seed: int = 0,
    observed: Optional[float] = None,
) -> dict:
    """Permutation null over PROBLEMS: the two condition labels are swapped per problem.

    `statistic` is a callable that REBUILDS the quantity given the set of problems whose
    two condition labels are exchanged:

        statistic(swapped: frozenset[problem_id]) -> float

    `statistic(frozenset())` is the observed statistic. `ling.metrics.lambda2` and
    `path_entropy` accept a mapping problem_id -> condition precisely so this is a
    one-liner:

        def stat(swapped):
            cond_a = {p: ("nl_base" if p in swapped else "formal") for p in ids}
            cond_b = {p: ("formal" if p in swapped else "nl_base") for p in ids}
            ...rebuild the pooled graph and take the difference...

    The unit permuted is the PROBLEM (prereg decision 3): the four realizations of a
    problem move together, and no token state is ever exchanged on its own.

    p-value uses the (1 + #extreme) / (1 + n_perm) convention, so it is never 0.
    """
    ids = _validate_problem_ids(problem_ids)
    if not callable(statistic):
        raise TypeError("statistic must be a callable taking a frozenset of swapped problems")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    rng = random.Random(seed)
    obs = float(statistic(frozenset())) if observed is None else float(observed)
    null = []
    n_failed = 0
    for _ in range(n_perm):
        swapped = frozenset(p for p in ids if rng.random() < 0.5)
        try:
            null.append(float(statistic(swapped)))
        except Exception:
            n_failed += 1
    if not null:
        raise RuntimeError("every permutation draw failed")
    if alternative == "less":
        extreme = sum(1 for v in null if v <= obs)
    elif alternative == "greater":
        extreme = sum(1 for v in null if v >= obs)
    else:
        extreme = sum(1 for v in null if abs(v) >= abs(obs))
    return {
        "observed": obs,
        "p": (1.0 + extreme) / (1.0 + len(null)),
        "n_perm": len(null),
        "n_failed": n_failed,
        "alternative": alternative,
        "null_mean": sum(null) / len(null),
        "null_sd": math.sqrt(
            sum((v - sum(null) / len(null)) ** 2 for v in null) / max(len(null) - 1, 1)),
        "permutation_unit": "problem (condition labels swapped within a problem)",
        "seed": seed,
    }


# --------------------------------------------------------------------------------------
# Multiplicity and transforms
# --------------------------------------------------------------------------------------


def benjamini_hochberg(pvals: Iterable[float], q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR across the layer-wise family (prereg decision 4, §A.10).

    Applies to anything reported PER LAYER. The primary endpoints are NOT corrected
    against each other. Returns rejection flags and BH-adjusted p-values in the input
    order.
    """
    p = [float(v) for v in pvals]
    m = len(p)
    if m == 0:
        return {"q": q, "n": 0, "rejected": [], "adjusted": [], "n_rejected": 0,
                "threshold": None}
    for v in p:
        if not (0.0 <= v <= 1.0):
            raise ValueError("p-value out of [0, 1]: %r" % v)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        adj[i] = min(1.0, prev)
    rejected = [adj[i] <= q for i in range(m)]
    crit = [p[order[r - 1]] for r in range(1, m + 1) if p[order[r - 1]] <= q * r / m]
    return {
        "q": q,
        "n": m,
        "rejected": rejected,
        "adjusted": adj,
        "n_rejected": sum(1 for r in rejected if r),
        "threshold": max(crit) if crit else None,
        "family": "layer-wise",
    }


def log_lambda2(values: Iterable[float], zero_policy: str = "mask") -> dict:
    """Log-transform lambda2 before regression (prereg decision 6, §A.9.2/§A.9.3).

    lambda2 is exactly 0 for a disconnected pooled graph, which is a real value, not a
    missing one, and log(0) is not. `zero_policy`:

        "mask"  (default) -- exclude the zeros, return a mask, and REPORT how many.
                             Honest, but the exclusion is informative-missingness and
                             must be stated wherever the regression is reported.
        "raise"           -- refuse; use when a zero indicates a plumbing bug.
        "floor"           -- replace 0 with half the smallest positive value observed.
                             A convenience, not a principled imputation.

    Natural log.
    """
    vals = [float(v) for v in values]
    for v in vals:
        if v < 0:
            raise ValueError("lambda2 cannot be negative: %r" % v)
    n_zero = sum(1 for v in vals if v == 0.0)
    if zero_policy == "raise":
        if n_zero:
            raise ValueError(
                "%d of %d lambda2 values are exactly 0 (disconnected pooled graph); "
                "log is undefined" % (n_zero, len(vals)))
        out, mask = [math.log(v) for v in vals], [True] * len(vals)
    elif zero_policy == "mask":
        mask = [v > 0.0 for v in vals]
        out = [math.log(v) if v > 0.0 else None for v in vals]
    elif zero_policy == "floor":
        pos = [v for v in vals if v > 0.0]
        if not pos:
            raise ValueError("all lambda2 values are 0; nothing to floor against")
        floor = min(pos) / 2.0
        out = [math.log(v if v > 0.0 else floor) for v in vals]
        mask = [True] * len(vals)
    else:
        raise ValueError("zero_policy must be 'mask', 'raise' or 'floor', got %r" % zero_policy)
    return {
        "log_values": out,
        "mask": mask,
        "n": len(vals),
        "n_zero": n_zero,
        "zero_policy": zero_policy,
        "note": "prereg decision 6: lambda2 is log-transformed before any regression",
    }
