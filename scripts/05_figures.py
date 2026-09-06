#!/usr/bin/env python3
"""Publication figures for the LP4FM 2026 workshop paper (vector PDF + PNG previews).

Inputs
  results/analysis.json                  written by scripts/04_analyze.py
  results/<run_id>/assignments.jsonl     the pod's only output (configs/assignments_schema.md)

Outputs
  paper/figs/fig1.{pdf,png}   H3 PRIMARY  lambda2 by layer-pair per condition (a) and the
                              pre-registered paired difference formal - nl_base with its
                              bootstrap CI (b).  Amendment 2: pairing unit = layer-pair.
  paper/figs/fig2.{pdf,png}   H2 PRIMARY  discrete-time survival curves for fork layer,
                              answer-preserving vs answer-breaking, censoring marked.
  paper/figs/fig3.{pdf,png}   methods     eigengap by layer with the selected k and the
                              flat-spectrum layers flagged.

Usage
  python scripts/05_figures.py                      # real results; fails loudly if absent
  python scripts/05_figures.py --synthetic          # plausible fake data, same schema,
                                                    # for checking layout before the run
  python scripts/05_figures.py --run-id <id>        # pick the assignments file explicitly

Design rules (validated palette; see the task brief): fixed colour per condition identity,
linestyle as secondary encoding, direct labels instead of a legend box, one y-axis per
panel, serif fonts with fonttype 42, recessive grid, no top/right spines, text in ink.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------------------
# Fixed visual encodings. Assigned by condition IDENTITY, never by plotting order.
# --------------------------------------------------------------------------------------

CONDITIONS = ("nl_base", "nl_para", "formal", "nl_break")

COLOR = {
    "nl_base": "#2a78d6",   # blue
    "nl_para": "#eb6834",   # orange
    "formal": "#1baf7a",    # aqua  (2.74:1 on white -> direct labels are mandatory)
    "nl_break": "#4a3aa7",  # violet
}
LINESTYLE = {
    "nl_base": "solid",
    "nl_para": "dashed",
    "formal": "dashdot",
    "nl_break": "dotted",
}
PRETTY = {
    "nl_base": "NL base",
    "nl_para": "NL paraphrase",
    "formal": "formal",
    "nl_break": "NL answer-breaking",
}

# Fig 2 groups (H2): answer-preserving vs answer-breaking realizations vs nl_base.
GROUP_COLOR = {"preserving": "#2a78d6", "breaking": "#eb6834"}
GROUP_LS = {"preserving": "solid", "breaking": "dashed"}
GROUP_LABEL = {"preserving": "answer-preserving", "breaking": "answer-breaking"}

INK = "#1a1a1a"
INK_SOFT = "#555555"
GRID = "#d9d9d9"
BAND = "#9a9a9a"
FLAT_SHADE = "#e8e8e8"

TEXT_WIDTH_IN = 5.5
LINE_W = 2.0        # px at 100 dpi == pt; 2px lines
MARKER_PT = 8.0     # >= 8px markers


def _setup_matplotlib():
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is not installed. Install it with `pip install matplotlib` "
            "(check free disk first) and re-run.")
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "lines.linewidth": LINE_W,
        "lines.markersize": MARKER_PT,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
    })
    return matplotlib


# --------------------------------------------------------------------------------------
# Data loading: real
# --------------------------------------------------------------------------------------


def _find_assignments(run_id):
    if run_id:
        p = os.path.join(_REPO, "results", run_id, "assignments.jsonl")
        if not os.path.exists(p):
            raise SystemExit("no assignments file at %s" % p)
        return p
    cands = sorted(glob.glob(os.path.join(_REPO, "results", "*", "assignments.jsonl")))
    if not cands:
        raise SystemExit(
            "no results/<run_id>/assignments.jsonl found. The GPU run has not produced "
            "assignments yet. Pass --synthetic to render layout with fake data.")
    if len(cands) > 1:
        raise SystemExit(
            "several runs found; pass --run-id to choose one:\n  " + "\n  ".join(cands))
    return cands[0]


def _read_assignments(path):
    layers = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                layers.append(json.loads(line))
    if not layers:
        raise SystemExit("%s is empty" % path)
    layers.sort(key=lambda d: d["layer"])
    key0 = [(r["problem_id"], r["condition"], r["anchor"]) for r in layers[0]["rows"]]
    for d in layers[1:]:
        key = [(r["problem_id"], r["condition"], r["anchor"]) for r in d["rows"]]
        if key != key0:
            raise SystemExit(
                "row order differs between layer %d and layer %d; the schema invariant "
                "(stable row order) is violated" % (layers[0]["layer"], d["layer"]))
    for d in layers:
        ks = {r.get("k", d["k"]) for r in d["rows"]}
        if ks and ks != {d["k"]}:
            raise SystemExit("layer %d has more than one k across rows: %s"
                             % (d["layer"], sorted(ks)))
    return layers, key0


def fork_layers_from_assignments(layers, key0, ref, other):
    """First layer at which the two conditions' anchor-cluster SETS are disjoint.

    Mirrors src/ling/metrics.fork_layer (set-level disjointness over the problem's anchors).
    Returns {problem_id: layer or None}; None == right-censored (never forks).
    """
    idx = {}
    for j, (pid, cond, _anchor) in enumerate(key0):
        idx.setdefault((pid, cond), []).append(j)
    problems = sorted({pid for pid, cond, _ in key0 if cond == ref})
    out = {}
    for pid in problems:
        i1 = idx.get((pid, ref))
        i2 = idx.get((pid, other))
        if i1 is None or i2 is None:
            raise SystemExit("problem %r lacks rows for %r or %r" % (pid, ref, other))
        fork = None
        for d in layers:
            cl = [r["cluster"] for r in d["rows"]]
            s1 = {cl[j] for j in i1}
            s2 = {cl[j] for j in i2}
            if s1.isdisjoint(s2):
                fork = d["layer"]
                break
        out[pid] = fork
    return out


def load_real(analysis_path, assignments_path):
    if not os.path.exists(analysis_path):
        raise SystemExit(
            "%s does not exist. Run scripts/04_analyze.py first, or pass --synthetic to "
            "render layout with fake data." % analysis_path)
    with open(analysis_path, "r", encoding="utf-8") as fh:
        analysis = json.load(fh)
    prim = analysis.get("primary_endpoints", {})
    h3 = prim.get("h3", {})
    h2 = prim.get("h2", {})
    if not h3.get("available"):
        raise SystemExit("H3 is not available in %s: %s"
                         % (analysis_path, h3.get("reason", "unknown reason")))
    expl = analysis.get("exploratory", {})
    l2c = expl.get("lambda2_by_condition", {}).get("by_condition")
    if not l2c:
        # Fall back to the two PRIMARY series if the exploratory block is missing.
        l2c = {"formal": h3["lambda2_formal"], "nl_base": h3["lambda2_base"]}
    layer_pairs = list(h3["layer_pairs"])

    layers, key0 = _read_assignments(assignments_path)
    ref = "nl_base"
    breaking = set(h2.get("answer_breaking_conditions") or ["nl_break"])
    conds_present = sorted({c for _, c, _ in key0})
    groups = {"preserving": {}, "breaking": {}}
    for cond in conds_present:
        if cond == ref:
            continue
        forks = fork_layers_from_assignments(layers, key0, ref, cond)
        g = "breaking" if cond in breaking else "preserving"
        for pid, f in forks.items():
            groups[g][(pid, cond)] = f

    return {
        "synthetic": False,
        "layer_pairs": layer_pairs,
        "lambda2": {c: list(v) for c, v in l2c.items()},
        "h3": {
            "mean": h3["mean_paired_difference"],
            "diff": list(h3["paired_differences"]),
            "ci": h3.get("bootstrap_ci"),
            # Not emitted by 04_analyze.py today; drawn when present.
            "ci_per_pair": h3.get("bootstrap_ci_per_layer_pair"),
            "wilcoxon_p": (h3.get("wilcoxon") or {}).get("p"),
            "n_pairs": h3.get("n_layer_pairs", len(layer_pairs)),
        },
        "layers": [d["layer"] for d in layers],
        "fork": {g: list(v.values()) for g, v in groups.items()},
        "eigengap": [d["eigengap"] for d in layers],
        "eigengap_flat": [bool(d.get("eigengap_flat", False)) for d in layers],
        "k": [d["k"] for d in layers],
    }


# --------------------------------------------------------------------------------------
# Data loading: synthetic (schema-matched, for layout checks before the GPU run)
# --------------------------------------------------------------------------------------


def synthetic(seed=0, n_layers=48, n_problems=100):
    rng = random.Random(seed)
    layers = list(range(n_layers))
    pairs = list(range(n_layers - 1))

    def smooth(base, amp, phase, noise):
        return [max(0.0, base + amp * math.sin((p + phase) / 7.0)
                    + rng.gauss(0, noise)) for p in pairs]

    lam = {
        "nl_base": smooth(0.085, 0.030, 3.0, 0.006),
        "nl_para": smooth(0.080, 0.028, 3.5, 0.006),
        "formal": smooth(0.062, 0.024, 2.2, 0.006),
        "nl_break": smooth(0.097, 0.032, 4.2, 0.007),
    }
    # Three disconnected pooled graphs (lambda2 == 0 is a real value, not missing).
    for p in (0, 1):
        lam["formal"][p] = 0.0
    diff = [f - b for f, b in zip(lam["formal"], lam["nl_base"])]
    mean = sum(diff) / len(diff)
    half = [0.008 + 0.004 * abs(math.sin(p / 5.0)) for p in pairs]
    ci_pp = {"ci_low": [d - h for d, h in zip(diff, half)],
             "ci_high": [d + h for d, h in zip(diff, half)]}

    def draw_forks(hazard0, growth, n):
        out = []
        for _ in range(n):
            f = None
            for L in layers:
                h = min(0.95, hazard0 * (1.0 + growth * L))
                if rng.random() < h:
                    f = L
                    break
            out.append(f)
        return out

    fork = {
        "preserving": draw_forks(0.010, 0.09, 2 * n_problems),  # nl_para + formal
        "breaking": draw_forks(0.030, 0.12, n_problems),        # nl_break
    }

    eig = []
    for L in layers:
        v = 0.012 + 0.045 * math.exp(-((L - 21) / 8.0) ** 2) + rng.gauss(0, 0.003)
        eig.append(max(0.002, v))
    flat = [e < 0.014 for e in eig]
    k = []
    for L in layers:
        k.append(int(round(8 + 6 * math.exp(-((L - 21) / 10.0) ** 2))) if not flat[L]
                 else 6)

    return {
        "synthetic": True,
        "layer_pairs": pairs,
        "lambda2": lam,
        "h3": {
            "mean": mean,
            "diff": diff,
            "ci": {"point": mean, "ci_low": mean - 0.0045, "ci_high": mean - 0.0004,
                   "alpha": 0.05, "n_boot": 200},
            "ci_per_pair": ci_pp,
            "wilcoxon_p": 0.0031,
            "n_pairs": len(pairs),
        },
        "layers": layers,
        "fork": fork,
        "eigengap": eig,
        "eigengap_flat": flat,
        "k": k,
    }


# --------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------


def _repel(ys, min_gap):
    """Push label y-positions apart (in data units) so none are closer than min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    adj = list(ys)
    for _ in range(50):
        moved = False
        for a, b in zip(order, order[1:]):
            gap = adj[b] - adj[a]
            if gap < min_gap:
                shift = (min_gap - gap) / 2.0
                adj[a] -= shift
                adj[b] += shift
                moved = True
        if not moved:
            break
    return adj


def direct_labels(ax, entries, x_pad=0.6, min_gap_frac=0.075):
    """Direct-label each series at its right end. entries = [(x_end, y_end, text)]."""
    if not entries:
        return
    lo, hi = ax.get_ylim()
    min_gap = min_gap_frac * (hi - lo)
    ys = [e[1] for e in entries]
    adj = _repel(ys, min_gap)
    for (x, y, text), ya in zip(entries, adj):
        if abs(ya - y) > 0.25 * min_gap:
            ax.plot([x, x + x_pad * 0.7], [y, ya], color=INK_SOFT, linewidth=0.5,
                    linestyle="solid", solid_capstyle="round", clip_on=False, zorder=3)
        ax.text(x + x_pad, ya, text, color=INK, fontsize=8, ha="left", va="center",
                clip_on=False, zorder=4)
    ymin = min(min(adj) - 0.6 * min_gap, lo)
    ymax = max(max(adj) + 0.6 * min_gap, hi)
    ax.set_ylim(ymin, ymax)


def _stamp_synthetic(fig):
    fig.text(0.5, 0.5, "SYNTHETIC DATA", ha="center", va="center", fontsize=28,
             color="#000000", alpha=0.08, rotation=30, zorder=0)


def _ci_bounds(ci):
    if not ci:
        return None
    if "ci_low" in ci and "ci_high" in ci:
        return ci["ci_low"], ci["ci_high"]
    if "low" in ci and "high" in ci:
        return ci["low"], ci["high"]
    return None


def _ticks(last, step=10):
    """Tick every `step`, plus the last index unless it would collide with a tick."""
    t = list(range(0, last + 1, step))
    if last - t[-1] >= 3:
        t.append(last)
    elif last != t[-1]:
        t[-1] = last
    return t


def _corner(ax, xs, ys, x_frac=0.55, y_frac=0.16):
    """'top' or 'bottom': the left-side corner with fewer data points in it."""
    lo, hi = ax.get_ylim()
    xcut = xs[0] + x_frac * (xs[-1] - xs[0])
    top = sum(1 for x, y in zip(xs, ys) if x <= xcut and y >= hi - y_frac * (hi - lo))
    bot = sum(1 for x, y in zip(xs, ys) if x <= xcut and y <= lo + y_frac * (hi - lo))
    return "top" if top <= bot else "bottom"


def _fmt(v, nd=4):
    return ("%%.%df" % nd) % v


# --------------------------------------------------------------------------------------
# Figure 1 -- H3 primary
# --------------------------------------------------------------------------------------


def fig1(data, out_dir, plt):
    pairs = data["layer_pairs"]
    lam = data["lambda2"]
    h3 = data["h3"]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, sharex=True, figsize=(TEXT_WIDTH_IN, 4.6),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18})

    # (a) lambda2 by layer-pair, one line per condition.
    entries = []
    for cond in CONDITIONS:
        if cond not in lam:
            continue
        ys = lam[cond]
        ax_a.plot(pairs, ys, color=COLOR[cond], linestyle=LINESTYLE[cond],
                  linewidth=LINE_W, zorder=2)
        entries.append((pairs[-1], ys[-1], PRETTY[cond]))
    ax_a.set_ylabel(r"$\lambda_2$ (pooled, cosine affinity)")
    ax_a.set_ylim(bottom=0)
    ax_a.margins(y=0.08)
    direct_labels(ax_a, entries)
    ax_a.text(0.0, 1.02, "(a) algebraic connectivity by layer-pair, per condition",
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=9, color=INK)

    # (b) paired difference formal - nl_base: the PRIMARY endpoint.
    diff = h3["diff"]
    ax_b.axhline(0.0, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    pp = h3.get("ci_per_pair")
    band_label = None
    if pp:
        lo, hi = _ci_bounds(pp)
        ax_b.fill_between(pairs, lo, hi, color=BAND, alpha=0.30, linewidth=0,
                          zorder=1)
        band_label = True
    ax_b.plot(pairs, diff, color=INK, linestyle="solid", linewidth=LINE_W, zorder=3)

    mean = h3["mean"]
    ci = _ci_bounds(h3.get("ci"))
    ax_b.axhline(mean, color=COLOR["formal"], linewidth=1.2, linestyle="dashdot",
                 zorder=2)
    if ci is not None:
        ax_b.axhspan(ci[0], ci[1], color=COLOR["formal"], alpha=0.16, linewidth=0,
                     zorder=1)

    ax_b.set_ylabel(r"$\Delta\lambda_2$  (formal $-$ NL base)")
    ax_b.set_xlabel("layer-pair index  $p$ = (block $p$, block $p{+}1$)")
    ax_b.margins(y=0.15)
    ax_b.set_xlim(pairs[0], pairs[-1])
    ax_b.set_xticks(_ticks(pairs[-1]))

    # Direct labels on the right for the difference line and the mean line.
    b_entries = [(pairs[-1], diff[-1], "per-pair difference"),
                 (pairs[-1], mean, "mean (primary)")]
    direct_labels(ax_b, b_entries)

    ci_txt = ""
    if ci is not None:
        ci_txt = "  95%% CI [%s, %s]" % (_fmt(ci[0]), _fmt(ci[1]))
    ptxt = ""
    if h3.get("wilcoxon_p") is not None:
        ptxt = ",  Wilcoxon (less) $p$ = %.2g" % h3["wilcoxon_p"]
    verdict = "consistent with H3 (negative)" if mean < 0 else "NOT negative; H3 not supported"
    ann = (r"mean paired $\Delta\lambda_2$ = %s%s" % (_fmt(mean), ci_txt)
           + "\n$n$ = %d layer-pairs%s;  %s" % (h3["n_pairs"], ptxt, verdict))
    # Put the annotation in whichever left corner has the fewest data points, and pad
    # that side of the y-range so the box never sits on the line or the zero reference.
    corner = _corner(ax_b, pairs, diff)
    ylo, yhi = ax_b.get_ylim()
    if corner == "top":
        ax_b.set_ylim(ylo, yhi + 0.28 * (yhi - ylo))
    else:
        ax_b.set_ylim(ylo - 0.28 * (yhi - ylo), yhi)
    if corner == "top":
        ax_b.text(0.01, 0.97, ann, transform=ax_b.transAxes, ha="left", va="top",
                  fontsize=8, color=INK, zorder=5,
                  bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                            edgecolor="none", alpha=0.85))
    else:
        ax_b.text(0.01, 0.03, ann, transform=ax_b.transAxes, ha="left", va="bottom",
                  fontsize=8, color=INK, zorder=5,
                  bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                            edgecolor="none", alpha=0.85))
    if band_label:
        title_b = "(b) PRIMARY: paired difference; band = 95% bootstrap CI, problems resampled"
    else:
        title_b = "(b) PRIMARY: paired difference; shaded = 95% bootstrap CI of the mean"
    ax_b.text(0.0, 1.02, title_b, transform=ax_b.transAxes, ha="left", va="bottom",
              fontsize=9, color=INK)

    for ax in (ax_a, ax_b):
        ax.tick_params(length=2.5, width=0.6)
    fig.subplots_adjust(left=0.13, right=0.78, top=0.95, bottom=0.11)
    if data["synthetic"]:
        _stamp_synthetic(fig)
    _save(fig, out_dir, "fig1")


# --------------------------------------------------------------------------------------
# Figure 2 -- H2 primary: discrete-time survival of fork layer
# --------------------------------------------------------------------------------------


def _survival(forks, layers):
    """S(L) = fraction of pairs that have NOT forked at or before layer L (discrete)."""
    n = len(forks)
    xs, ys = [], []
    for L in layers:
        forked = sum(1 for f in forks if f is not None and f <= L)
        xs.append(L)
        ys.append((n - forked) / float(n) if n else float("nan"))
    return xs, ys


def fig2(data, out_dir, plt):
    layers = data["layers"]
    fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH_IN, 3.0))
    entries = []
    censor_rows = []
    for g in ("preserving", "breaking"):
        forks = data["fork"].get(g, [])
        if not forks:
            continue
        xs, ys = _survival(forks, layers)
        n = len(forks)
        n_cens = sum(1 for f in forks if f is None)
        # Step curve: discrete time, value holds until the next layer.
        xs_plot = [layers[0] - 1] + xs
        ys_plot = [1.0] + ys
        ax.plot(xs_plot, ys_plot, drawstyle="steps-post", color=GROUP_COLOR[g],
                linestyle=GROUP_LS[g], linewidth=LINE_W, zorder=2)
        # Censoring: every censored pair leaves the risk set at the last layer. Draw one
        # tick per group at (last layer, S(last)) and state the count in the label.
        if n_cens:
            ax.plot([xs[-1]], [ys[-1]], marker="|", markersize=MARKER_PT + 2,
                    markeredgewidth=1.6, color=GROUP_COLOR[g], linestyle="none",
                    zorder=4, clip_on=False)
        entries.append((xs[-1], ys[-1],
                        "%s\n$n$ = %d, censored %d" % (GROUP_LABEL[g], n, n_cens)))
        censor_rows.append((g, n, n_cens))

    ax.set_xlabel("layer  $L$  (output of block $L$)")
    ax.set_ylabel("$S(L)$ = P(fork layer $> L$)")
    ax.set_xlim(layers[0] - 1, layers[-1])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(_ticks(layers[-1]))
    direct_labels(ax, entries, min_gap_frac=0.16)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    ax.text(0.0, 1.03,
            "fork layer, discrete-time survival;  '|' = right-censored at depth "
            "(never forked; kept, not dropped)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK)
    ax.tick_params(length=2.5, width=0.6)
    fig.subplots_adjust(left=0.13, right=0.72, top=0.90, bottom=0.17)
    if data["synthetic"]:
        _stamp_synthetic(fig)
    _save(fig, out_dir, "fig2")


# --------------------------------------------------------------------------------------
# Figure 3 -- methods: eigengap by layer, selected k, flat-spectrum layers
# --------------------------------------------------------------------------------------


def fig3(data, out_dir, plt):
    layers = data["layers"]
    eig = data["eigengap"]
    flat = data["eigengap_flat"]
    ks = data["k"]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, sharex=True, figsize=(TEXT_WIDTH_IN, 3.4),
        gridspec_kw={"height_ratios": [1.6, 0.8], "hspace": 0.16})

    # Flat-spectrum layers: shaded column across both panels + hollow marker.
    flat_layers = [L for L, f in zip(layers, flat) if f]
    for L in flat_layers:
        for ax in (ax_a, ax_b):
            ax.axvspan(L - 0.5, L + 0.5, color=FLAT_SHADE, linewidth=0, zorder=0)

    ax_a.plot(layers, eig, color=COLOR["nl_base"], linestyle="solid", linewidth=LINE_W,
              zorder=2)
    ok_x = [L for L, f in zip(layers, flat) if not f]
    ok_y = [e for e, f in zip(eig, flat) if not f]
    ax_a.plot(ok_x, ok_y, linestyle="none", marker="o", markersize=4.5,
              markerfacecolor=COLOR["nl_base"], markeredgecolor="white",
              markeredgewidth=0.6, zorder=3)
    if flat_layers:
        fl_y = [e for e, f in zip(eig, flat) if f]
        ax_a.plot(flat_layers, fl_y, linestyle="none", marker="o", markersize=MARKER_PT,
                  markerfacecolor="white", markeredgecolor=INK_SOFT, markeredgewidth=1.2,
                  zorder=4)

    # Selected layer = max eigengap among non-flat layers (H1 note / H3 k-selection).
    cand = [(e, L) for L, e, f in zip(layers, eig, flat) if not f] or \
           list(zip(eig, layers))
    e_max, L_max = max(cand)
    ax_a.plot([L_max], [e_max], linestyle="none", marker="D", markersize=MARKER_PT,
              markerfacecolor=COLOR["nl_para"], markeredgecolor=INK, markeredgewidth=0.8,
              zorder=5)
    ax_a.annotate("max eigengap: layer %d, $k$ = %d" % (L_max, ks[layers.index(L_max)]),
                  xy=(L_max, e_max), xytext=(14, 10), textcoords="offset points",
                  ha="left", va="bottom", fontsize=8, color=INK,
                  arrowprops=dict(arrowstyle="-", color=INK_SOFT, linewidth=0.6),
                  zorder=6)
    ax_a.set_ylabel("eigengap (NL base)")
    ax_a.set_ylim(0, 1.45 * max(eig))
    entries = [(layers[-1], eig[-1], "eigengap")]
    if flat_layers:
        # Label the flat marker convention once, at the right edge.
        ylo, yhi = ax_a.get_ylim()
        entries.append((layers[-1], ylo + 0.12 * (yhi - ylo),
                        "open circle / shaded:\nflat spectrum (%d layers)"
                        % len(flat_layers)))
    direct_labels(ax_a, entries, x_pad=1.1, min_gap_frac=0.2)
    ax_a.text(0.0, 1.03, "(a) eigengap by layer on NL base; $k$ chosen once per layer "
              "and reused for all conditions", transform=ax_a.transAxes, ha="left",
              va="bottom", fontsize=9, color=INK)

    ax_b.plot(layers, ks, drawstyle="steps-mid", color=COLOR["nl_break"],
              linestyle="dashed", linewidth=LINE_W, zorder=2)
    ax_b.set_ylabel("selected $k$")
    ax_b.set_xlabel("layer  $L$  (output of block $L$)")
    kmin, kmax = min(ks), max(ks)
    ax_b.set_ylim(kmin - 1, kmax + 1)
    ax_b.set_yticks(sorted({kmin, kmax, ks[layers.index(L_max)]}))
    ax_b.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax_b.set_xticks(_ticks(layers[-1]))
    direct_labels(ax_b, [(layers[-1], ks[-1], "$k$ by layer")], min_gap_frac=0.3)
    ax_b.set_ylim(kmin - 1, kmax + 1)
    ax_b.text(0.0, 1.04, "(b) selected $k$", transform=ax_b.transAxes, ha="left",
              va="bottom", fontsize=9, color=INK)

    for ax in (ax_a, ax_b):
        ax.tick_params(length=2.5, width=0.6)
    fig.subplots_adjust(left=0.13, right=0.76, top=0.92, bottom=0.14)
    if data["synthetic"]:
        _stamp_synthetic(fig)
    _save(fig, out_dir, "fig3")


# --------------------------------------------------------------------------------------


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, stem + ".pdf")
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=200)
    print("wrote %s and %s" % (pdf, png))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=os.path.join(_REPO, "results", "analysis.json"))
    ap.add_argument("--run-id", default=None,
                    help="results/<run_id>/assignments.jsonl to use (auto if only one)")
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "paper", "figs"))
    ap.add_argument("--synthetic", action="store_true",
                    help="render with schema-matched fake data (layout check only)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --synthetic")
    args = ap.parse_args(argv)

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if args.synthetic:
        data = synthetic(seed=args.seed)
        print("SYNTHETIC data: figures are stamped and must not go in the paper.")
    else:
        assignments = _find_assignments(args.run_id)
        data = load_real(args.analysis, assignments)
        print("real data: %s + %s" % (args.analysis, assignments))
        if data["h3"].get("ci_per_pair") is None:
            print("note: analysis.json has no per-layer-pair bootstrap CI; Fig 1(b) shows "
                  "the bootstrap CI of the MEAN difference (the primary endpoint) as a band.")

    fig1(data, args.out_dir, plt)
    fig2(data, args.out_dir, plt)
    fig3(data, args.out_dir, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
