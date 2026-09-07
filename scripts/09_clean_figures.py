#!/usr/bin/env python3
"""Clean publication figures for the LP4FM 2026 paper.

A redesign pass over scripts/05_figures.py and scripts/08_headline_figure.py.
Same data, same claims -- restrained seaborn styling, one consistent condition
palette, direct labels, 10 pt type throughout, vector PDF (+ PNG preview).
NOTHING here recomputes the science; every number is read from results/ or
transcribed from prereg/DEVIATIONS.md with the observation cited in a comment.

Reads
  results/analysis.json                     scripts/04_analyze.py  (eigengap-k)
  results/analysis_fixedk8.json             scripts/04_analyze.py  (forced k=8)
  results/main/assignments.jsonl            eigengap-k cluster assignments
  results/fixedk8/assignments.jsonl         forced-k=8 cluster assignments
  results/g2_behavioral.json                gate G2 accuracy + Wilson CIs

Writes  paper/figs/clean/*.{pdf,png}
  fig1_headline     fork rate vs whether the answer changed, by condition
  fig2_lambda2      lambda2 by layer-pair; formal - nl_base paired difference; k-sensitivity
  fig3_forklayer    cumulative fork incidence by condition, eigengap-k vs forced k=8
  fig4_surface      McFadden R2 for fork occurrence, all conditions vs formal excluded
  fig5_accuracy     solve accuracy by condition with Wilson 95% intervals
  fig6_clustering   eigengap by layer and the k it selects; degenerate-partition layers flagged

Usage
  python scripts/09_clean_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "paper", "figs", "clean")

FS = 10  # every piece of type in every figure is 10 pt

# --------------------------------------------------------------------------------------
# Style. One palette, assigned by condition identity. Colour-blind safe (Okabe-Ito-ish).
# --------------------------------------------------------------------------------------

COLOR = {
    "nl_base":         "#6B6B6B",   # reference: neutral grey
    "nl_para":         "#3B6DB3",   # blue
    "formal":          "#C0403B",   # red   -- the rewrite that restructures the geometry
    "nl_minimal_orth": "#3F8F5B",   # green
    "nl_minimal_lex":  "#7D5BA6",   # purple
    "nl_break":        "#E08214",   # amber -- the only edit that changes the answer
}
PRETTY = {
    "nl_base": "NL base",
    "nl_para": "NL paraphrase",
    "formal": "formal notation",
    "nl_minimal_orth": "NL minimal (digits→words)",
    "nl_minimal_lex": "NL minimal (name swap)",
    "nl_break": "NL answer-breaking",
}
INK, INK_SOFT = "#1A1A1A", "#5A5A5A"


def _style():
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.titleweight": "bold",
        "axes.labelsize": FS,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def _save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, stem + ".{pdf,png}"))


# --------------------------------------------------------------------------------------
# Shared data loading
# --------------------------------------------------------------------------------------

def _load_json(rel):
    with open(os.path.join(_REPO, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_assignments(rel):
    layers = []
    with open(os.path.join(_REPO, rel), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                layers.append(json.loads(line))
    layers.sort(key=lambda d: d["layer"])
    key0 = [(r["problem_id"], r["condition"], r["anchor"]) for r in layers[0]["rows"]]
    return layers, key0


def fork_layers(layers, key0, ref, other):
    """First layer at which the two conditions' anchor-cluster SETS are disjoint.

    Mirrors scripts/05_figures.fork_layers_from_assignments / src/ling/metrics.
    Returns {problem_id: layer or None}; None == never forks (right-censored).
    """
    idx = {}
    for j, (pid, cond, _a) in enumerate(key0):
        idx.setdefault((pid, cond), []).append(j)
    out = {}
    for pid in sorted({p for p, c, _ in key0 if c == ref}):
        i1, i2 = idx.get((pid, ref)), idx.get((pid, other))
        if i1 is None or i2 is None:
            continue  # Amendment 6 declared hole -- skip the pair, never fabricate
        fork = None
        for d in layers:
            cl = [r["cluster"] for r in d["rows"]]
            if {cl[j] for j in i1}.isdisjoint({cl[j] for j in i2}):
                fork = d["layer"]
                break
        out[pid] = fork
    return out


# --------------------------------------------------------------------------------------
# Figure 1 -- headline
# --------------------------------------------------------------------------------------

def fig1_headline():
    """Fork rate by condition against whether the model's answer changed.

    Fork rates: forced-k=8 run (DEVIATIONS Obs. 4). Behavioural rates: gate G2.

    Colour encodes ONE thing: whether the edit changed the answer. The four
    answer-preserving conditions share a colour because they are one category;
    giving each its own hue would be decoration, and it would bury the contrast
    the figure exists to make.
    """
    PRESERVE = "#3B6DB3"   # answer preserved
    BREAK    = "#D1660F"   # answer changed -- the only row that differs

    # name, plain-language gloss, forked, n_pairs  (DEVIATIONS Obs. 4, forced k=8)
    rows = [
        ("formal notation",              "quantities as symbols, given/goal frame", 100, 100),
        ("paraphrase",                   "reworded, same numbers",                   35, 100),
        ("minimal edit: digits→words", "one number respelled",                    8,  99),
        ("minimal edit: name swap",      "one proper name changed",                   4,  70),
        ("minimal edit: one token",      "answer is now different",                   0, 100),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ys = np.arange(len(rows))[::-1]
    XCOL = 116.0          # right-hand column
    BAR = 0.5

    for y, (name, gloss, k, n) in zip(ys, rows):
        brk = (k == 0)
        c = BREAK if brk else PRESERVE
        rate = 100.0 * k / n

        if brk:
            ax.add_patch(plt.Rectangle((0, y - 0.5), XCOL + 30, 1.0,
                                       color=c, alpha=0.075, lw=0, zorder=0,
                                       clip_on=False))
            ax.hlines(y, 0, 1.1, color=c, lw=7, zorder=3)   # visible zero stub
        else:
            ax.barh(y, rate, height=BAR, color=c, zorder=2, lw=0)

        ax.text(rate + 2.4 if not brk else 4.0, y, f"{k}/{n}",
                va="center", ha="left", fontsize=FS, zorder=4,
                color=INK if not brk else c,
                fontweight="bold" if brk else "normal")

        ax.text(-3.5, y + 0.19, name, ha="right", va="center", fontsize=FS,
                color=INK if not brk else c,
                fontweight="bold" if brk else "normal")
        ax.text(-3.5, y - 0.22, gloss, ha="right", va="center", fontsize=FS,
                color=INK_SOFT, style="italic")

        ax.text(XCOL, y, "Yes (100/100)" if brk else "No",
                ha="left", va="center", fontsize=FS, clip_on=False,
                color=c if brk else INK_SOFT,
                fontweight="bold" if brk else "normal")

    top = len(rows) - 0.30
    ax.text(0, top, "Divergence measure fires", fontsize=FS,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(XCOL, top, "Answer changed?", fontsize=FS, fontweight="bold",
            color=INK, ha="left", va="bottom", clip_on=False)

    ax.set_xlim(0, 108)
    ax.set_ylim(-0.58, len(rows) - 0.02)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("% of pairs")
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="x", length=3)
    _save(fig, "fig1_headline")


# --------------------------------------------------------------------------------------
# Figure 2 -- lambda2 (H3)
# --------------------------------------------------------------------------------------

def fig2_lambda2():
    a = _load_json("results/analysis.json")
    h3 = a["primary_endpoints"]["h3"]
    bc = a["exploratory"]["lambda2_by_condition"]["by_condition"]
    pairs = np.asarray(h3["layer_pairs"], float)
    diff = np.asarray(h3["paired_differences"], float)
    mean = h3["mean_paired_difference"]
    ci = h3["bootstrap_ci"]
    k8 = _load_json("results/analysis_fixedk8.json")["primary_endpoints"]["h3"]

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.30})

    # (a) lambda2 by layer-pair. formal and nl_base carry the contrast; the other
    #     four conditions sit on top of nl_base -- draw them faint for context.
    for cond in ("nl_minimal_orth", "nl_minimal_lex", "nl_para", "nl_break"):
        axA.plot(pairs, bc[cond], color=COLOR[cond], lw=0.9, alpha=0.30, zorder=1)
    axA.plot(pairs, bc["nl_base"], color=COLOR["nl_base"], lw=2.0, zorder=3,
             label="NL base")
    axA.plot(pairs, bc["formal"], color=COLOR["formal"], lw=2.0, zorder=3,
             label="formal notation")
    axA.plot([], [], color=INK_SOFT, lw=0.9, alpha=0.4,
             label="paraphrase, minimal edits")
    axA.set_ylabel(r"$\lambda_2$  (pooled, cosine affinity)")
    axA.set_ylim(-0.05, 2.5)
    axA.set_yticks([0, 1, 2])
    axA.set_title(r"(a) algebraic connectivity $\lambda_2$ by layer-pair "
                  r"— quantized to $\{0,1,2\}$", loc="left")
    axA.legend(loc="upper left", handlelength=1.7, ncol=3, columnspacing=1.2,
               borderaxespad=0.2)

    # (b) paired difference formal - nl_base: the pre-registered primary endpoint.
    axB.axhline(0, color=INK_SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axB.plot(pairs, diff, color=INK, lw=1.6, zorder=3)
    axB.axhspan(ci["ci_low"], ci["ci_high"], color=COLOR["formal"], alpha=0.16, zorder=1)
    axB.axhline(mean, color=COLOR["formal"], lw=1.5, ls="dashdot", zorder=2)
    axB.set_ylabel(r"$\Delta\lambda_2$  (formal $-$ NL base)")
    axB.set_xlabel(r"layer-pair index  $p$  =  (block $p$, block $p{+}1$)")
    axB.set_xlim(pairs[0], pairs[-1])
    axB.set_ylim(-0.28, 2.35)
    axB.set_xticks([0, 10, 20, 30, 40, 46])
    axB.set_title("(b) paired difference — pre-registered primary endpoint", loc="left")
    axB.annotate(f"mean $\\Delta\\lambda_2$ = +{mean:.2f}",
                 (pairs[-1], mean), xytext=(-4, 7), textcoords="offset points",
                 ha="right", va="bottom", fontsize=FS, color=COLOR["formal"],
                 fontweight="bold")
    txt = (f"mean +{mean:.3f}, 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]\n"
           f"pre-registered direction was NEGATIVE — sign is opposite\n"
           f"forced k=8: mean +{k8['mean_paired_difference']:.3f}, "
           f"Wilcoxon p = {k8['wilcoxon']['p']:.2f}  (not robust to k)")
    axB.text(0.985, 0.965, txt, transform=axB.transAxes, va="top", ha="right",
             color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK_SOFT, lw=0.6,
                       alpha=0.93))

    for ax in (axA, axB):
        sns.despine(ax=ax)
        ax.tick_params(length=3)
    _save(fig, "fig2_lambda2")


# --------------------------------------------------------------------------------------
# Figure 3 -- cumulative fork incidence
# --------------------------------------------------------------------------------------

def _cum_incidence(forks, layers):
    forks = list(forks)
    n = len(forks)
    return np.array([sum(1 for f in forks if f is not None and f <= L) / n
                     for L in layers])


def fig3_forklayer():
    order = ("formal", "nl_para", "nl_minimal_orth", "nl_minimal_lex", "nl_break")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True)
    handles = {}

    for ax, (rel, title) in zip(axes, [
        ("results/main/assignments.jsonl",
         "(a) eigengap-selected $k$  (mostly $k$ = 2)"),
        ("results/fixedk8/assignments.jsonl",
         "(b) forced $k$ = 8"),
    ]):
        layers_raw, key0 = _read_assignments(rel)
        layers = [d["layer"] for d in layers_raw]
        for cond in order:
            fk = fork_layers(layers_raw, key0, "nl_base", cond)
            if not fk:
                continue
            inc = _cum_incidence(fk.values(), layers)
            emph = cond in ("formal", "nl_break")
            (ln,) = ax.plot(layers, 100 * inc, color=COLOR[cond],
                            lw=2.6 if emph else 1.6, zorder=3 if emph else 2)
            handles.setdefault(cond, ln)
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer $L$")
        ax.set_xlim(0, 47)
        ax.set_xticks([0, 12, 24, 36, 47])
        sns.despine(ax=ax)
        ax.tick_params(length=3)

    axes[0].set_ylabel("% of pairs forked by layer $L$")
    axes[0].set_ylim(-4, 108)
    axes[0].legend([handles[c] for c in order], [PRETTY[c] for c in order],
                   loc="upper left", handlelength=1.6, borderaxespad=0.3,
                   labelspacing=0.35)
    fig.subplots_adjust(wspace=0.09)
    _save(fig, "fig3_forklayer")


# --------------------------------------------------------------------------------------
# Figure 4 -- surface attribution (DEVIATIONS Obs. 5b -> 5c -> 5d)
# --------------------------------------------------------------------------------------

def fig4_surface():
    # Univariate McFadden R2 for fork occurrence (k=8). DEVIATIONS Obs. 5c / 5d.
    preds = ["lexical\nedit distance", "syntactic\n(dep-bag TV)", "token-count\ndifference"]
    all5 = [0.2507, 0.3946, 0.5254]
    no_formal = [0.1695, 0.1215, 0.0501]

    # Per-condition fork rate vs token-length ratio. DEVIATIONS Obs. 5c table.
    cond_rows = [
        ("formal",          1.54, 100.0),
        ("nl_para",         0.97,  35.0),
        ("nl_minimal_orth", 1.02,   8.1),
        ("nl_minimal_lex",  1.00,   5.7),
        ("nl_break",        1.00,   0.0),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.8, 3.6),
                                   gridspec_kw={"width_ratios": [1.1, 1.0]})

    x = np.arange(len(preds))
    w = 0.38
    axA.bar(x - w / 2, all5, w, color="#8FA9C8", label="all 5 conditions")
    axA.bar(x + w / 2, no_formal, w, color="#C0403B", label="formal excluded")
    for xi, v in zip(x - w / 2, all5):
        axA.text(xi, v + 0.016, f"{v:.2f}", ha="center", fontsize=FS, color=INK_SOFT)
    for xi, v in zip(x + w / 2, no_formal):
        axA.text(xi, v + 0.016, f"{v:.2f}", ha="center", fontsize=FS, color=INK_SOFT)
    axA.set_xticks(x)
    axA.set_xticklabels(preds)
    axA.set_ylabel("McFadden $R^2$  (fork occurrence)")
    axA.set_ylim(0, 0.74)
    axA.set_title("(a) no surface measure survives dropping\n"
                  r"the one long condition  ($n$: 469 $\to$ 369)", loc="left")
    axA.legend(loc="upper left", handlelength=1.3, borderaxespad=0.4)
    sns.despine(ax=axA)

    for cond, ratio, rate in cond_rows:
        axB.scatter(ratio, rate, s=70, color=COLOR[cond], zorder=3,
                    edgecolor="white", linewidth=0.9,
                    label=PRETTY[cond].replace(" notation", "").replace("NL ", ""))
    axB.set_xlabel("token-length ratio  (rewrite / base)")
    axB.set_ylabel("fork rate  (%)")
    axB.set_title("(b) only the 54%-longer\ncondition forks", loc="left")
    axB.set_xlim(0.9, 1.68)
    axB.set_ylim(-8, 116)
    axB.legend(loc="center right", handletextpad=0.2, labelspacing=0.35,
               borderaxespad=0.4)
    sns.despine(ax=axB)

    for ax in (axA, axB):
        ax.tick_params(length=3)
    fig.subplots_adjust(wspace=0.4)
    _save(fig, "fig4_surface")


# --------------------------------------------------------------------------------------
# Figure 5 -- behavioural accuracy
# --------------------------------------------------------------------------------------

def fig5_accuracy():
    g2 = _load_json("results/g2_behavioral.json")["per_condition"]
    order = ["nl_base", "nl_para", "nl_minimal_lex", "nl_minimal_orth", "nl_break",
             "formal"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    for i, cond in enumerate(order):
        d = g2[cond]
        acc = 100 * d["accuracy"]
        lo, hi = 100 * d["wilson95"][0], 100 * d["wilson95"][1]
        c = COLOR[cond]
        ax.plot([lo, hi], [i, i], color=c, lw=2.4, alpha=0.55, zorder=2)
        ax.plot(acc, i, "o", ms=9, color=c, zorder=3, markeredgecolor="white",
                markeredgewidth=0.9)
        note = ""
        if cond == "formal":
            note = "   − 14 pp vs. base  (p = 0.007)"
        if cond == "nl_break":
            note = "   (own answer; old answer 0/100)"
        ax.text(hi + 1.4, i, f"{d['n_correct']}/{d['n']}{note}", va="center",
                fontsize=FS, color=c if cond in ("formal", "nl_break") else INK_SOFT,
                fontweight="bold" if cond == "formal" else "normal")

    ax.axvline(70, color=COLOR["nl_base"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY[c] for c in order])
    ax.set_xlabel("solve accuracy  (%)  with Wilson 95% interval")
    ax.set_xlim(40, 92)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_title("Formal rewriting restructures the geometry and degrades accuracy",
                 loc="left")
    sns.despine(ax=ax, left=True)
    ax.tick_params(length=3)
    _save(fig, "fig5_accuracy")


# --------------------------------------------------------------------------------------
# Figure 6 -- clustering front end
# --------------------------------------------------------------------------------------

def fig6_clustering():
    layers_raw, _ = _read_assignments("results/main/assignments.jsonl")
    layers = np.array([d["layer"] for d in layers_raw])
    eig = np.array([d["eigengap"] for d in layers_raw])
    ks = np.array([d["k"] for d in layers_raw])

    # smallest cluster share per layer -> flag degenerate partitions
    small = []
    for d in layers_raw:
        cnt = {}
        for r in d["rows"]:
            cnt[r["cluster"]] = cnt.get(r["cluster"], 0) + 1
        small.append(min(cnt.values()) / len(d["rows"]))
    small = np.array(small)
    degen = small < 0.01

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 0.85], "hspace": 0.28})

    for L in layers[degen]:
        for ax in (axA, axB):
            ax.axvspan(L - 0.5, L + 0.5, color="#EED9C4", alpha=0.7, lw=0, zorder=0)

    axA.plot(layers, eig, color=COLOR["nl_para"], lw=1.8, zorder=2)
    axA.scatter(layers[~degen], eig[~degen], s=22, color=COLOR["nl_para"], zorder=3,
                edgecolor="white", linewidth=0.6)
    axA.scatter(layers[degen], eig[degen], s=48, facecolor="white",
                edgecolor=COLOR["formal"], linewidth=1.5, zorder=4)
    axA.set_ylabel("eigengap (NL base)")
    axA.set_ylim(0, 0.9)
    axA.set_title("(a) spectral eigengap by layer", loc="left")
    axA.annotate("shaded: partition isolates < 1% of states\n"
                 "(single-token clusters on 6 layers)",
                 (17, 0.66), fontsize=FS, color=COLOR["formal"])

    axB.step(layers, ks, where="mid", color=COLOR["nl_break"], lw=1.8)
    axB.set_ylabel("selected $k$")
    axB.set_xlabel("layer $L$")
    axB.set_yscale("log")
    axB.set_yticks([2, 3, 20])
    axB.set_yticklabels(["2", "3", "20"])
    axB.set_ylim(1.7, 26)
    axB.set_xlim(-0.5, 47.5)
    axB.set_xticks([0, 12, 24, 36, 47])
    axB.set_title(r"(b) $k$ = 2 on 40 of 48 layers", loc="left")

    for ax in (axA, axB):
        sns.despine(ax=ax)
        ax.tick_params(length=3)
    _save(fig, "fig6_clustering")


# --------------------------------------------------------------------------------------

def main():
    _style()
    fig1_headline()
    fig2_lambda2()
    fig3_forklayer()
    fig4_surface()
    fig5_accuracy()
    fig6_clustering()
    print("\nall figures in", OUT)


if __name__ == "__main__":
    main()
