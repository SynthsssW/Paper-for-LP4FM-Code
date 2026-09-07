#!/usr/bin/env python3
"""Clean publication figures for the LP4FM 2026 paper.

A redesign pass over scripts/05_figures.py and scripts/08_headline_figure.py.
Same data, same claims -- restrained seaborn styling, one consistent condition
palette, direct labels, vector PDF + 300-dpi PNG. NOTHING here recomputes the
science; every number is read from results/ or transcribed from
prereg/DEVIATIONS.md with the observation cited in a comment.

Reads
  results/analysis.json                     scripts/04_analyze.py  (eigengap-k)
  results/analysis_fixedk8.json             scripts/04_analyze.py  (forced k=8)
  results/main/assignments.jsonl            eigengap-k cluster assignments
  results/fixedk8/assignments.jsonl         forced-k=8 cluster assignments
  results/g2_behavioral.json                gate G2 accuracy + Wilson CIs
  data/corpus/covariates_measures.json      per-pair surface measures

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
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

    rows = [
        ("formal notation",              "quantities as symbols, given/goal frame", 100, 100),
        ("paraphrase",                   "reworded, same numbers",                   35, 100),
        ("minimal edit: digits\u2192words", "one number respelled",                    8,  99),
        ("minimal edit: name swap",      "one proper name changed",                   4,  70),
        ("minimal edit: one digit",      "answer is now different",                   0, 100),
    ]

    fig, ax = plt.subplots(figsize=(5.6, 2.95))
    ys = np.arange(len(rows))[::-1]
    XCOL = 116.0          # right-hand column
    BAR = 0.46

    for y, (name, gloss, k, n) in zip(ys, rows):
        brk = (k == 0)
        c = BREAK if brk else PRESERVE
        rate = 100.0 * k / n

        if brk:
            ax.add_patch(plt.Rectangle((0, y - 0.5), XCOL + 26, 1.0,
                                       color=c, alpha=0.075, lw=0, zorder=0,
                                       clip_on=False))
            ax.hlines(y, 0, 1.1, color=c, lw=6, zorder=3)   # visible zero stub
        else:
            ax.barh(y, rate, height=BAR, color=c, zorder=2, lw=0)

        ax.text(rate + 2.2 if not brk else 3.6, y, f"{k}/{n}",
                va="center", ha="left", fontsize=8.2, zorder=4,
                color=INK if not brk else c,
                fontweight="bold" if brk else "normal")

        ax.text(-3.5, y + 0.17, name, ha="right", va="center", fontsize=8.4,
                color=INK if not brk else c,
                fontweight="bold" if brk else "normal")
        ax.text(-3.5, y - 0.21, gloss, ha="right", va="center", fontsize=6.9,
                color=INK_SOFT, style="italic")

        ax.text(XCOL, y, "Yes (100/100)" if brk else "No",
                ha="left", va="center", fontsize=8.4, clip_on=False,
                color=c if brk else INK_SOFT,
                fontweight="bold" if brk else "normal")

    top = len(rows) - 0.34
    ax.text(0, top, "Divergence measure fires", fontsize=8.6,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(XCOL, top, "Answer changed?", fontsize=8.6, fontweight="bold",
            color=INK, ha="left", va="bottom", clip_on=False)

    ax.set_xlim(0, 108)
    ax.set_ylim(-0.62, len(rows) - 0.12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("% of problems")
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="x", length=3, labelsize=7.6)

    fig.text(0.012, -0.10,
             "100 GSM-style problems, each rewritten five ways and compared with the "
             "original. The measure is the first layer at which\n"
             "the two prompts' anchor-state clusters separate (forced $k$=8). Only the "
             "one-digit edit changes the answer: the model\n"
             "returns the old answer 0/100 times and its new answer at 67% accuracy "
             "(70% on the originals). It is never detected.",
             ha="left", va="top", fontsize=6.6, color=INK_SOFT, linespacing=1.55)
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
        2, 1, figsize=(5.5, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.28})

    # (a) lambda2 by layer-pair. formal and nl_base carry the contrast; the other
    #     four conditions sit on top of nl_base -- draw them faint for context.
    for cond in ("nl_minimal_orth", "nl_minimal_lex", "nl_para", "nl_break"):
        axA.plot(pairs, bc[cond], color=COLOR[cond], lw=0.8, alpha=0.30, zorder=1)
    axA.plot(pairs, bc["nl_base"], color=COLOR["nl_base"], lw=1.8, zorder=3,
             label="NL base")
    axA.plot(pairs, bc["formal"], color=COLOR["formal"], lw=1.8, zorder=3,
             label="formal notation")
    axA.plot([], [], color=INK_SOFT, lw=0.8, alpha=0.4,
             label="paraphrase, minimal edits")
    axA.set_ylabel(r"$\lambda_2$  (pooled, cosine affinity)")
    axA.set_ylim(-0.05, 2.35)
    axA.set_yticks([0, 1, 2])
    axA.set_title(r"(a) algebraic connectivity $\lambda_2$ by layer-pair "
                  r"— quantized to $\{0,1,2\}$", loc="left", fontsize=8.8)
    axA.legend(loc="upper left", fontsize=7, handlelength=1.6, ncol=3,
               columnspacing=1.1, borderaxespad=0.2)

    # (b) paired difference formal - nl_base: the pre-registered primary endpoint.
    axB.axhline(0, color=INK_SOFT, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axB.plot(pairs, diff, color=INK, lw=1.4, zorder=3)
    axB.axhspan(ci["ci_low"], ci["ci_high"], color=COLOR["formal"], alpha=0.16, zorder=1)
    axB.axhline(mean, color=COLOR["formal"], lw=1.3, ls="dashdot", zorder=2)
    axB.set_ylabel(r"$\Delta\lambda_2$  (formal $-$ NL base)")
    axB.set_xlabel(r"layer-pair index  $p$  =  (block $p$, block $p{+}1$)")
    axB.set_xlim(pairs[0], pairs[-1])
    axB.set_ylim(-0.25, 2.05)
    axB.set_xticks([0, 10, 20, 30, 40, 46])
    axB.set_title("(b) paired difference — pre-registered primary endpoint",
                  loc="left", fontsize=8.8)
    axB.annotate(f"mean $\\Delta\\lambda_2$ = +{mean:.2f}",
                 (pairs[-1], mean), xytext=(-4, 6), textcoords="offset points",
                 ha="right", va="bottom", fontsize=7, color=COLOR["formal"],
                 fontweight="bold")
    txt = (f"mean +{mean:.3f}, 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]\n"
           f"pre-registered direction was NEGATIVE — sign is opposite\n"
           f"forced k=8: mean +{k8['mean_paired_difference']:.3f}, "
           f"Wilcoxon p = {k8['wilcoxon']['p']:.2f}  (not robust to k)")
    axB.text(0.985, 0.96, txt, transform=axB.transAxes, va="top", ha="right",
             fontsize=6.7, color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK_SOFT, lw=0.5,
                       alpha=0.92))

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
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharey=True)
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
                            lw=2.4 if emph else 1.4, zorder=3 if emph else 2)
            handles.setdefault(cond, ln)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("layer $L$")
        ax.set_xlim(0, 47)
        ax.set_xticks([0, 12, 24, 36, 47])
        sns.despine(ax=ax)
        ax.tick_params(length=3)

    axes[0].set_ylabel("% of pairs forked by layer $L$")
    axes[0].set_ylim(-4, 108)
    axes[0].legend([handles[c] for c in order], [PRETTY[c] for c in order],
                   loc="upper left", fontsize=6.6, handlelength=1.5,
                   borderaxespad=0.3, labelspacing=0.35)
    fig.subplots_adjust(wspace=0.1, bottom=0.22)
    fig.text(0.5, 0.02,
             "Cumulative fork incidence vs. NL base. NL answer-breaking (amber) stays "
             "flat at 0% in both panels; under\nforced k=8 the formal rewrite forks "
             "every problem by layer ~3 — the contrast and the diagnostic's "
             "sensitivity to k both show here.",
             ha="center", va="top", fontsize=6.4, color=INK_SOFT, linespacing=1.4)
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

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.7),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    x = np.arange(len(preds))
    w = 0.38
    axA.bar(x - w / 2, all5, w, color="#8FA9C8", label="all 5 conditions  ($n$ = 469)")
    axA.bar(x + w / 2, no_formal, w, color="#C0403B",
            label="formal excluded  ($n$ = 369)")
    for xi, v in zip(x - w / 2, all5):
        axA.text(xi, v + 0.014, f"{v:.2f}", ha="center", fontsize=6.6, color=INK_SOFT)
    for xi, v in zip(x + w / 2, no_formal):
        axA.text(xi, v + 0.014, f"{v:.2f}", ha="center", fontsize=6.6, color=INK_SOFT)
    axA.set_xticks(x)
    axA.set_xticklabels(preds, fontsize=6.8)
    axA.set_ylabel("McFadden $R^2$  (fork occurrence)")
    axA.set_ylim(0, 0.66)
    axA.set_title("(a) no surface measure survives\ndropping the one long condition",
                  loc="left", fontsize=8)
    axA.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    sns.despine(ax=axA)

    # manual label placement -- orth / lex / break are all bunched at ratio ~1.0
    lab = {
        "formal":          dict(xytext=(-6, -2), ha="right"),
        "nl_para":         dict(xytext=(0, 10), ha="center"),
        "nl_minimal_orth": dict(xytext=(9, 7), ha="left"),
        "nl_minimal_lex":  dict(xytext=(9, -1), ha="left"),
        "nl_break":        dict(xytext=(9, -9), ha="left"),
    }
    for cond, ratio, rate in cond_rows:
        axB.scatter(ratio, rate, s=48, color=COLOR[cond], zorder=3,
                    edgecolor="white", linewidth=0.8)
        axB.annotate(PRETTY[cond].replace(" notation", "").replace("NL ", ""),
                     (ratio, rate), textcoords="offset points", fontsize=6.3,
                     color=COLOR[cond], va="center", **lab[cond])
    axB.set_xlabel("token-length ratio  (rewrite / base)")
    axB.set_ylabel("fork rate  (%)")
    axB.set_title("(b) only the 54%-longer\ncondition forks", loc="left", fontsize=8)
    axB.set_xlim(0.9, 1.68)
    axB.set_ylim(-10, 115)
    sns.despine(ax=axB)

    for ax in (axA, axB):
        ax.tick_params(length=3)
    fig.subplots_adjust(wspace=0.42)
    fig.text(0.5, -0.08,
             "Which surface property drives forking is unidentified: token-count "
             "difference looks dominant across all conditions\n(a, left bars) but "
             "collapses to $R^2$=0.05 once the formal rewrite is removed (right bars). "
             "See DEVIATIONS Obs. 5d.",
             ha="center", va="top", fontsize=6.3, color=INK_SOFT, linespacing=1.4)
    _save(fig, "fig4_surface")


# --------------------------------------------------------------------------------------
# Figure 5 -- behavioural accuracy
# --------------------------------------------------------------------------------------

def fig5_accuracy():
    g2 = _load_json("results/g2_behavioral.json")["per_condition"]
    order = ["nl_base", "nl_para", "nl_minimal_lex", "nl_minimal_orth", "nl_break",
             "formal"]
    fig, ax = plt.subplots(figsize=(5.5, 2.7))

    for i, cond in enumerate(order):
        d = g2[cond]
        acc = 100 * d["accuracy"]
        lo, hi = 100 * d["wilson95"][0], 100 * d["wilson95"][1]
        c = COLOR[cond]
        ax.plot([lo, hi], [i, i], color=c, lw=2.0, alpha=0.55, zorder=2)
        ax.plot(acc, i, "o", ms=8, color=c, zorder=3, markeredgecolor="white",
                markeredgewidth=0.8)
        note = ""
        if cond == "formal":
            note = "  − 14 pp vs. NL base (McNemar p = 0.007)"
        if cond == "nl_break":
            note = "  (own perturbed answer; old answer 0/100)"
        ax.text(acc + 1.2, i, f"{d['n_correct']}/{d['n']}{note}", va="center",
                fontsize=7, color=c if cond in ("formal", "nl_break") else INK_SOFT,
                fontweight="bold" if cond == "formal" else "normal")

    ax.axvline(70, color=COLOR["nl_base"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY[c] for c in order])
    ax.set_xlabel("solve accuracy  (%)  with Wilson 95% interval")
    ax.set_xlim(40, 90)
    ax.set_ylim(-0.6, len(order) - 0.4)
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
        2, 1, figsize=(5.5, 3.4), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 0.8], "hspace": 0.22})

    for L in layers[degen]:
        for ax in (axA, axB):
            ax.axvspan(L - 0.5, L + 0.5, color="#EED9C4", alpha=0.7, lw=0, zorder=0)

    axA.plot(layers, eig, color=COLOR["nl_para"], lw=1.6, zorder=2)
    axA.scatter(layers[~degen], eig[~degen], s=16, color=COLOR["nl_para"], zorder=3,
                edgecolor="white", linewidth=0.5)
    axA.scatter(layers[degen], eig[degen], s=34, facecolor="white",
                edgecolor=COLOR["formal"], linewidth=1.3, zorder=4)
    axA.set_ylabel("eigengap (NL base)")
    axA.set_ylim(0, 0.82)
    axA.set_title("(a) spectral eigengap by layer", loc="left")
    axA.annotate("shaded: partition isolates < 1% of states\n"
                 "(single-token clusters on 6 layers)",
                 (17, 0.62), fontsize=6.3, color=COLOR["formal"])

    axB.step(layers, ks, where="mid", color=COLOR["nl_break"], lw=1.6)
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
