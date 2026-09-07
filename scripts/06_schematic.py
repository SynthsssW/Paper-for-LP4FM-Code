#!/usr/bin/env python3
"""Fig 0 — method schematic. ILLUSTRATIVE ONLY: toy values, no data.

Three panels, one story:
  (a) aligned anchor positions — the same semantic positions in nl_base and formal
      despite different token counts / surface forms (DESIGN.md A.6);
  (b) pooled token states clustered at layer p and layer p+1 with k chosen ONCE
      per layer on nl_base by eigengap and reused for every condition;
  (c) the per-condition cluster-transition graph and the three metrics:
      lambda2 (LOW = a good cut exists; H3 predicts formal DECREASES it),
      path entropy (range [0, log k]), fork layer (first layer at which two
      conditions' assignment sets are disjoint; right-censored if never).

Writes paper/figs/fig0_method.pdf and paper/figs/fig0_method.png (200 dpi).
Visual encodings match scripts/05_figures.py: colour fixed by condition identity,
linestyle as secondary encoding, direct labels, serif fonts, fonttype 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figs"

COLOR = {"nl_base": "#2a78d6", "formal": "#1baf7a", "nl_break": "#4a3aa7",
         "nl_para": "#eb6834"}
LINESTYLE = {"nl_base": "solid", "formal": "dashdot", "nl_break": "dotted",
             "nl_para": "dashed"}

INK = "#1a1a1a"
INK_SOFT = "#6b6b6b"
INK_FAINT = "#b8b8b8"
PAPER = "#ffffff"


def setup_matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib is not installed for this interpreter.")
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "figure.dpi": 100,
        "savefig.dpi": 200,
    })
    return matplotlib


# --------------------------------------------------------------------------------------
# Toy content. Hand-chosen; nothing here is measured.
# --------------------------------------------------------------------------------------

# One toy problem, two realizations. Anchors: each quantity-introduction site + FINAL.
NL_LINES = [["Ana", "has", "12", "apples", ","],
            ["buys", "5", "more", ".", "Total", "?"]]
NL_ANCHORS = {(0, 2): "q1", (1, 1): "q2", (1, 5): "FINAL"}

# `formal` = GIVEN bindings + WHERE (situation retained in prose, quantities replaced
# by identifiers) + FIND. Quantities are introduced in the GIVEN block.
FM_LINES = [["GIVEN", "n", "=", "12", ";", "m", "=", "5"],
            ["WHERE", "Ana", "has", "n", "apples", ",", "buys", "m", "more"],
            ["FIND", "answer", "=", "?"]]
FM_ANCHORS = {(0, 3): "q1", (0, 7): "q2", (2, 3): "FINAL"}

K = 4

# Pooled token states: (condition, cluster at layer p, cluster at layer p+1).
STATES = [
    ("nl_base", 1, 1), ("nl_base", 1, 3), ("nl_base", 2, 4), ("nl_base", 3, 1),
    ("nl_base", 4, 2),
    ("formal", 1, 1), ("formal", 1, 2), ("formal", 2, 2), ("formal", 3, 3),
    ("formal", 4, 4),
]

# Toy eigenvalues for the eigengap inset (nl_base, layer p): clear gap after 4.
EIGS = [0.00, 0.03, 0.06, 0.09, 0.36, 0.41, 0.45]

# Fork-layer strip: per-layer set of clusters occupied by one problem's anchors.
FORK_LAYERS = list(range(7))
FORK_SETS = {
    "nl_base":  [{1, 3}, {1, 3}, {2, 3}, {2, 4}, {2, 4}, {1, 2}, {1, 2}],
    "nl_break": [{1, 3}, {1, 2}, {2, 4}, {1, 3}, {1, 3}, {3, 4}, {3, 4}],
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_title(ax, text):
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=INK)


def pt_to_ax(fig, ax, pts, axis=0):
    """Convert a length in points to axes-fraction units along x (0) or y (1)."""
    inv = ax.transAxes.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform(fig.dpi_scale_trans.transform((pts / 72, pts / 72)))
    return (p1 - p0)[axis]


def lay_tokens(fig, ax, tokens, anchors, x0, y, color, ls, fs=6.6, gap_pt=2.6):
    """Place tokens left-to-right with measured widths; box + label anchor tokens."""
    import matplotlib.patches as mpatches
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    gap_ax = pt_to_ax(fig, ax, gap_pt, 0)
    pad_x = pt_to_ax(fig, ax, 1.4, 0)
    x = x0
    label_text = {"q1": "$q_1$", "q2": "$q_2$", "FINAL": "FINAL"}
    for i, tok in enumerate(tokens):
        is_anchor = i in anchors
        t = ax.text(x, y, tok, transform=ax.transAxes, ha="left", va="center",
                    fontsize=fs, color=INK,
                    fontweight="bold" if is_anchor else "normal", zorder=4)
        bb = t.get_window_extent(renderer).transformed(inv)
        w, h = bb.width, bb.height
        if is_anchor:
            rect = mpatches.FancyBboxPatch(
                (bb.x0 - pad_x, bb.y0 - 0.2 * h), w + 2 * pad_x, h * 1.4,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=ax.transAxes, facecolor=color, alpha=0.22,
                edgecolor=color, linestyle=ls, linewidth=0.9, zorder=3)
            ax.add_patch(rect)
            ax.text(bb.x0 + w / 2, bb.y1 + 0.35 * h, label_text[anchors[i]],
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6,
                    color=color, zorder=5)
        x += w + gap_ax + (2 * pad_x if is_anchor else 0)
    return x


# --------------------------------------------------------------------------------------
# Panel (a) — aligned positions
# --------------------------------------------------------------------------------------

def draw_panel_a(fig, ax):
    blank(ax)
    panel_title(ax, "(a)  Aligned anchor positions")

    cb, cf = COLOR["nl_base"], COLOR["formal"]
    lb, lf = LINESTYLE["nl_base"], LINESTYLE["formal"]

    def header(y, text, color, ls):
        ax.text(0.0, y + 0.012, text, transform=ax.transAxes, fontsize=6.8,
                color=color, ha="left", va="bottom", fontweight="bold")
        ax.plot([0.0, 1.0], [y, y], color=color, linestyle=ls, linewidth=0.8,
                transform=ax.transAxes, clip_on=False)

    # nl_base block
    header(0.895, "nl_base   11 tokens, prose", cb, lb)
    ys = [0.815, 0.715]
    for li, line in enumerate(NL_LINES):
        anchors = {j: lab for (l, j), lab in NL_ANCHORS.items() if l == li}
        lay_tokens(fig, ax, line, anchors, 0.0, ys[li], cb, lb)

    # bridge
    ax.text(0.5, 0.605,
            "$\\Updownarrow$  same three anchors ($q_1$, $q_2$, FINAL),\n"
            "matched by meaning, not by token index",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.3,
            color=INK, linespacing=1.25)

    # formal block
    header(0.50, "formal   21 tokens, bindings + prose", cf, lf)
    ys = [0.42, 0.32, 0.22]
    for li, line in enumerate(FM_LINES):
        anchors = {j: lab for (l, j), lab in FM_ANCHORS.items() if l == li}
        lay_tokens(fig, ax, line, anchors, 0.0, ys[li], cf, lf)

    # footer
    ax.text(0.0, 0.0,
            "Anchor = each quantity-introduction site, plus the\n"
            "final token. Only anchor states enter the analysis,\n"
            "so length and surface form do not enter the graph.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0,
            color=INK_SOFT, linespacing=1.25)


# --------------------------------------------------------------------------------------
# Panel (b) — clustering at two adjacent layers
# --------------------------------------------------------------------------------------

def cluster_centers(side):
    xs = (0.20, 0.30, 0.18, 0.29) if side == "L" else (0.70, 0.80, 0.68, 0.79)
    ys = (0.885, 0.715, 0.545, 0.375)
    return {c: (xs[c - 1], ys[c - 1]) for c in range(1, 5)}


def draw_panel_b(fig, ax):
    import matplotlib.patches as mpatches
    blank(ax)
    panel_title(ax, "(b)  Clustering")

    rng = np.random.default_rng(7)
    cL, cR = cluster_centers("L"), cluster_centers("R")

    for side, centers in (("L", cL), ("R", cR)):
        for c, (x, y) in centers.items():
            ell = mpatches.Ellipse((x, y), 0.20, 0.125, transform=ax.transAxes,
                                   facecolor="none", edgecolor=INK_FAINT,
                                   linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
            ax.add_patch(ell)
            lab = f"$c_{c}$" if side == "L" else f"$c_{c}'$"
            dx = -0.125 if side == "L" else 0.125
            ax.text(x + dx, y, lab, transform=ax.transAxes, ha="center",
                    va="center", fontsize=6.6, color=INK_SOFT)

    jit = rng.uniform(-0.05, 0.05, size=(len(STATES), 4))
    jit[:, [1, 3]] *= 0.65
    marker = {"nl_base": "o", "formal": "^"}
    for i, (cond, cp, cn) in enumerate(STATES):
        xl, yl = cL[cp][0] + jit[i, 0], cL[cp][1] + jit[i, 1]
        xr, yr = cR[cn][0] + jit[i, 2], cR[cn][1] + jit[i, 3]
        ax.annotate("", xy=(xr, yr), xytext=(xl, yl), xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=INK_FAINT, linewidth=0.55,
                                    mutation_scale=5, shrinkA=2.5, shrinkB=2.5,
                                    connectionstyle="arc3,rad=0.06"), zorder=2)
        for x, y in ((xl, yl), (xr, yr)):
            ax.plot([x], [y], marker=marker[cond], markersize=3.6,
                    markerfacecolor=COLOR[cond], markeredgecolor=PAPER,
                    markeredgewidth=0.4, transform=ax.transAxes, zorder=3,
                    linestyle="none")

    ax.text(0.22, 0.965, "layer $p$", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=INK)
    ax.text(0.73, 0.965, "layer $p{+}1$", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, color=INK)
    ax.text(0.5, 0.965, "$\\longrightarrow$", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, color=INK_SOFT)

    # k selection note (full width) + eigengap inset (right) + marker key (left)
    ax.text(0.5, 0.245,
            "$k=4$ chosen ONCE per layer on nl_base\n"
            "by eigengap, then reused for every\n"
            "condition at that layer",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.3, color=INK,
            linespacing=1.25)

    for j, (cond, lab) in enumerate((("nl_base", "nl_base state"),
                                      ("formal", "formal state"))):
        y = 0.115 - 0.06 * j
        ax.plot([0.03], [y], marker=marker[cond], markersize=3.6,
                markerfacecolor=COLOR[cond], markeredgecolor=PAPER, markeredgewidth=0.4,
                transform=ax.transAxes, linestyle="none", clip_on=False)
        ax.text(0.065, y, lab, transform=ax.transAxes, ha="left", va="center",
                fontsize=6.3, color=COLOR[cond])

    x0, x1, y0 = 0.50, 0.97, 0.035
    ax.plot([x0, x1], [y0, y0], color=INK_FAINT, linewidth=0.6, transform=ax.transAxes)
    xs = np.linspace(x0 + 0.02, x1 - 0.02, len(EIGS))
    ys = y0 + 0.012 + np.array(EIGS) * 0.24
    ax.plot(xs[:K], ys[:K], marker="o", markersize=2.4, color=COLOR["nl_base"],
            linestyle="none", transform=ax.transAxes, clip_on=False)
    ax.plot(xs[K:], ys[K:], marker="o", markersize=2.4, color=INK_FAINT,
            linestyle="none", transform=ax.transAxes, clip_on=False)
    xg = 0.5 * (xs[K - 1] + xs[K])
    ax.plot([xg, xg], [ys[K - 1] + 0.012, ys[K] - 0.012], color=INK_SOFT, linewidth=0.7,
            transform=ax.transAxes, linestyle=(0, (2, 1.5)))
    ax.text(xg - 0.02, 0.5 * (ys[K - 1] + ys[K]), "eigengap", transform=ax.transAxes,
            ha="right", va="center", fontsize=5.8, color=INK_SOFT)
    ax.text(0.5 * (x0 + x1), y0 - 0.012, "nl_base Laplacian spectrum",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=INK_SOFT)


# --------------------------------------------------------------------------------------
# Panel (c) — transition graph and the three metrics
# --------------------------------------------------------------------------------------

def bipartite(ax, cond, x_left, x_right, ys, edges, cut=None, title=None, note=None):
    col, ls = COLOR[cond], LINESTYLE[cond]
    for (a, b, w) in edges:
        ax.plot([x_left, x_right], [ys[a], ys[b]], color=col, linestyle=ls,
                linewidth=0.5 + 0.55 * w, alpha=0.9, transform=ax.transAxes,
                solid_capstyle="round", zorder=2)
    for c, y in ys.items():
        for x in (x_left, x_right):
            ax.plot([x], [y], marker="o", markersize=5.0, markerfacecolor=PAPER,
                    markeredgecolor=INK_SOFT, markeredgewidth=0.7,
                    transform=ax.transAxes, zorder=3, linestyle="none")
        ax.text(x_left - 0.03, y, f"$c_{c}$", transform=ax.transAxes, ha="right",
                va="center", fontsize=6.2, color=INK_SOFT)
        ax.text(x_right + 0.03, y, f"$c_{c}'$", transform=ax.transAxes, ha="left",
                va="center", fontsize=6.2, color=INK_SOFT)
    if cut is not None:
        ax.plot([x_left - 0.075, x_right + 0.075], [cut, cut], color=INK,
                linestyle=(0, (2.5, 2)), linewidth=0.8, transform=ax.transAxes, zorder=4)
        ax.text(x_right + 0.085, cut, "cut", transform=ax.transAxes, ha="left",
                va="center", fontsize=6.0, color=INK)
    if title:
        ax.text(0.5 * (x_left + x_right), max(ys.values()) + 0.045, title,
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
                color=col, fontweight="bold")
    if note:
        ax.text(0.5 * (x_left + x_right), min(ys.values()) - 0.045, note,
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
                color=INK, linespacing=1.2)


def draw_panel_c(fig, ax):
    import matplotlib.patches as mpatches
    blank(ax)
    panel_title(ax, "(c)  Transition graph and metrics")

    ys = {1: 0.905, 2: 0.835, 3: 0.765, 4: 0.695}
    nl_edges = [(1, 1, 1), (1, 3, 1), (2, 2, 1), (2, 4, 1), (3, 1, 1), (3, 3, 1),
                (4, 2, 1), (4, 4, 1)]
    fm_edges = [(1, 1, 1.4), (1, 2, 1), (2, 1, 1), (2, 2, 1.4), (3, 3, 1.4), (3, 4, 1),
                (4, 3, 1), (4, 4, 1.4), (2, 3, 0.15)]
    bipartite(ax, "nl_base", 0.11, 0.33, ys, nl_edges, title="nl_base",
              note="well-mixed, no good cut:\n$\\lambda_2$ higher, entropy higher")
    bipartite(ax, "formal", 0.62, 0.84, ys, fm_edges, cut=0.5 * (ys[2] + ys[3]),
              title="formal",
              note="a good cut exists:\n$\\lambda_2$ LOW, entropy lower")
    ax.text(0.475, 0.80, "vs.", transform=ax.transAxes, ha="center", va="center",
            fontsize=6.5, color=INK_SOFT)

    # metric definitions
    ax.text(0.0, 0.545,
            "$\\lambda_2$ = algebraic connectivity of the pooled graph. LOW when\n"
            "a good cut exists (Cheeger): fewer, better-separated regions\n"
            "visited in sequence.  H3 predicts formal $-$ nl_base $<$ 0.\n"
            "Path entropy = spread over transition paths, in $[0,\\ \\log k]$.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=INK,
            linespacing=1.28)

    # fork layer strip
    ax.text(0.0, 0.355, "Fork layer", transform=ax.transAxes, ha="left", va="center",
            fontsize=6.6, color=INK, fontweight="bold")
    ax.text(0.20, 0.355,
            "= first layer at which the two conditions' cluster\n"
            "sets are DISJOINT (nl_base vs. nl_break)",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color=INK,
            linespacing=1.2)
    y_rows = {"nl_base": 0.225, "nl_break": 0.160}
    x0, dx, cw, ch = 0.205, 0.104, 0.092, 0.052
    fork = None
    for li, layer in enumerate(FORK_LAYERS):
        a, b = FORK_SETS["nl_base"][li], FORK_SETS["nl_break"][li]
        overlap = len(a & b) > 0
        if not overlap and fork is None:
            fork = layer
        x = x0 + li * dx
        ax.text(x + cw / 2, y_rows["nl_base"] + ch / 2 + 0.008, f"$\\ell={layer}$",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6,
                color=INK_SOFT)
        for cond in ("nl_base", "nl_break"):
            s = FORK_SETS[cond][li]
            y = y_rows[cond]
            rect = mpatches.Rectangle(
                (x, y - ch / 2), cw, ch, transform=ax.transAxes,
                facecolor=(COLOR[cond] if overlap else PAPER),
                alpha=(0.18 if overlap else 1.0),
                edgecolor=COLOR[cond], linestyle=LINESTYLE[cond], linewidth=0.7,
                zorder=2)
            ax.add_patch(rect)
            ax.text(x + cw / 2, y, "{" + ",".join(str(v) for v in sorted(s)) + "}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=5.9,
                    color=INK, zorder=3)
    for cond in ("nl_base", "nl_break"):
        ax.text(x0 - 0.012, y_rows[cond], cond, transform=ax.transAxes, ha="right",
                va="center", fontsize=6.2, color=COLOR[cond], fontweight="bold")
    xf = x0 + fork * dx - dx / 2 + cw / 2
    y_top = y_rows["nl_base"] + ch / 2 + 0.03
    y_bot = y_rows["nl_break"] - ch / 2 - 0.012
    ax.plot([xf, xf], [y_bot, y_top], color=INK, linewidth=0.9, transform=ax.transAxes,
            zorder=4)
    ax.annotate(f"fork layer = {fork}: clusters shared before (shaded), none after.\n"
                "A pair that never forks is right-censored, not dropped.",
                xy=(xf, y_bot), xytext=(0.5, 0.0), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="bottom", fontsize=6.0,
                color=INK, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.6, shrinkA=0,
                                shrinkB=0, relpos=(0.36, 1.0)))


# --------------------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------------------

def main():
    setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5.5, 3.7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.42, 1.18, 1.95],
                  left=0.012, right=0.992, top=0.915, bottom=0.085, wspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    fig.canvas.draw()  # text measurement in panel (a) needs a renderer

    draw_panel_a(fig, ax_a)
    draw_panel_b(fig, ax_b)
    draw_panel_c(fig, ax_c)

    fig.text(0.5, 0.012,
             "Illustrative schematic with hand-chosen toy values ($k=4$, one problem, "
             "a handful of states). Not data; results are in Figs. 1–3.",
             ha="center", va="bottom", fontsize=6.5, color=INK_SOFT, style="italic")

    pdf = OUT_DIR / "fig0_method.pdf"
    png = OUT_DIR / "fig0_method.png"
    fig.savefig(pdf, facecolor=PAPER)
    fig.savefig(png, dpi=200, facecolor=PAPER)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    sys.exit(main())
