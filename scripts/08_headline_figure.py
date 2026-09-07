"""Headline figure: the fork diagnostic fires on answer-preserving rewrites and
never on the one edit that changes the answer.

Numbers are transcribed from prereg/DEVIATIONS.md (Observations 4 and 6) and
results/g2_behavioral.json. Fork rates: forced-k=8 exploratory run. Behavioural
rates (answer changed / accuracy): gate G2.

Writes paper/figs/fig_headline.pdf and .png (200 dpi).
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = pathlib.Path(__file__).resolve().parent.parent / "paper" / "figs"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#1a1a1a", "#555555", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
})

# condition, what changed, forks, n, answer changed?
ROWS = [
    ("formal",          "formal notation, GIVEN/WHERE/FIND frame", 100, 100, False),
    ("nl_para",         "paraphrase",                              35, 100, False),
    ("nl_minimal_orth", "one number, digits→words",           8,  99, False),
    ("nl_minimal_lex",  "one proper name swapped",                  4,  70, False),
    ("nl_break",        "one digit changed",                        0, 100, True),
]

fig = plt.figure(figsize=(5.5, 2.9))
ax = fig.add_axes([0.335, 0.265, 0.50, 0.635])   # leave room left for labels, right for column
n = len(ROWS)
ys = list(range(n))[::-1]
H = 0.62

for y, (cond, what, k, N, brk) in zip(ys, ROWS):
    rate = 100 * k / N
    col = ORANGE if brk else BLUE
    if brk:
        # hatched row band + zero marker: second encoding, survives grayscale
        ax.add_patch(Rectangle((-2, y - 0.46), 106, 0.92, facecolor=ORANGE, alpha=0.10,
                               edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((-2, y - 0.46), 106, 0.92, facecolor="none",
                               edgecolor=ORANGE, hatch="///", alpha=0.18, lw=0, zorder=0))
        ax.plot([0, 0], [y - H / 2, y + H / 2], color=ORANGE, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.text(2.0, y, f"detected {k}/{N}", va="center", ha="left", fontsize=8,
                fontweight="bold", color=ORANGE, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    else:
        ax.barh(y, rate, height=H, color=col, edgecolor="none", zorder=2)
        ax.text(rate + 1.8 if rate < 85 else rate - 1.8, y, f"{k}/{N}", va="center",
                ha="left" if rate < 85 else "right", fontsize=7.5,
                color=INK2 if rate < 85 else "white", zorder=4)
    # left labels: condition (mono-ish) + what changed
    w = "bold" if brk else "normal"
    ax.text(-3.5, y + 0.15, cond, ha="right", va="center", fontsize=8,
            fontweight=w, color=ORANGE if brk else INK, transform=ax.transData)
    ax.text(-3.5, y - 0.20, what, ha="right", va="center", fontsize=6.6,
            color=ORANGE if brk else INK3, style="italic")
    # right column: did the answer change?
    if brk:
        ax.text(104, y + 0.15, "YES", ha="left", va="center", fontsize=8.5,
                fontweight="bold", color=ORANGE)
        ax.text(104, y - 0.20, f"{N}/{N}", ha="left", va="center", fontsize=6.6, color=ORANGE)
    else:
        ax.text(104, y, "no", ha="left", va="center", fontsize=8, color=INK3)

# column headers
ax.text(0, n - 0.35, "Divergence measure fires (fork rate, %)", ha="left", va="bottom",
        fontsize=8, color=INK, fontweight="bold")
ax.text(104, n - 0.35, "Answer\nchanged?", ha="left", va="bottom", fontsize=7, color=INK,
        fontweight="bold", linespacing=1.0)

ax.set_xlim(-2, 104)
ax.set_ylim(-0.6, n - 0.35)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="x", length=2.5, colors=INK2, pad=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK3)
ax.spines["bottom"].set_linewidth(0.6)
ax.set_xlabel("% of problems", color=INK2, labelpad=2)

# the claim, drawn on the nl_break row
ax.text(40, ys[-1], "the only edit that changes the answer\nis the only one it never detects",
        ha="left", va="center", fontsize=7.6, color=ORANGE, style="italic", zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

fig.text(0.02, 0.025,
         "Rewrites of 100 GSM-style problems vs. the original (nl_base). Blue = answer preserved; "
         "orange, hatched = answer changed by a one-digit edit.\n"
         "Fork rate: first-divergence-layer event vs. nl_base, forced-k=8 exploratory run. "
         "Answer-changed rate: gate G2 (greedy, thinking off);\n"
         "nl_break returned the old answer 0/100 times and its new answer at 67% accuracy (nl_base: 70%).",
         fontsize=6.2, color=INK3, ha="left", va="bottom", linespacing=1.25)

fig.savefig(OUT / "fig_headline.pdf")
fig.savefig(OUT / "fig_headline.png", dpi=200)
print("wrote", OUT / "fig_headline.pdf")
