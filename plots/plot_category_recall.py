"""
plot_category_recall.py

Bar chart: claim-level recall of AI reviews per FabScore fabrication category.
For each category (Data / Experiment / Result Fabrication) and Overall,
shows the fraction of FabScore fabrications that AI reviews caught.

Output: plots/category_recall.pdf  (and .png)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data (from analyze_airev_coverage.py output)
# ---------------------------------------------------------------------------
categories = [
    "Data\nFabrication",
    "Experiment\nFabrication",
    "Result\nFabrication",
    "Overall",
]
caught = [14,  90,  38, 142]
total  = [27, 883, 117, 1027]
recall = [c / t for c, t in zip(caught, total)]

colors = ["#9ecae1"] * len(categories)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.6
bar_gap   = 0.2
x = np.arange(len(categories)) * (bar_width + bar_gap)
bars  = ax.bar(x, [r * 100 for r in recall], width=bar_width,
               color=colors, edgecolor="#000000", linewidth=1.2)

# Value labels: bold % inside bar
for bar, c, t, r in zip(bars, caught, total, recall):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() / 2,
        f"{r*100:.1f}%",
        ha="center", va="center", fontsize=17, fontweight="bold", color="black",
    )

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=15)
ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
ax.set_ylabel("Fabrications Covered by AI Reviews", fontsize=17)
ax.set_ylim(0, max(r * 100 for r in recall) * 1.35)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.tick_params(axis="y", labelsize=16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"category_recall.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.show()
