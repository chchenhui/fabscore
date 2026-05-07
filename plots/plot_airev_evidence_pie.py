"""
plot_airev_evidence_pie.py

Pie chart of AI review evidence classification categories.
Output: plots/airev_evidence_pie.pdf  (and .png)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
labels = [
    "Intra-paper\nContradictions",
    "Mathematical\nImplausibility",
    "Lack of Quantitative\nEvidence",
    "Code-to-Text\nMismatch",
]
counts = [31, 5, 11, 2]
colors = ["#8FC7EA", "#A6DBB0", "#FFC98F", "#EEA09B"]

total = sum(counts)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))


wedges, texts, autotexts = ax.pie(
    counts,
    labels=None,
    colors=colors,
    autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
    pctdistance=0.72,
    startangle=140,
    radius=1.0,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)

for at in autotexts:
    at.set_fontsize(18)
    at.set_color("black")
    at.set_fontweight("bold")

# Draw the small slice (4.1%) label outside with a leader line
small_idx = counts.index(min(counts))  # index 3 (Code-to-Text Mismatch, n=2)
wedge = wedges[small_idx]
angle = (wedge.theta1 + wedge.theta2) / 2
import numpy as np
r = 0.8
x_edge  = r * np.cos(np.deg2rad(angle))
y_edge  = r * np.sin(np.deg2rad(angle))
x_outer = r * 1.52 * np.cos(np.deg2rad(angle))
y_outer = r * 1.32 * np.sin(np.deg2rad(angle)) 
ax.annotate(
    "4.1%",
    xy=(x_edge, y_edge),
    xytext=(x_outer, y_outer),
    fontsize=18, fontweight="bold", color="black",
    ha="center", va="center",
)

ax.set_title("")

# Legend outside
legend_handles = [
    mpatches.Patch(color=colors[i], label=labels[i].replace(chr(10), ' '))
    for i in range(len(labels))
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(1.5, 1.15),
    ncol=1,
    fontsize=15,
    frameon=False,
)

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"airev_evidence_pie.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out}")

plt.show()
