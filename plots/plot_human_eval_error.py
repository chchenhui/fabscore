"""
plot_human_eval_error.py

Visualize error analysis for ClaudeCode and Codex human evaluation CSVs.
Outputs: plots/human_eval_error_analysis.pdf and .png
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
FILES = {
    "Claude Code": "../human_reviews/FabScore Human Eval - ClaudeCode.csv",
    "Codex":       "../human_reviews/FabScore Human Eval - Codex.csv",
}

ALL_LABELS = ["Verified", "Data Fabrication", "Experiment Fabrication",
              "Result Fabrication", "Insufficient Evidence", "No Code Files"]
LABEL_SHORT = {
    "Verified":               "Verified",
    "Data Fabrication":       "Data Fab.",
    "Experiment Fabrication": "Exp. Fab.",
    "Result Fabrication":     "Result Fab.",
    "Insufficient Evidence":  "Insuf. Evid.",
    "No Code Files":          "No Code",
}
MODEL_COLORS = {"Claude Code": "#4C72B0", "Codex": "#DD8452"}

data = {}
for model, fpath in FILES.items():
    with open(fpath) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    disagree = [r for r in rows if r["Do you agree? (Yes/No)"].strip().lower() == "no"]
    # confusion counts: model_verdict -> human_verdict
    matrix = defaultdict(lambda: defaultdict(int))
    for r in disagree:
        cv = r["Verdict"].strip()
        yv = r["Your Verdict"].strip()
        matrix[cv][yv] += 1
    data[model] = {"rows": rows, "total": total, "disagree": disagree, "matrix": matrix}

# ---------------------------------------------------------------------------
# Figure layout: 1 row of 3 panels
#   [Panel A] Disagreement rate bar chart (2 bars)
#   [Panel B] Confusion heatmap – Claude Code
#   [Panel C] Confusion heatmap – Codex
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 5.5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.6, 1.6], wspace=0.45)
ax_bar  = fig.add_subplot(gs[0])
ax_cc   = fig.add_subplot(gs[1])
ax_cx   = fig.add_subplot(gs[2])

# ---- Panel A: Disagreement rate ----
models = list(FILES.keys())
dis_rates = [len(data[m]["disagree"]) / data[m]["total"] * 100 for m in models]
bars = ax_bar.bar([0, 0.6], dis_rates,
                  color=[MODEL_COLORS[m] for m in models],
                  width=0.28, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, dis_rates):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax_bar.set_ylabel("Disagreement Rate (%)", fontsize=13)
ax_bar.set_title("(a) Overall Disagreement", fontsize=13, fontweight="bold", pad=8)
ax_bar.set_ylim(0, max(dis_rates) * 1.3)
ax_bar.set_xticks([0, 0.6])
ax_bar.set_xticklabels(models, fontsize=13)
ax_bar.tick_params(axis="y", labelsize=13)
for spine in ["top", "right"]:
    ax_bar.spines[spine].set_visible(False)

# totals annotation
for i, (x, m) in enumerate(zip([0, 0.6], models)):
    ax_bar.text(x, -3.5, f"n={data[m]['total']}", ha="center", fontsize=13, color="#555555")

# ---- Helper: draw confusion heatmap ----
def draw_confusion(ax, model, title):
    mat = data[model]["matrix"]
    # model predicted only from these 3 categories
    row_labels = ["Data Fab.", "Exp. Fab.", "Result Fab."]
    row_keys   = ["Data Fabrication", "Experiment Fabrication", "Result Fabrication"]
    # human corrected to these categories (only those that appear)
    all_human = set()
    for cv in row_keys:
        for yv in mat[cv]:
            all_human.add(yv)
    col_keys = [k for k in ALL_LABELS if k in all_human]
    col_labels = [LABEL_SHORT[k] for k in col_keys]

    Z = np.zeros((len(row_keys), len(col_keys)), dtype=int)
    for i, rv in enumerate(row_keys):
        for j, cv in enumerate(col_keys):
            Z[i, j] = mat[rv][cv]

    im = ax.imshow(Z, cmap="Blues", aspect="auto", vmin=0)

    # cell annotations
    vmax = Z.max() if Z.max() > 0 else 1
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            val = Z[i, j]
            if val == 0:
                continue
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=12, rotation=30, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlabel("Human Verdict", fontsize=14, labelpad=4)
    ax.set_ylabel("Generated Verdict", fontsize=14, labelpad=4)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    # grid lines
    for x in np.arange(-0.5, len(col_keys), 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, len(row_keys), 1):
        ax.axhline(y, color="white", linewidth=1.5)

draw_confusion(ax_cc, "Claude Code", "(b) Claude Code Error Matrix")
draw_confusion(ax_cx, "Codex",       "(c) Codex Error Matrix")

fig.suptitle("Human Evaluation Error Analysis", fontsize=15, fontweight="bold", y=1.02)

for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"human_eval_error.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {out}")

plt.show()
