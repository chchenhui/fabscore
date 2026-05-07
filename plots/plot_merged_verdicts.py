import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from plot_stacked_bar import VERDICT_COUNTS


# Standardized sizes for high-impact presentation
plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})


CATEGORIES = [
    "Verified",
    "Unverifiable",
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
]

UNIFORM_COLOR = "#9ecae1"
COLORS = {category: UNIFORM_COLOR for category in CATEGORIES}

def build_merged_counts():
    merged_counts = {category: 0 for category in CATEGORIES}

    for counts in VERDICT_COUNTS.values():
        merged_counts["Verified"] += counts.get("Verified", 0)
        merged_counts["Unverifiable"] += (
            counts.get("Insufficient Evidence", 0)
            + counts.get("No Code Files", 0)
        )
        merged_counts["Data Fabrication"] += counts.get("Data Fabrication", 0)
        merged_counts["Experiment Fabrication"] += counts.get("Experiment Fabrication", 0)
        merged_counts["Result Fabrication"] += counts.get("Result Fabrication", 0)

    return merged_counts


def main():
    merged_counts = build_merged_counts()
    total_n = sum(merged_counts.values())

    labels = CATEGORIES
    counts = np.array([merged_counts[label] for label in labels])
    rates = counts / total_n
    
    # Increased bar width (0.6) and decreased gap (0.3) for a bolder look
    bar_width = 0.6
    bar_gap = 0.3
    x = np.arange(len(labels)) * (bar_width + bar_gap)

    # Set figsize to exactly 10:8.4 ratio as requested
    fig, ax = plt.subplots(figsize=(10.0, 8.4), dpi=200)

    bars = ax.bar(
        x,
        rates,
        width=bar_width,
        color=[COLORS[label] for label in labels],
        edgecolor="#000000",
        linewidth=1.5,
        zorder=3,
        alpha=0.9,
    )

    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.01,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=24,
            fontweight="bold",
            color="black",
            zorder=5,
            clip_on=False,
        )

    ax.set_ylabel("") 
    ax.set_xlabel("") 
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=24)
    
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(0, max(rates) + 0.15)

    yticks = np.linspace(0, 0.6, 4)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t * 100)}%" for t in yticks], fontsize=24)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=1.0,
        color="#d8d8d8",
        alpha=0.9,
        zorder=0,
    )
    ax.xaxis.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", pad=10)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "merged_verdict_distribution.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved 10:8.4 aspect bolder-bar merged verdict plot to {out}")


if __name__ == "__main__":
    main()