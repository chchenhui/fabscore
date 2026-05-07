"""Visualizations for per-category success rate breakdown and inference timing comparison.

Generates two PDFs:
  1) Grouped bar chart of success rate by category for conditions A/B/C
  2) Bar chart comparing average inference time per instance across conditions

Usage:
  python -m bfcl_cfg_diffusion.analysis.plot_category_breakdown
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = EXP_ROOT / "bfcl_cfg_diffusion" / "scores"
INPUT_PATH = SCORES_DIR / "category_breakdown_results.json"

CONDITION_ORDER = ["unconstrained", "best_of_2", "lave_cfg_v3"]
CONDITION_LABELS = {
    "unconstrained": "(A) Unconstrained",
    "best_of_2": "(B) Best-of-2",
    "lave_cfg_v3": "(C) LAVE CFG",
}
COND_COLORS = ["#1976D2", "#7B1FA2", "#388E3C"]

CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance",
]
CATEGORY_DISPLAY = {
    "simple_python": "Simple\n(Python)",
    "simple_java": "Simple\n(Java)",
    "simple_javascript": "Simple\n(JS)",
    "multiple": "Multiple",
    "parallel": "Parallel",
    "parallel_multiple": "Parallel\nMultiple",
    "irrelevance": "Irrelevance",
}


def load_results():
    with open(INPUT_PATH) as f:
        return json.load(f)


def plot_category_success(data, output_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    cats = CATEGORIES + ["overall"]
    cat_labels = [CATEGORY_DISPLAY.get(c, c) for c in CATEGORIES] + ["Overall"]
    n_cats = len(cats)
    n_conds = len(CONDITION_ORDER)
    width = 0.25
    x = np.arange(n_cats)

    per_cat = data["per_category_success"]

    for ci, cond in enumerate(CONDITION_ORDER):
        means = []
        stds = []
        for cat in cats:
            d = per_cat[cond].get(cat, {"mean": 0, "std": 0})
            means.append(d["mean"] * 100)
            stds.append(d["std"] * 100)
        offset = (ci - (n_conds - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=CONDITION_LABELS[cond], color=COND_COLORS[ci],
                      edgecolor="white", linewidth=0.5, capsize=2, alpha=0.85)
        for bar, m in zip(bars, means):
            if m >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.axvline(x=len(CATEGORIES) - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("BFCL Success Rate by Category and Condition", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 85)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved category breakdown to {output_path}")


def plot_timing(data, output_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    timing = data["timing"]
    means = [timing[c]["avg_time_per_instance"] for c in CONDITION_ORDER]
    stds = [timing[c]["std_time_per_instance"] for c in CONDITION_ORDER]
    overheads = [timing[c]["relative_overhead"] for c in CONDITION_ORDER]

    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(x, means, 0.5, yerr=stds, color=COND_COLORS,
                  edgecolor="white", linewidth=0.5, capsize=4, alpha=0.85)

    for i, (bar, m, oh) in enumerate(zip(bars, means, overheads)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.4,
                f"{m:.2f}s\n({oh:.2f}x)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Avg Inference Time per Instance (s)", fontsize=11)
    ax.set_title("Inference Time Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timing comparison to {output_path}")


def main():
    data = load_results()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    plot_category_success(data, SCORES_DIR / "category_breakdown.pdf")
    plot_timing(data, SCORES_DIR / "timing_comparison.pdf")


if __name__ == "__main__":
    main()
