"""Visualizations for failure taxonomy analysis.

Produces two PDF plots:
  1) Stacked bar chart of error categories per condition (A, B, C)
  2) Grouped bar chart of parse failure rate across BFCL category types per condition

Usage:
  python -m bfcl_cfg_diffusion.analysis.plot_failure_taxonomy \
      --input bfcl_cfg_diffusion/scores/failure_taxonomy_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = EXP_ROOT / "bfcl_cfg_diffusion" / "scores"

CONDITION_ORDER = ["unconstrained", "best_of_2", "lave_cfg_v3"]
CONDITION_SHORT = {
    "unconstrained": "(A) Unconstrained",
    "best_of_2": "(B) Best-of-2",
    "lave_cfg_v3": "(C) LAVE CFG",
}

LABELS = ["success", "parse_failure", "wrong_function", "wrong_arguments"]
LABEL_DISPLAY = {
    "success": "Success",
    "parse_failure": "Parse Failure",
    "wrong_function": "Wrong Function",
    "wrong_arguments": "Wrong Arguments",
}
LABEL_COLORS = {
    "success": "#4CAF50",
    "parse_failure": "#E53935",
    "wrong_function": "#FB8C00",
    "wrong_arguments": "#FDD835",
}

CATEGORY_GROUPS = {
    "Simple": ["simple_python", "simple_java", "simple_javascript"],
    "Multiple": ["multiple"],
    "Parallel": ["parallel"],
    "Parallel Multiple": ["parallel_multiple"],
    "Irrelevance": ["irrelevance"],
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_stacked_bar(results, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(CONDITION_ORDER))
    width = 0.5

    bottoms = np.zeros(len(CONDITION_ORDER))
    for label in LABELS:
        vals = []
        errs = []
        for cond in CONDITION_ORDER:
            r = results[cond]["mean_overall"][label]
            vals.append(r["pct_mean"])
            errs.append(r["pct_std"])
        vals = np.array(vals)
        ax.bar(x, vals, width, bottom=bottoms,
               label=LABEL_DISPLAY[label], color=LABEL_COLORS[label],
               edgecolor="white", linewidth=0.5)

        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 4:
                ax.text(x[i], b + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_SHORT[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Percentage of Examples (%)", fontsize=11)
    ax.set_title("Failure Taxonomy by Condition", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stacked bar chart to {output_path}")


def plot_parse_fail_by_category(results, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    group_names = list(CATEGORY_GROUPS.keys())
    n_groups = len(group_names)
    n_conds = len(CONDITION_ORDER)
    width = 0.22
    x = np.arange(n_groups)

    cond_colors = ["#1976D2", "#7B1FA2", "#388E3C"]

    for ci, cond in enumerate(CONDITION_ORDER):
        vals = []
        errs = []
        for gname, subcats in CATEGORY_GROUPS.items():
            pf_counts = []
            total_counts = []
            for seed_str, seed_data in results[cond]["per_seed"].items():
                seed_pf = 0
                seed_total = 0
                for subcat in subcats:
                    cat_data = seed_data.get("per_category", {}).get(subcat, {})
                    seed_pf += cat_data.get("parse_failure", {}).get("count", 0)
                    seed_total += cat_data.get("total", 0)
                if seed_total > 0:
                    pf_counts.append(seed_pf / seed_total * 100)
                else:
                    pf_counts.append(0)
            vals.append(np.mean(pf_counts))
            errs.append(np.std(pf_counts))

        offset = (ci - (n_conds - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      label=CONDITION_SHORT[cond], color=cond_colors[ci],
                      edgecolor="white", linewidth=0.5, capsize=3, alpha=0.85)

        for bar, v in zip(bars, vals):
            if v >= 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylabel("Parse Failure Rate (%)", fontsize=11)
    ax.set_title("Parse Failure Rate by BFCL Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parse failure by category chart to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SCORES_DIR / "failure_taxonomy_results.json"))
    args = parser.parse_args()

    results = load_results(args.input)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    plot_stacked_bar(results, SCORES_DIR / "failure_taxonomy_stacked.pdf")
    plot_parse_fail_by_category(results, SCORES_DIR / "parse_fail_by_category.pdf")


if __name__ == "__main__":
    main()
