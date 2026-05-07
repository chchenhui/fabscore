"""Plot size ablation: nDCG@10 vs N_p with reference baselines.

Reads pada/results/size_ablation.json, misaligned.json, in_domain.json.
Produces a line plot with log-scale x-axis showing how retrieval quality
scales with the number of public anchor pairs. Includes recovery ratio
on a secondary y-axis.

Usage:
  python -m pada.scripts.plot_size_ablation
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASETS = ["scifact", "trec-covid", "fiqa", "arguana"]
DATASET_LABELS = {
    "scifact": "SciFact",
    "trec-covid": "TREC-COVID",
    "fiqa": "FiQA",
    "arguana": "ArguAna",
}
COLORS = {
    "scifact": "#1f77b4",
    "trec-covid": "#ff7f0e",
    "fiqa": "#2ca02c",
    "arguana": "#d62728",
}


def main():
    with open(RESULTS_DIR / "size_ablation.json") as f:
        ablation = json.load(f)
    with open(RESULTS_DIR / "misaligned.json") as f:
        misaligned = json.load(f)
    with open(RESULTS_DIR / "in_domain.json") as f:
        in_domain = json.load(f)

    sizes = sorted(int(k) for k in ablation.keys())

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for ds_name in DATASETS:
        ndcg_values = [ablation[str(s)][ds_name]["nDCG@10"] for s in sizes]
        ax1.plot(sizes, ndcg_values, "o-", color=COLORS[ds_name],
                 label=DATASET_LABELS[ds_name], linewidth=2, markersize=6)

        mis_val = misaligned[ds_name]["nDCG@10"]
        ind_val = in_domain[ds_name]["nDCG@10_mean"]
        ax1.axhline(y=mis_val, color=COLORS[ds_name], linestyle=":", alpha=0.4, linewidth=1)
        ax1.axhline(y=ind_val, color=COLORS[ds_name], linestyle="--", alpha=0.4, linewidth=1)

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Public Anchor Pairs ($N_p$)", fontsize=12)
    ax1.set_ylabel("nDCG@10", fontsize=12)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.tick_params(axis="x", which="minor", bottom=False)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    for ds_name in DATASETS:
        mis_val = misaligned[ds_name]["nDCG@10"]
        ind_val = in_domain[ds_name]["nDCG@10_mean"]
        headroom = ind_val - mis_val
        if headroom < 1e-9:
            continue
        rho_values = [(ablation[str(s)][ds_name]["nDCG@10"] - mis_val) / headroom for s in sizes]
        ax2.plot(sizes, rho_values, "s--", color=COLORS[ds_name], alpha=0.35,
                 markersize=4, linewidth=1)

    ax2.set_ylabel("Recovery Ratio ($\\rho$)", fontsize=12, alpha=0.6)
    ax2.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)

    from matplotlib.lines import Line2D
    legend_elements = []
    for ds_name in DATASETS:
        legend_elements.append(Line2D([0], [0], color=COLORS[ds_name], marker="o",
                                       label=DATASET_LABELS[ds_name], linewidth=2, markersize=6))
    legend_elements.append(Line2D([0], [0], color="gray", linestyle=":", alpha=0.6,
                                   label="Misaligned baseline"))
    legend_elements.append(Line2D([0], [0], color="gray", linestyle="--", alpha=0.6,
                                   label="In-domain adapter"))
    legend_elements.append(Line2D([0], [0], color="gray", linestyle="--", marker="s",
                                   alpha=0.4, markersize=4, label="Recovery ratio $\\rho$"))

    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax1.set_title("Public-Anchor Adapter: Effect of Anchor Set Size ($N_p$)", fontsize=13)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "anchor_size_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
