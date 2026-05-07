"""Plot grouped bar chart comparing Misaligned, Shuffled-Pair, Public-Anchor,
and In-Domain adapter nDCG@10 across all 4 BEIR datasets. Computes and prints
the decision-rule check: M(public-anchor) - M(shuffled) >= 0.02.
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
DATASET_LABELS = ["SciFact", "TREC-COVID", "FiQA", "ArguAna"]
THRESHOLD = 0.02


def main():
    with open(RESULTS_DIR / "misaligned.json") as f:
        misaligned = json.load(f)
    with open(RESULTS_DIR / "shuffled_pair.json") as f:
        shuffled = json.load(f)
    with open(RESULTS_DIR / "public_anchor.json") as f:
        public_anchor = json.load(f)
    with open(RESULTS_DIR / "in_domain.json") as f:
        in_domain = json.load(f)

    methods = ["Misaligned", "Shuffled-Pair", "Public-Anchor", "In-Domain"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    ndcg_values = {m: [] for m in methods}
    ndcg_stds = {m: [] for m in methods}

    print("=" * 70)
    print("NULL CONTROL DECISION RULE CHECK")
    print("=" * 70)
    print(f"{'Dataset':<15} {'M(pub-anchor)':<15} {'M(shuffled)':<15} {'Delta':<10} {'Pass?':<8}")
    print("-" * 63)

    all_pass = True
    for ds in DATASETS:
        m_mis = misaligned[ds]["nDCG@10"]
        m_shuf = shuffled[ds]["nDCG@10_mean"]
        m_pub = public_anchor[ds]["nDCG@10_mean"]
        m_ind = in_domain[ds]["nDCG@10_mean"]

        ndcg_values["Misaligned"].append(m_mis)
        ndcg_stds["Misaligned"].append(0)
        ndcg_values["Shuffled-Pair"].append(m_shuf)
        ndcg_stds["Shuffled-Pair"].append(shuffled[ds]["nDCG@10_std"])
        ndcg_values["Public-Anchor"].append(m_pub)
        ndcg_stds["Public-Anchor"].append(public_anchor[ds]["nDCG@10_std"])
        ndcg_values["In-Domain"].append(m_ind)
        ndcg_stds["In-Domain"].append(in_domain[ds]["nDCG@10_std"])

        delta = m_pub - m_shuf
        passes = delta >= THRESHOLD
        if not passes:
            all_pass = False
        print(f"{ds:<15} {m_pub:<15.5f} {m_shuf:<15.5f} {delta:<10.5f} {'YES' if passes else 'NO':<8}")

    print("-" * 63)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'} (threshold >= {THRESHOLD})")
    print()

    x = np.arange(len(DATASETS))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax.bar(
            x + offsets[i] * width,
            ndcg_values[method],
            width,
            label=method,
            color=color,
            yerr=ndcg_stds[method],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Dataset", fontsize=13)
    ax.set_ylabel("nDCG@10", fontsize=13)
    ax.set_title("Null Control Comparison: nDCG@10 Across BEIR Datasets", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "null_control_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
