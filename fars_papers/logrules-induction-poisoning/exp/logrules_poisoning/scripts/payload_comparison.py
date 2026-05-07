"""Payload comparison analysis for induction-stage poisoning.
Compares effectiveness of payloads D, E, F across datasets and attack budgets.
Produces payload_comparison.csv, grouped bar chart, and heatmap figures.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PAYLOAD_LABELS = {
    "D": "D (instruction-style anti-wildcard)",
    "E": "E (JSON priming anti-wildcard)",
    "F": "F (delimiter key-value anti-wildcard)",
}


def load_data():
    c0 = pd.read_csv(os.path.join(RESULTS_DIR, "c0_clean.csv"))
    c1 = pd.read_csv(os.path.join(RESULTS_DIR, "c1_poisoned.csv"))
    return c0, c1


def compute_drops(c0, c1):
    c0_lookup = c0.set_index(["dataset", "seed"])[
        ["test_PA", "test_FTA", "test_wildcard_ratio"]
    ].rename(columns={
        "test_PA": "c0_PA",
        "test_FTA": "c0_FTA",
        "test_wildcard_ratio": "c0_wildcard_ratio",
    })

    merged = c1.merge(c0_lookup, left_on=["dataset", "seed"], right_index=True)
    merged["PA_drop"] = merged["c0_PA"] - merged["test_PA"]
    merged["FTA_drop"] = merged["c0_FTA"] - merged["test_FTA"]
    return merged


def aggregate(merged):
    grouped = merged.groupby(["payload", "k", "dataset"]).agg(
        mean_PA_drop=("PA_drop", "mean"),
        std_PA_drop=("PA_drop", "std"),
        mean_FTA_drop=("FTA_drop", "mean"),
        mean_delta_wildcard=("delta_wildcard", "mean"),
        mean_template_disagreement=("template_disagreement", "mean"),
    ).reset_index()
    return grouped


def save_csv(agg):
    out = os.path.join(RESULTS_DIR, "payload_comparison.csv")
    agg.to_csv(out, index=False, float_format="%.6f")
    print(f"Saved: {out} ({len(agg)} rows)")


def plot_grouped_bar(agg):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    datasets = ["BGL", "Linux", "HDFS"]
    colors = {"D": "#d62728", "E": "#1f77b4", "F": "#2ca02c"}
    k_values = sorted(agg["k"].unique())
    bar_width = 0.25
    x = np.arange(len(k_values))

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        subset = agg[agg["dataset"] == ds]
        for i, payload in enumerate(["D", "E", "F"]):
            p_data = subset[subset["payload"] == payload].sort_values("k")
            vals = [p_data[p_data["k"] == k]["mean_PA_drop"].values[0]
                    if len(p_data[p_data["k"] == k]) > 0 else 0 for k in k_values]
            ax.bar(x + i * bar_width, vals, bar_width,
                   label=f"Payload {payload}", color=colors[payload], alpha=0.85)

        ax.set_title(f"{ds}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Poisoning budget k", fontsize=12)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        if ax_idx == 0:
            ax.set_ylabel("Mean PA Drop (C0 - C1)", fontsize=12)
        if ax_idx == 2:
            ax.legend(fontsize=10, loc="upper left")

    fig.suptitle("Payload Effectiveness: PA Drop by Payload Type and Budget",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "payload_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_heatmap(agg):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    k_values = sorted(agg["k"].unique())

    vmin = agg["mean_PA_drop"].min()
    vmax = agg["mean_PA_drop"].max()

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        subset = agg[agg["k"] == k].pivot(
            index="payload", columns="dataset", values="mean_PA_drop"
        )[["BGL", "Linux", "HDFS"]]
        subset = subset.reindex(["D", "E", "F"])

        sns.heatmap(subset, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    vmin=max(vmin, -0.1), vmax=max(vmax, 0.2),
                    ax=ax, cbar=(idx == 3),
                    cbar_kws={"label": "PA Drop"} if idx == 3 else {})
        ax.set_title(f"k = {k}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Payload" if idx == 0 else "")
        ax.set_xlabel("Dataset")

    fig.suptitle("PA Drop Heatmap: Payload x Dataset (per k)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "payload_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_overall_heatmap(agg):
    overall = agg.groupby(["payload", "dataset"])["mean_PA_drop"].mean().reset_index()
    pivot = overall.pivot(index="payload", columns="dataset", values="mean_PA_drop")
    pivot = pivot[["BGL", "Linux", "HDFS"]].reindex(["D", "E", "F"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "Mean PA Drop (avg over k)"})
    ax.set_title("Overall PA Drop: Payload x Dataset\n(averaged over k={1,3,5,7})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Payload")
    ax.set_xlabel("Dataset")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "payload_overall_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(agg):
    print("\n=== Payload Comparison Summary ===\n")

    for payload in ["D", "E", "F"]:
        p = agg[agg["payload"] == payload]
        mean_drop = p.groupby("dataset")["mean_PA_drop"].mean()
        overall = p["mean_PA_drop"].mean()
        print(f"Payload {payload} ({PAYLOAD_LABELS[payload]}):")
        for ds in ["BGL", "Linux", "HDFS"]:
            print(f"  {ds}: mean PA drop = {mean_drop[ds]:.4f}")
        print(f"  Overall mean PA drop = {overall:.4f}")

        best_row = p.loc[p["mean_PA_drop"].idxmax()]
        print(f"  Best config: k={int(best_row['k'])}, {best_row['dataset']}, "
              f"PA drop={best_row['mean_PA_drop']:.4f}")
        print()

    overall_rank = agg.groupby("payload")["mean_PA_drop"].mean().sort_values(ascending=False)
    print("Overall payload ranking (by mean PA drop across all k and datasets):")
    for rank, (payload, val) in enumerate(overall_rank.items(), 1):
        print(f"  {rank}. Payload {payload}: {val:.4f}")
    print()

    for k in sorted(agg["k"].unique()):
        subset = agg[agg["k"] == k]
        rank = subset.groupby("payload")["mean_PA_drop"].mean().sort_values(ascending=False)
        print(f"  k={k}: {' > '.join(f'{p}({v:.4f})' for p, v in rank.items())}")


def main():
    c0, c1 = load_data()
    merged = compute_drops(c0, c1)
    agg = aggregate(merged)
    save_csv(agg)
    plot_grouped_bar(agg)
    plot_heatmap(agg)
    plot_overall_heatmap(agg)
    print_summary(agg)

    summary = {}
    for payload in ["D", "E", "F"]:
        p = agg[agg["payload"] == payload]
        summary[payload] = {
            "overall_mean_PA_drop": round(p["mean_PA_drop"].mean(), 4),
            "per_dataset": {
                ds: round(p[p["dataset"] == ds]["mean_PA_drop"].mean(), 4)
                for ds in ["BGL", "Linux", "HDFS"]
            },
            "per_k": {
                int(k): round(p[p["k"] == k]["mean_PA_drop"].mean(), 4)
                for k in sorted(p["k"].unique())
            },
        }

    out = os.path.join(RESULTS_DIR, "payload_comparison_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
