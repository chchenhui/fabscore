"""Compile lambda sweep results (lambda in {0, 0.1, 0.2, 0.5}) with N=2 releases.
Collects attack + STS12 results from main experiment files and lambda_sweep/ dir.
Produces CSV, two trade-off plots, and prints a summary table."""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "lambda_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    0.0: {
        "attack": BASE_DIR / "anisotropic_attack_results.json",
        "sts12": BASE_DIR / "anisotropic_results.json",
    },
    0.1: {
        "attack": SWEEP_DIR / "smoothed_attack_lam0.10_results.json",
        "sts12": SWEEP_DIR / "smoothed_sts12_lam0.10_results.json",
    },
    0.2: {
        "attack": BASE_DIR / "smoothed_attack_results.json",
        "sts12": BASE_DIR / "smoothed_results.json",
    },
    0.5: {
        "attack": SWEEP_DIR / "smoothed_attack_lam0.50_results.json",
        "sts12": SWEEP_DIR / "smoothed_sts12_lam0.50_results.json",
    },
}


def load_and_validate():
    rows = []
    missing = []
    for lam, paths in sorted(SOURCES.items()):
        for key, p in paths.items():
            if not p.exists():
                missing.append(f"lambda={lam}, {key}: {p}")

    if missing:
        print("ERROR: Missing result files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    for lam, paths in sorted(SOURCES.items()):
        with open(paths["attack"]) as f:
            atk = json.load(f)
        with open(paths["sts12"]) as f:
            sts_raw = json.load(f)
        sts = sts_raw.get("sts12", sts_raw)

        rows.append({
            "lambda": lam,
            "accuracy_mean": atk["accuracy_mean"],
            "accuracy_std": atk["accuracy_std"],
            "macro_f1_mean": atk["macro_f1_mean"],
            "macro_f1_std": atk["macro_f1_std"],
            "sts12_pearson_mean": sts["noisy_pearson_mean"],
            "sts12_pearson_std": sts["noisy_pearson_std"],
        })

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame):
    out = SWEEP_DIR / "lambda_sweep_results.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"CSV saved to {out}")
    return out


def plot_accuracy_vs_lambda(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        df["lambda"], df["accuracy_mean"], yerr=df["accuracy_std"],
        marker="o", capsize=4, linewidth=2, markersize=8, color="#2196F3",
        label="Concept-ID Accuracy"
    )
    ax.axhline(y=0.20, color="gray", linestyle="--", linewidth=1, label="Chance (0.20)")
    ax.set_xlabel(r"Smoothing Parameter $\lambda$", fontsize=12)
    ax.set_ylabel("Concept-ID Accuracy", fontsize=12)
    ax.set_title(r"Concept-ID Accuracy vs $\lambda$ (N=2 releases)", fontsize=13)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(df["lambda"].values)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = SWEEP_DIR / "accuracy_vs_lambda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot 1 saved to {out}")


def plot_dual_axis(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_acc = "#2196F3"
    ax1.errorbar(
        df["lambda"], df["accuracy_mean"], yerr=df["accuracy_std"],
        marker="o", capsize=4, linewidth=2, markersize=8, color=color_acc,
        label="Concept-ID Accuracy"
    )
    ax1.axhline(y=0.20, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Chance (0.20)")
    ax1.set_xlabel(r"Smoothing Parameter $\lambda$", fontsize=12)
    ax1.set_ylabel("Concept-ID Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.05, 1.10)
    ax1.set_xticks(df["lambda"].values)

    ax2 = ax1.twinx()
    color_sts = "#FF5722"
    ax2.errorbar(
        df["lambda"], df["sts12_pearson_mean"], yerr=df["sts12_pearson_std"],
        marker="s", capsize=4, linewidth=2, markersize=8, color=color_sts,
        label="STS12 Pearson"
    )
    ax2.set_ylabel("STS12 Pearson Correlation", fontsize=12, color=color_sts)
    ax2.tick_params(axis="y", labelcolor=color_sts)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    ax1.set_title(r"Privacy Leakage vs Utility Trade-Off ($\lambda$ sweep, N=2)", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out = SWEEP_DIR / "tradeoff_dual_axis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot 2 saved to {out}")


def main():
    print("=== Compiling Lambda Sweep Results ===\n")
    df = load_and_validate()
    print(df.to_string(index=False))
    print()
    save_csv(df)
    plot_accuracy_vs_lambda(df)
    plot_dual_axis(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
