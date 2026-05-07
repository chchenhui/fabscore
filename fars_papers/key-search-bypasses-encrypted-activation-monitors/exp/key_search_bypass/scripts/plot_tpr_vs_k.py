# Generate TPR vs K plot for key-search attack results.
# Line plot with log-scale x-axis, two curves (FPR=1e-3, FPR=1e-4),
# error bands, and reference lines for unencrypted and encrypted K=1 baselines.

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJ_DIR / "key_search_bypass" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "key_search_attack.json") as f:
        attack_results = json.load(f)
    with open(RESULTS_DIR / "baseline_unencrypted_monitor.json") as f:
        baseline = json.load(f)

    K_values = attack_results["K_values"]
    agg = attack_results["aggregated"]

    unenc_tpr_1e3 = baseline["aggregated"]["fpr_0.001_tpr_mean"]
    unenc_tpr_1e4 = baseline["aggregated"]["fpr_0.0001_tpr_mean"]

    tpr_1e3_mean = [agg[str(k)]["fpr_0.001"]["tpr_mean"] for k in K_values]
    tpr_1e3_std = [agg[str(k)]["fpr_0.001"]["tpr_std"] for k in K_values]
    tpr_1e4_mean = [agg[str(k)]["fpr_0.0001"]["tpr_mean"] for k in K_values]
    tpr_1e4_std = [agg[str(k)]["fpr_0.0001"]["tpr_std"] for k in K_values]

    enc_k1_tpr_1e3 = tpr_1e3_mean[0]
    enc_k1_tpr_1e4 = tpr_1e4_mean[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    K_arr = np.array(K_values)
    tpr_1e3_mean_arr = np.array(tpr_1e3_mean)
    tpr_1e3_std_arr = np.array(tpr_1e3_std)
    tpr_1e4_mean_arr = np.array(tpr_1e4_mean)
    tpr_1e4_std_arr = np.array(tpr_1e4_std)

    ax.plot(K_arr, tpr_1e3_mean_arr, "o-", color="tab:blue", label="TPR@FPR=1e-3", linewidth=2, markersize=6)
    ax.fill_between(K_arr, tpr_1e3_mean_arr - tpr_1e3_std_arr, tpr_1e3_mean_arr + tpr_1e3_std_arr,
                    alpha=0.2, color="tab:blue")

    ax.plot(K_arr, tpr_1e4_mean_arr, "s-", color="tab:red", label="TPR@FPR=1e-4", linewidth=2, markersize=6)
    ax.fill_between(K_arr, tpr_1e4_mean_arr - tpr_1e4_std_arr, tpr_1e4_mean_arr + tpr_1e4_std_arr,
                    alpha=0.2, color="tab:red")

    ax.axhline(y=unenc_tpr_1e3, color="tab:blue", linestyle="--", alpha=0.5,
               label=f"Unencrypted TPR@FPR=1e-3 ({unenc_tpr_1e3:.3f})")
    ax.axhline(y=unenc_tpr_1e4, color="tab:red", linestyle="--", alpha=0.5,
               label=f"Unencrypted TPR@FPR=1e-4 ({unenc_tpr_1e4:.3f})")

    ax.axhline(y=enc_k1_tpr_1e3, color="tab:blue", linestyle=":", alpha=0.4,
               label=f"Encrypted K=1 @FPR=1e-3 ({enc_k1_tpr_1e3:.3f})")
    ax.axhline(y=enc_k1_tpr_1e4, color="tab:red", linestyle=":", alpha=0.4,
               label=f"Encrypted K=1 @FPR=1e-4 ({enc_k1_tpr_1e4:.3f})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(K_values)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel("Key-Search Budget K", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title("Key-Search Attack: Monitor TPR vs Search Budget K", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "tpr_vs_K.pdf"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "tpr_vs_K.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    print("\n" + "=" * 70)
    print("Summary Table: TPR@FPR vs Key-Search Budget K")
    print("=" * 70)
    print(f"{'K':>4} | {'TPR@FPR=1e-3':>20} | {'TPR@FPR=1e-4':>20}")
    print("-" * 50)
    for i, k in enumerate(K_values):
        print(f"{k:>4} | {tpr_1e3_mean[i]:.4f} +/- {tpr_1e3_std[i]:.4f}    | "
              f"{tpr_1e4_mean[i]:.4f} +/- {tpr_1e4_std[i]:.4f}")
    print("-" * 50)
    print(f"Unencrypted baseline: TPR@FPR=1e-3 = {unenc_tpr_1e3:.4f}, TPR@FPR=1e-4 = {unenc_tpr_1e4:.4f}")
    print(f"Encrypted K=1:       TPR@FPR=1e-3 = {enc_k1_tpr_1e3:.4f}, TPR@FPR=1e-4 = {enc_k1_tpr_1e4:.4f}")

    drop_1e3 = enc_k1_tpr_1e3 - tpr_1e3_mean[K_values.index(32)]
    drop_1e4 = enc_k1_tpr_1e4 - tpr_1e4_mean[K_values.index(32)]
    print(f"\nTPR drop from K=1 to K=32:")
    print(f"  @FPR=1e-3: {drop_1e3:.4f} ({drop_1e3*100:.1f} percentage points)")
    print(f"  @FPR=1e-4: {drop_1e4:.4f} ({drop_1e4*100:.1f} percentage points)")


if __name__ == "__main__":
    main()
