# Generate comparison plots for the diversity ablation study (no GPU needed).
# Loads key_search_attack_diverse.json and key_search_no_div.json, overlays
# TPR@FPR vs K curves and prints a summary table.
import sys
import json
import numpy as np
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = PROJ_DIR / "key_search_bypass" / "results"
    fig_dir = PROJ_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    full_path = results_dir / "key_search_attack_diverse.json"
    no_div_path = results_dir / "key_search_no_div.json"

    with open(full_path) as f:
        full_data = json.load(f)
    with open(no_div_path) as f:
        no_div_data = json.load(f)

    div_stats_path = results_dir / "diversity_comparison_stats.json"
    div_stats = {}
    if div_stats_path.exists():
        with open(div_stats_path) as f:
            div_stats = json.load(f)

    common_ks = [1, 2, 4, 8, 16, 32, 64]

    full_tpr_1e3 = []
    full_tpr_1e4 = []
    full_tpr_1e3_std = []
    full_tpr_1e4_std = []
    for k in common_ks:
        a = full_data["random_attack"]["aggregated"][str(k)]
        full_tpr_1e3.append(a["fpr_0.001"]["tpr_mean"])
        full_tpr_1e3_std.append(a["fpr_0.001"]["tpr_std"])
        full_tpr_1e4.append(a["fpr_0.0001"]["tpr_mean"])
        full_tpr_1e4_std.append(a["fpr_0.0001"]["tpr_std"])

    no_div_tpr_1e3 = []
    no_div_tpr_1e4 = []
    no_div_tpr_1e3_std = []
    no_div_tpr_1e4_std = []
    for k in common_ks:
        a = no_div_data["random_attack"]["aggregated"][str(k)]
        no_div_tpr_1e3.append(a["fpr_0.001"]["tpr_mean"])
        no_div_tpr_1e3_std.append(a["fpr_0.001"]["tpr_std"])
        no_div_tpr_1e4.append(a["fpr_0.0001"]["tpr_mean"])
        no_div_tpr_1e4_std.append(a["fpr_0.0001"]["tpr_std"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(common_ks))
    ax1.errorbar(x, full_tpr_1e3, yerr=full_tpr_1e3_std, marker="o", linewidth=2, capsize=4, label="Full (λ₂=0.5)")
    ax1.errorbar(x, no_div_tpr_1e3, yerr=no_div_tpr_1e3_std, marker="s", linewidth=2, capsize=4, label="No diversity (λ₂=0)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in common_ks])
    ax1.set_xlabel("K (number of keys)", fontsize=12)
    ax1.set_ylabel("TPR", fontsize=12)
    ax1.set_title("TPR @ FPR=1e-3 vs K", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(x, full_tpr_1e4, yerr=full_tpr_1e4_std, marker="o", linewidth=2, capsize=4, label="Full (λ₂=0.5)")
    ax2.errorbar(x, no_div_tpr_1e4, yerr=no_div_tpr_1e4_std, marker="s", linewidth=2, capsize=4, label="No diversity (λ₂=0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in common_ks])
    ax2.set_xlabel("K (number of keys)", fontsize=12)
    ax2.set_ylabel("TPR", fontsize=12)
    ax2.set_title("TPR @ FPR=1e-4 vs K", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Key-Search Attack: Full Diversity vs No Diversity Encryptor", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = fig_dir / "ablation_diversity_tpr_curve.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved TPR vs K comparison to {fig_path}")
    plt.close()

    full_enc = full_data.get("encryptor", {})
    no_div_enc = no_div_data.get("encryptor", {})

    full_div_l2 = div_stats.get("Full (λ₂=0.5)", {}).get("mean_pairwise_l2", full_enc.get("div_mean", "N/A"))
    no_div_div_l2 = div_stats.get("No diversity (λ₂=0)", {}).get("mean_pairwise_l2", no_div_enc.get("div_mean", "N/A"))

    full_k1 = full_data["random_attack"]["aggregated"]["1"]["fpr_0.001"]["tpr_mean"]
    full_k32 = full_data["random_attack"]["aggregated"]["32"]["fpr_0.001"]["tpr_mean"]
    no_div_k1 = no_div_data["random_attack"]["aggregated"]["1"]["fpr_0.001"]["tpr_mean"]
    no_div_k32 = no_div_data["random_attack"]["aggregated"]["32"]["fpr_0.001"]["tpr_mean"]

    full_drop = full_k1 - full_k32
    no_div_drop = no_div_k1 - no_div_k32

    print("\n" + "=" * 100)
    print("ABLATION SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Variant':<25} | {'KL Div':>8} | {'ASR@10':>8} | {'Mean L2':>8} | {'TPR@1e-3 K=1':>14} | {'TPR@1e-3 K=32':>14} | {'TPR Drop':>10}"
    print(header)
    print("-" * len(header))

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"{'Full (λ₂=0.5)':<25} | {fmt(full_enc.get('kl', 'N/A')):>8} | {fmt(full_enc.get('asr', 'N/A')):>8} | {fmt(full_div_l2):>8} | {full_k1:>14.4f} | {full_k32:>14.4f} | {full_drop*100:>9.1f}pp")
    print(f"{'No diversity (λ₂=0)':<25} | {fmt(no_div_enc.get('kl', 'N/A')):>8} | {fmt(no_div_enc.get('asr', 'N/A')):>8} | {fmt(no_div_div_l2):>8} | {no_div_k1:>14.4f} | {no_div_k32:>14.4f} | {no_div_drop*100:>9.1f}pp")
    print("=" * 100)

    summary = {
        "full_diversity": {
            "kl": full_enc.get("kl"),
            "asr": full_enc.get("asr"),
            "mean_pairwise_l2": full_div_l2,
            "tpr_at_1e3_k1": full_k1,
            "tpr_at_1e3_k32": full_k32,
            "tpr_drop_k1_to_k32": full_drop,
        },
        "no_diversity": {
            "kl": no_div_enc.get("kl"),
            "asr": no_div_enc.get("asr"),
            "mean_pairwise_l2": no_div_div_l2,
            "tpr_at_1e3_k1": no_div_k1,
            "tpr_at_1e3_k32": no_div_k32,
            "tpr_drop_k1_to_k32": no_div_drop,
        },
    }

    summary_path = results_dir / "ablation_diversity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
