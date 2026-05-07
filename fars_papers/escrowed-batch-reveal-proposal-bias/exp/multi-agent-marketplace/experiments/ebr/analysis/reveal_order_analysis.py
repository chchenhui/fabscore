"""Reveal-position bias analysis for EBR (Condition C).

Analyzes whether EBR eliminates ordering bias or merely shifts it from
arrival-order to list-position (primacy) bias. Computes reveal-position
selection rates, Wilson CIs, one-sided binomial test, and generates
comparison visualizations against HardGate arrival-order bias.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT.parents[0] / "EXPERIMENT_RESULTS"


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - half_width), min(1, center + half_width))


def load_reveal_data(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if r["paid_reveal_position"] and r["paid_reveal_position"] != "None":
                r["paid_reveal_position"] = int(r["paid_reveal_position"])
                r["paid_arrival_rank"] = int(r["paid_arrival_rank"])
                rows.append(r)
    return rows


def compute_selection_rates(rows: list[dict]) -> dict:
    n = len(rows)
    counts = {1: 0, 2: 0, 3: 0}
    for r in rows:
        pos = r["paid_reveal_position"]
        if pos in counts:
            counts[pos] += 1

    rates = {}
    for pos in [1, 2, 3]:
        x = counts[pos]
        rate = x / n if n > 0 else 0.0
        ci = wilson_ci(x, n)
        rates[pos] = {
            "count": x,
            "n": n,
            "rate": round(rate, 4),
            "ci_95_lower": round(ci[0], 4),
            "ci_95_upper": round(ci[1], 4),
        }

    binom_result = stats.binomtest(counts[1], n, p=1/3, alternative="greater")
    p_value = binom_result.pvalue

    chi2_observed = [counts[1], counts[2], counts[3]]
    chi2_expected = [n / 3.0] * 3
    chi2_stat, chi2_p = stats.chisquare(chi2_observed, chi2_expected)

    return {
        "n_valid": n,
        "reveal_position_rates": {str(k): v for k, v in rates.items()},
        "binomial_test_pos1": {
            "H0": "p(reveal_pos=1) = 1/3",
            "H1": "p(reveal_pos=1) > 1/3",
            "x": counts[1],
            "n": n,
            "observed_rate": round(counts[1] / n, 4),
            "p_value": round(float(p_value), 8),
            "significant_at_005": bool(p_value < 0.05),
        },
        "chi_squared_uniformity": {
            "statistic": round(float(chi2_stat), 3),
            "p_value": round(float(chi2_p), 6),
            "observed": chi2_observed,
            "expected": [round(e, 1) for e in chi2_expected],
        },
    }


def interpret_results(analysis: dict) -> dict:
    pos1_rate = analysis["reveal_position_rates"]["1"]["rate"]
    p_value = analysis["binomial_test_pos1"]["p_value"]
    significant = analysis["binomial_test_pos1"]["significant_at_005"]

    if pos1_rate <= 0.40:
        category = "no_bias"
        interpretation = (
            f"1st-reveal-position rate ({pos1_rate:.1%}) is <= 0.40 and "
            f"{'not significantly above 1/3' if not significant else 'marginally significant'}. "
            "EBR successfully eliminates ordering bias; the LLM treats all proposals "
            "equivalently when presented simultaneously."
        )
        recommendation = "EBR is effective as-is. No further debiasing needed."
    elif pos1_rate >= 0.50:
        category = "bias_shifted"
        interpretation = (
            f"1st-reveal-position rate ({pos1_rate:.1%}) is >= 0.50 and "
            f"{'significantly' if significant else 'not significantly'} above 1/3 "
            f"(p = {p_value:.6f}). "
            "Bias has shifted from arrival-order to list-position. "
            "This triggers the 'Pivot' branch: sequential visibility was causal for "
            "arrival-order bias, but generic primacy persists in batched presentation."
        )
        recommendation = (
            "Recommend adding a forced comparison step or rotating presentation order "
            "across multiple LLM calls to mitigate list-position primacy."
        )
    else:
        category = "mixed_signal"
        interpretation = (
            f"1st-reveal-position rate ({pos1_rate:.1%}) is between 0.40-0.50. "
            "Partial debiasing but residual list-position effect. Mixed signal."
        )
        recommendation = (
            "Consider rotating presentation order or forced pairwise comparison."
        )

    return {
        "category": category,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def plot_reveal_order_bias(analysis: dict, output_path: str):
    positions = [1, 2, 3]
    rates_data = analysis["reveal_position_rates"]
    rates = [rates_data[str(p)]["rate"] for p in positions]
    ci_lowers = [rates_data[str(p)]["ci_95_lower"] for p in positions]
    ci_uppers = [rates_data[str(p)]["ci_95_upper"] for p in positions]
    errors_lower = [r - cl for r, cl in zip(rates, ci_lowers)]
    errors_upper = [cu - r for r, cu in zip(rates, ci_uppers)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        positions, rates,
        yerr=[errors_lower, errors_upper],
        capsize=5, color=["#e74c3c", "#3498db", "#2ecc71"],
        edgecolor="black", linewidth=0.8, alpha=0.85, width=0.6,
    )
    ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1.5, label="Uniform (1/3)")
    ax.set_xlabel("Reveal Position (in shuffled batch)", fontsize=12)
    ax.set_ylabel("Selection Rate", fontsize=12)
    ax.set_title("EBR Reveal-Position Selection Rates\n(Condition C: Batched Reveal)", fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels(["1st", "2nd", "3rd"], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)

    n = analysis["n_valid"]
    for i, (pos, rate) in enumerate(zip(positions, rates)):
        count = rates_data[str(pos)]["count"]
        ax.text(pos, rate + errors_upper[i] + 0.03, f"{count}/{n}\n({rate:.1%})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    p_val = analysis["binomial_test_pos1"]["p_value"]
    ax.text(0.98, 0.98,
            f"Binomial test (pos1 > 1/3):\np = {p_val:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison(analysis: dict, hardgate_data: dict, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    hg_hist = hardgate_data["aggregate"]["rank_histogram"]
    hg_n = sum(hg_hist.values())
    hg_positions = [1, 2, 3]
    hg_rates = [hg_hist.get(str(p), hg_hist.get(p, 0)) / hg_n for p in hg_positions]
    hg_counts = [hg_hist.get(str(p), hg_hist.get(p, 0)) for p in hg_positions]
    hg_cis = [wilson_ci(c, hg_n) for c in hg_counts]
    hg_errors_lower = [r - ci[0] for r, ci in zip(hg_rates, hg_cis)]
    hg_errors_upper = [ci[1] - r for r, ci in zip(hg_rates, hg_cis)]

    ax1.bar(
        hg_positions, hg_rates,
        yerr=[hg_errors_lower, hg_errors_upper],
        capsize=5, color=["#e74c3c", "#3498db", "#2ecc71"],
        edgecolor="black", linewidth=0.8, alpha=0.85, width=0.6,
    )
    ax1.axhline(y=1/3, color="gray", linestyle="--", linewidth=1.5, label="Uniform (1/3)")
    ax1.set_xlabel("Arrival Rank (sequential order)", fontsize=11)
    ax1.set_ylabel("Selection Rate", fontsize=11)
    ax1.set_title("Condition B: HardGate\n(Sequential Reveal)", fontsize=12)
    ax1.set_xticks(hg_positions)
    ax1.set_xticklabels(["1st", "2nd", "3rd"], fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=9)
    for i, (pos, rate) in enumerate(zip(hg_positions, hg_rates)):
        ax1.text(pos, rate + hg_errors_upper[i] + 0.03, f"{hg_counts[i]}/{hg_n}\n({rate:.1%})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ebr_rates_data = analysis["reveal_position_rates"]
    ebr_positions = [1, 2, 3]
    ebr_rates = [ebr_rates_data[str(p)]["rate"] for p in ebr_positions]
    ebr_ci_lowers = [ebr_rates_data[str(p)]["ci_95_lower"] for p in ebr_positions]
    ebr_ci_uppers = [ebr_rates_data[str(p)]["ci_95_upper"] for p in ebr_positions]
    ebr_errors_lower = [r - cl for r, cl in zip(ebr_rates, ebr_ci_lowers)]
    ebr_errors_upper = [cu - r for r, cu in zip(ebr_rates, ebr_ci_uppers)]
    ebr_n = analysis["n_valid"]

    ax2.bar(
        ebr_positions, ebr_rates,
        yerr=[ebr_errors_lower, ebr_errors_upper],
        capsize=5, color=["#e74c3c", "#3498db", "#2ecc71"],
        edgecolor="black", linewidth=0.8, alpha=0.85, width=0.6,
    )
    ax2.axhline(y=1/3, color="gray", linestyle="--", linewidth=1.5, label="Uniform (1/3)")
    ax2.set_xlabel("Reveal Position (shuffled batch order)", fontsize=11)
    ax2.set_title("Condition C: EBR\n(Batched Reveal, Shuffled)", fontsize=12)
    ax2.set_xticks(ebr_positions)
    ax2.set_xticklabels(["1st", "2nd", "3rd"], fontsize=10)
    ax2.legend(fontsize=9)
    for i, (pos, rate) in enumerate(zip(ebr_positions, ebr_rates)):
        count = ebr_rates_data[str(pos)]["count"]
        ax2.text(pos, rate + ebr_errors_upper[i] + 0.03, f"{count}/{ebr_n}\n({rate:.1%})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Arrival-Order Bias (HardGate) vs. Reveal-Position Bias (EBR)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    csv_path = REPO_ROOT / "experiments" / "ebr" / "results" / "ebr_gemini" / "reveal_order_data.csv"
    rows = load_reveal_data(str(csv_path))
    print(f"Loaded {len(rows)} valid runs from {csv_path}")

    analysis = compute_selection_rates(rows)
    interpretation = interpret_results(analysis)

    print("\n" + "=" * 70)
    print("REVEAL-POSITION BIAS ANALYSIS")
    print("=" * 70)
    print(f"\nValid runs: {analysis['n_valid']}")
    print("\nReveal-position selection rates:")
    for pos in ["1", "2", "3"]:
        d = analysis["reveal_position_rates"][pos]
        print(f"  Position {pos}: {d['count']}/{d['n']} = {d['rate']:.3f} "
              f"[95% CI: ({d['ci_95_lower']:.3f}, {d['ci_95_upper']:.3f})]")

    btest = analysis["binomial_test_pos1"]
    print(f"\nBinomial test (pos1 > 1/3): p = {btest['p_value']:.8f} "
          f"{'***' if btest['p_value'] < 0.001 else '**' if btest['p_value'] < 0.01 else '*' if btest['p_value'] < 0.05 else 'n.s.'}")

    chi2 = analysis["chi_squared_uniformity"]
    print(f"Chi-squared vs uniform: stat = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}")

    print(f"\nInterpretation: {interpretation['category']}")
    print(f"  {interpretation['interpretation']}")
    print(f"  Recommendation: {interpretation['recommendation']}")

    figures_dir = REPO_ROOT / "experiments" / "ebr" / "figures"
    plot_reveal_order_bias(analysis, str(figures_dir / "reveal_order_bias.png"))

    hardgate_path = RESULTS_DIR / "hardgate_gemini_flash" / "RESULTS.json"
    with open(hardgate_path) as f:
        hardgate_data = json.load(f)
    plot_comparison(analysis, hardgate_data, str(figures_dir / "reveal_order_comparison.png"))

    full_results = {
        **analysis,
        "interpretation": interpretation,
    }

    out_json = REPO_ROOT / "experiments" / "ebr" / "analysis" / "reveal_order_results.json"
    with open(out_json, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    return full_results


if __name__ == "__main__":
    main()
