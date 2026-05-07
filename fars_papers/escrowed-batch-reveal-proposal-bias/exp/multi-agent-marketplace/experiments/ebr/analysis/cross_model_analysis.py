"""Cross-model robustness analysis: HardGate vs EBR on gemini-2.5-flash and claude-sonnet-4-5.

Computes statistical tests for both models and generates a grouped bar chart
comparing earliest-arrival rates across models and conditions.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT.parent / "EXPERIMENT_RESULTS"
EBR_RESULTS = REPO_ROOT / "experiments" / "ebr" / "results"
FIGURES_DIR = REPO_ROOT / "experiments" / "ebr" / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - half_width), min(1, center + half_width))


def two_proportion_ztest(x1, n1, x2, n2, alternative="smaller"):
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se_pool == 0:
        return 0.0, 1.0, (0.0, 0.0)
    z = (p1 - p2) / se_pool
    if alternative == "smaller":
        p_value = stats.norm.cdf(z)
    else:
        p_value = 2 * stats.norm.cdf(-abs(z))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = (p1 - p2) - 1.96 * se_diff
    ci_high = (p1 - p2) + 1.96 * se_diff
    return z, p_value, (ci_low, ci_high)


def fisher_exact_2x2(x1, n1, x2, n2, alternative="less"):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
    return odds_ratio, p_value


def load_gemini_data():
    hg = json.load(open(RESULTS_DIR / "hardgate_gemini_flash" / "RESULTS.json"))
    ebr = json.load(open(RESULTS_DIR / "ebr_gemini_flash" / "RESULTS.json"))

    hg_hist = hg["aggregate"]["rank_histogram"] if "aggregate" in hg else hg.get("rank_histogram", {})
    hg_n = hg["total_runs"]
    hg_x = hg_hist.get("1", hg_hist.get(1, 0))

    ebr_hist = ebr.get("rank_histogram", {})
    ebr_n = ebr["total_runs"]
    ebr_x = ebr_hist.get("1", ebr_hist.get(1, 0))

    hg_comp = hg["aggregate"]["completion_rate_mean"] if "aggregate" in hg else hg.get("completion_rate_mean", 0)
    ebr_comp = ebr.get("completion_rate_mean", 0)

    return {
        "hardgate": {"x": int(hg_x), "n": int(hg_n), "completion": hg_comp},
        "ebr": {"x": int(ebr_x), "n": int(ebr_n), "completion": ebr_comp},
    }


def load_claude_data():
    hg = json.load(open(EBR_RESULTS / "hardgate_claude" / "summary.json"))
    ebr = json.load(open(EBR_RESULTS / "ebr_claude" / "summary.json"))

    hg_x = hg["rank_histogram"].get("1", 0)
    hg_n = hg["total_runs"]
    ebr_x = ebr["rank_histogram"].get("1", 0)
    ebr_n = ebr["total_runs"]

    return {
        "hardgate": {"x": int(hg_x), "n": int(hg_n), "completion": hg["completion_rate_mean"]},
        "ebr": {"x": int(ebr_x), "n": int(ebr_n), "completion": ebr["completion_rate_mean"]},
    }


def analyze_model(name, data):
    hg = data["hardgate"]
    ebr = data["ebr"]

    hg_rate = hg["x"] / hg["n"]
    ebr_rate = ebr["x"] / ebr["n"]
    delta = hg_rate - ebr_rate

    z, p, ci = two_proportion_ztest(ebr["x"], ebr["n"], hg["x"], hg["n"], alternative="smaller")
    or_val, p_fisher = fisher_exact_2x2(ebr["x"], ebr["n"], hg["x"], hg["n"], alternative="less")

    hg_ci = wilson_ci(hg["x"], hg["n"])
    ebr_ci = wilson_ci(ebr["x"], ebr["n"])

    comp_drop = hg["completion"] - ebr["completion"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  HardGate: {hg['x']}/{hg['n']} = {hg_rate:.3f}  [95% CI: ({hg_ci[0]:.3f}, {hg_ci[1]:.3f})]")
    print(f"  EBR:      {ebr['x']}/{ebr['n']} = {ebr_rate:.3f}  [95% CI: ({ebr_ci[0]:.3f}, {ebr_ci[1]:.3f})]")
    print(f"  Delta (B-C): {delta:.3f}")
    print(f"  Z-test: z={z:.3f}, p={p:.6f} (one-sided, H1: C < B)")
    print(f"  Fisher: OR={or_val:.3f}, p={p_fisher:.6f}")
    print(f"  95% CI for (C-B): ({ci[0]:.3f}, {ci[1]:.3f})")
    print(f"  Completion drop: {comp_drop:.3f}")

    return {
        "model": name,
        "hardgate_x": hg["x"], "hardgate_n": hg["n"],
        "hardgate_rate": round(hg_rate, 3),
        "hardgate_ci": [round(c, 3) for c in hg_ci],
        "hardgate_completion": hg["completion"],
        "ebr_x": ebr["x"], "ebr_n": ebr["n"],
        "ebr_rate": round(ebr_rate, 3),
        "ebr_ci": [round(c, 3) for c in ebr_ci],
        "ebr_completion": ebr["completion"],
        "delta_B_minus_C": round(delta, 3),
        "ci_95_C_minus_B": [round(ci[0], 3), round(ci[1], 3)],
        "z_statistic": round(z, 3),
        "p_value_one_sided": round(p, 8),
        "fisher_odds_ratio": round(or_val, 3),
        "fisher_p_value": round(p_fisher, 8),
        "completion_drop": round(comp_drop, 3),
        "significant": bool(p < 0.05),
        "large_effect": bool(delta >= 0.15),
    }


def make_grouped_bar_chart(gemini_result, claude_result, output_path):
    models = ["gemini-2.5-flash", "claude-sonnet-4-5"]
    hg_rates = [gemini_result["hardgate_rate"], claude_result["hardgate_rate"]]
    ebr_rates = [gemini_result["ebr_rate"], claude_result["ebr_rate"]]

    hg_ci_low = [r - wilson_ci(d["hardgate_x"], d["hardgate_n"])[0]
                 for r, d in zip(hg_rates, [gemini_result, claude_result])]
    hg_ci_high = [wilson_ci(d["hardgate_x"], d["hardgate_n"])[1] - r
                  for r, d in zip(hg_rates, [gemini_result, claude_result])]
    ebr_ci_low = [r - wilson_ci(d["ebr_x"], d["ebr_n"])[0]
                  for r, d in zip(ebr_rates, [gemini_result, claude_result])]
    ebr_ci_high = [wilson_ci(d["ebr_x"], d["ebr_n"])[1] - r
                   for r, d in zip(ebr_rates, [gemini_result, claude_result])]

    x = np.arange(len(models))
    width = 0.30

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_hg = ax.bar(x - width/2, hg_rates, width, label="HardGate (B)",
                     color="#e74c3c", edgecolor="black", linewidth=0.5,
                     yerr=[hg_ci_low, hg_ci_high], capsize=5, error_kw={"linewidth": 1.5})
    bars_ebr = ax.bar(x + width/2, ebr_rates, width, label="EBR (C)",
                      color="#3498db", edgecolor="black", linewidth=0.5,
                      yerr=[ebr_ci_low, ebr_ci_high], capsize=5, error_kw={"linewidth": 1.5})

    ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="Random chance (1/3)")

    for i, (hg_r, ebr_r) in enumerate(zip(hg_rates, ebr_rates)):
        ax.text(i - width/2, hg_r + hg_ci_high[i] + 0.02, f"{hg_r:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + width/2, ebr_r + ebr_ci_high[i] + 0.02, f"{ebr_r:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, res in enumerate([gemini_result, claude_result]):
        star = "***" if res["p_value_one_sided"] < 0.001 else "**" if res["p_value_one_sided"] < 0.01 else "*" if res["p_value_one_sided"] < 0.05 else "n.s."
        y_top = max(hg_rates[i] + hg_ci_high[i], ebr_rates[i] + ebr_ci_high[i]) + 0.07
        ax.annotate("", xy=(i - width/2, y_top), xytext=(i + width/2, y_top),
                     arrowprops=dict(arrowstyle="-", color="black", lw=1.2))
        ax.text(i, y_top + 0.01, f"{star}\np={res['p_value_one_sided']:.4f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Earliest-Arrival Chosen Rate", fontsize=12)
    ax.set_title("Cross-Model Comparison: HardGate vs EBR\nEarliest-Arrival Bias by Model", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to: {output_path}")


def main():
    gemini_data = load_gemini_data()
    claude_data = load_claude_data()

    gemini_result = analyze_model("gemini-2.5-flash", gemini_data)
    claude_result = analyze_model("claude-sonnet-4-5", claude_data)

    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'HardGate(B)':<15} {'EBR(C)':<15} {'Delta(B-C)':<12} {'p-value':<12} {'Sig?'}")
    print("-" * 80)
    for r in [gemini_result, claude_result]:
        sig = "Yes***" if r["p_value_one_sided"] < 0.001 else "Yes**" if r["p_value_one_sided"] < 0.01 else "Yes*" if r["p_value_one_sided"] < 0.05 else "No"
        print(f"{r['model']:<22} {r['hardgate_rate']:<15.3f} {r['ebr_rate']:<15.3f} {r['delta_B_minus_C']:<12.3f} {r['p_value_one_sided']:<12.6f} {sig}")

    print(f"\n{'='*70}")
    print("GENERALIZATION ASSESSMENT")
    print(f"{'='*70}")

    gemini_sig = gemini_result["significant"] and gemini_result["large_effect"]
    claude_sig = claude_result["significant"] and claude_result["large_effect"]

    if gemini_sig and claude_sig:
        verdict = "STRONG_GENERALIZATION"
        detail = ("Both models show statistically significant reduction with large effect size "
                  f"(delta >= 0.15, p < 0.05). EBR generalizes well as a model-agnostic protocol intervention.")
    elif gemini_sig and not claude_sig:
        if claude_result["significant"]:
            verdict = "PARTIAL_GENERALIZATION"
            detail = (f"Gemini shows large significant effect (delta={gemini_result['delta_B_minus_C']:.3f}). "
                      f"Claude shows significant but small effect (delta={claude_result['delta_B_minus_C']:.3f}). "
                      "EBR may depend on model-specific satisficing tendencies.")
        else:
            claude_hg_near_random = abs(claude_result["hardgate_rate"] - 1/3) < 0.10
            if claude_hg_near_random:
                verdict = "PARTIAL_GENERALIZATION_FLOOR_EFFECT"
                detail = (f"Gemini shows large significant effect (delta={gemini_result['delta_B_minus_C']:.3f}). "
                          f"Claude HardGate already near random chance ({claude_result['hardgate_rate']:.3f}), "
                          "so there is insufficient baseline bias for EBR to reduce. "
                          "This is a floor effect, not evidence against EBR's mechanism.")
            else:
                verdict = "PARTIAL_GENERALIZATION"
                detail = (f"Only gemini shows the effect. Claude does not show significant reduction. "
                          "The intervention may depend on model-specific satisficing tendencies.")
    elif not gemini_sig and not claude_sig:
        verdict = "NO_EFFECT"
        detail = "Neither model shows a significant effect. The negative result is robust across models."
    else:
        verdict = "CLAUDE_ONLY"
        detail = "Only claude shows the effect; unexpected pattern."

    print(f"  Verdict: {verdict}")
    print(f"  {detail}")

    output = {
        "gemini_2_5_flash": gemini_result,
        "claude_sonnet_4_5": claude_result,
        "generalization_verdict": verdict,
        "generalization_detail": detail,
        "cross_model_table": {
            "headers": ["Model", "HardGate (B) Earliest-Arrival", "EBR (C) Earliest-Arrival", "Delta (B-C)", "p-value"],
            "rows": [
                [r["model"], r["hardgate_rate"], r["ebr_rate"], r["delta_B_minus_C"], r["p_value_one_sided"]]
                for r in [gemini_result, claude_result]
            ]
        }
    }

    chart_path = FIGURES_DIR / "cross_model_comparison.png"
    make_grouped_bar_chart(gemini_result, claude_result, chart_path)

    output_path = REPO_ROOT.parent / "EXPERIMENT_RESULTS" / "cross_model_robustness"
    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "RESULTS.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path / 'RESULTS.json'}")

    return output


if __name__ == "__main__":
    results = main()
