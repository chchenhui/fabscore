"""Statistical tests for EBR effectiveness evaluation.

Compares earliest-arrival chosen rates across experimental conditions using
two-proportion z-tests and Fisher's exact tests. Primary comparison: HardGate (B)
vs EBR (C). Secondary: all conditions vs B.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_DIR = Path(__file__).resolve().parents[4] / "EXPERIMENT_RESULTS"


def load_condition(subdir: str) -> dict:
    p = RESULTS_DIR / subdir / "RESULTS.json"
    with open(p) as f:
        return json.load(f)


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alternative="smaller"):
    """Two-proportion z-test.
    
    H0: p1 = p2
    alternative='smaller': H1: p1 < p2  (i.e., condition 1 has lower rate)
    Returns z-statistic, one-sided p-value, and 95% CI for (p1 - p2).
    """
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


def fisher_exact_2x2(x1: int, n1: int, x2: int, n2: int, alternative="less"):
    """Fisher's exact test on 2x2 contingency table."""
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
    return odds_ratio, p_value


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - half_width), min(1, center + half_width))


def run_all_tests():
    softwait = load_condition("softwait_gemini_flash")
    its = load_condition("its_gemini_flash")
    hardgate = load_condition("hardgate_gemini_flash")
    ebr = load_condition("ebr_gemini_flash")

    def get_counts(data: dict) -> tuple:
        if "aggregate" in data:
            n = data["total_runs"]
            hist = data["aggregate"]["rank_histogram"]
        else:
            n = data.get("total_runs", data.get("metrics", {}).get("total_runs", 0))
            hist = data.get("rank_histogram", data.get("metrics", {}).get("rank_histogram", {}))
        x_earliest = hist.get("1", hist.get(1, 0))
        return int(x_earliest), int(n)

    x_a, n_a = get_counts(softwait)
    x_its, n_its = get_counts(its)
    x_b, n_b = get_counts(hardgate)
    x_c, n_c = get_counts(ebr)

    print("=" * 70)
    print("STATISTICAL TESTS: EBR Effectiveness Evaluation")
    print("=" * 70)

    conditions = {
        "SoftWait (A)": (x_a, n_a),
        "ITS (A')": (x_its, n_its),
        "HardGate (B)": (x_b, n_b),
        "EBR (C)": (x_c, n_c),
    }
    print("\n--- Raw Counts ---")
    for name, (x, n) in conditions.items():
        ci = wilson_ci(x, n)
        print(f"  {name}: {x}/{n} = {x/n:.3f}  [95% Wilson CI: ({ci[0]:.3f}, {ci[1]:.3f})]")

    print("\n--- Primary Comparison: EBR (C) vs HardGate (B) ---")
    z_bc, p_bc, ci_bc = two_proportion_ztest(x_c, n_c, x_b, n_b, alternative="smaller")
    or_bc, pf_bc = fisher_exact_2x2(x_c, n_c, x_b, n_b, alternative="less")
    delta_bc = x_b / n_b - x_c / n_c

    print(f"  HardGate rate: {x_b/n_b:.3f}  |  EBR rate: {x_c/n_c:.3f}")
    print(f"  Delta (B - C): {delta_bc:.3f}")
    print(f"  95% CI for (C - B): ({ci_bc[0]:.3f}, {ci_bc[1]:.3f})")
    print(f"  Z-test: z = {z_bc:.3f}, p = {p_bc:.6f} (one-sided, H1: C < B)")
    print(f"  Fisher's exact: OR = {or_bc:.3f}, p = {pf_bc:.6f} (one-sided)")

    print("\n--- Secondary Comparisons vs HardGate (B) ---")
    secondary_results = {}
    for name, (x, n) in [("SoftWait (A)", (x_a, n_a)), ("ITS (A')", (x_its, n_its))]:
        z, p, ci = two_proportion_ztest(x, n, x_b, n_b, alternative="smaller")
        delta = x_b / n_b - x / n
        print(f"  {name} vs B: delta = {delta:.3f}, z = {z:.3f}, p = {p:.6f}")
        secondary_results[name] = {"z": z, "p": p, "delta": delta, "ci": ci}

    print("\n--- Completion Rate Comparison ---")
    comp_b = hardgate["aggregate"]["completion_rate_mean"] if "aggregate" in hardgate else hardgate.get("completion_rate_mean", hardgate.get("metrics", {}).get("completion_rate_mean"))
    comp_c = ebr.get("completion_rate_mean", ebr.get("metrics", {}).get("completion_rate_mean"))
    comp_drop = comp_b - comp_c
    print(f"  HardGate completion: {comp_b:.3f}")
    print(f"  EBR completion:      {comp_c:.3f}")
    print(f"  Drop:                {comp_drop:.3f} pp")
    print(f"  Threshold (<=10pp):  {'PASS' if comp_drop <= 0.10 else 'FAIL'}")

    print("\n--- Decision Rule Application ---")
    accept_threshold = x_b / n_b - 0.20
    refute_threshold = x_b / n_b - 0.10
    ebr_rate = x_c / n_c

    print(f"  EarliestArrival(B) = {x_b/n_b:.3f}")
    print(f"  EarliestArrival(C) = {ebr_rate:.3f}")
    print(f"  Accept threshold:  C <= {accept_threshold:.3f}")
    print(f"  Refute threshold:  C >= {refute_threshold:.3f}")

    if ebr_rate <= accept_threshold and comp_drop <= 0.10:
        decision = "ACCEPT/PROCEED"
        decision_detail = (
            f"EarliestArrival(C) = {ebr_rate:.3f} <= {accept_threshold:.3f} = EarliestArrival(B) - 0.20, "
            f"and completion rate drop = {comp_drop:.3f} <= 0.10. "
            "Sequential visibility is a causal driver of residual bias; EBR is an effective protocol intervention."
        )
    elif ebr_rate >= refute_threshold:
        decision = "REFUTE"
        decision_detail = (
            f"EarliestArrival(C) = {ebr_rate:.3f} >= {refute_threshold:.3f} = EarliestArrival(B) - 0.10. "
            "After payment gating, sequential visibility is not the dominant remaining driver."
        )
    else:
        decision = "PIVOT"
        decision_detail = (
            f"EarliestArrival(C) = {ebr_rate:.3f} is between refute and accept thresholds. "
            "Bias may shift from arrival-order to list-position effect."
        )

    print(f"\n  DECISION: {decision}")
    print(f"  {decision_detail}")

    results = {
        "primary_comparison_B_vs_C": {
            "hardgate_earliest_count": x_b,
            "hardgate_n": n_b,
            "hardgate_rate": round(x_b / n_b, 3),
            "ebr_earliest_count": x_c,
            "ebr_n": n_c,
            "ebr_rate": round(x_c / n_c, 3),
            "delta_B_minus_C": round(delta_bc, 3),
            "ci_95_C_minus_B": [round(ci_bc[0], 3), round(ci_bc[1], 3)],
            "z_statistic": round(z_bc, 3),
            "p_value_one_sided": round(p_bc, 8),
            "fisher_odds_ratio": round(or_bc, 3),
            "fisher_p_value": round(pf_bc, 8),
        },
        "completion_rate_comparison": {
            "hardgate_completion": comp_b,
            "ebr_completion": comp_c,
            "drop_pp": round(comp_drop, 3),
            "within_threshold": comp_drop <= 0.10,
        },
        "wilson_confidence_intervals": {
            name: {"rate": round(x / n, 3), "ci_95": [round(c, 3) for c in wilson_ci(x, n)]}
            for name, (x, n) in conditions.items()
        },
        "secondary_comparisons": {
            name: {
                "delta_B_minus_X": round(v["delta"], 3),
                "z_statistic": round(v["z"], 3),
                "p_value_one_sided": round(v["p"], 6),
            }
            for name, v in secondary_results.items()
        },
        "decision": decision,
        "decision_detail": decision_detail,
    }

    out_path = Path(__file__).parent / "statistical_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    run_all_tests()
