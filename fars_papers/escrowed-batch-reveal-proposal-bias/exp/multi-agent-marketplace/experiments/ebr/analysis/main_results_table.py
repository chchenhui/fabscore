"""Generate main results table and grouped bar chart for EBR evaluation.

Loads results from all experimental conditions, prints a formatted comparison
table, and produces a bar chart with Wilson CIs and the random-chance baseline.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


RESULTS_DIR = Path(__file__).resolve().parents[4] / "EXPERIMENT_RESULTS"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    hw = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - hw), min(1, center + hw))


def load_condition(subdir: str) -> dict:
    with open(RESULTS_DIR / subdir / "RESULTS.json") as f:
        return json.load(f)


def get_counts(data: dict) -> tuple:
    if "aggregate" in data:
        n = data["total_runs"]
        hist = data["aggregate"]["rank_histogram"]
        comp = data["aggregate"]["completion_rate_mean"]
    else:
        metrics = data.get("metrics", data)
        n = data.get("total_runs", metrics.get("total_runs", 0))
        hist = metrics.get("rank_histogram", {})
        comp = metrics.get("completion_rate_mean", data.get("completion_rate_mean", 0))
    x = hist.get("1", hist.get(1, 0))
    return int(x), int(n), float(comp)


def load_stat_results() -> dict:
    p = Path(__file__).parent / "statistical_test_results.json"
    with open(p) as f:
        return json.load(f)


def print_table(rows):
    header = ["Method", "Condition", "Earliest-Arrival Rate", "Completion Rate", "p-value vs B"]
    widths = [max(len(header[i]), max(len(str(r[i])) for r in rows)) for i in range(len(header))]

    def fmt_row(r):
        return "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(fmt_row(header))
    print(sep)
    for r in rows:
        print(fmt_row(r))


def main():
    softwait = load_condition("softwait_gemini_flash")
    its = load_condition("its_gemini_flash")
    hardgate = load_condition("hardgate_gemini_flash")
    ebr = load_condition("ebr_gemini_flash")
    stat = load_stat_results()

    x_a, n_a, c_a = get_counts(softwait)
    x_its, n_its, c_its = get_counts(its)
    x_b, n_b, c_b = get_counts(hardgate)
    x_c, n_c, c_c = get_counts(ebr)

    p_ac = stat["secondary_comparisons"]["SoftWait (A)"]["p_value_one_sided"]
    p_its = stat["secondary_comparisons"]["ITS (A')"]["p_value_one_sided"]
    p_bc = stat["primary_comparison_B_vs_C"]["p_value_one_sided"]

    rows = [
        ("Random baseline", "--", "0.333 (theoretical)", "--", "--"),
        ("Magentic reported", "--", "0.60-1.00 (literature)", "--", "--"),
        (
            "SoftWait (prompt-only)",
            "A",
            f"{x_a/n_a:.3f} +/- {np.std([1.0]*x_a + [0.0]*(n_a-x_a)):.3f} (n={n_a})",
            f"{c_a:.3f}",
            f"{p_ac:.4f}",
        ),
        (
            "Inference-time scaling",
            "A+ITS",
            f"{x_its/n_its:.3f} +/- {np.std([1.0]*x_its + [0.0]*(n_its-x_its)):.3f} (n={n_its})",
            f"{c_its:.3f}",
            f"{p_its:.4f}",
        ),
        (
            "QuoteBatch / HardGate",
            "B",
            f"{x_b/n_b:.3f} +/- {np.std([1.0]*x_b + [0.0]*(n_b-x_b)):.3f} (n={n_b})",
            f"{c_b:.3f}",
            "--",
        ),
        (
            "QuoteBatch + EBR (ours)",
            "C",
            f"{x_c/n_c:.3f} +/- {np.std([1.0]*x_c + [0.0]*(n_c-x_c)):.3f} (n={n_c})",
            f"{c_c:.3f}",
            f"{p_bc:.8f}",
        ),
    ]

    print("=" * 80)
    print("MAIN RESULTS TABLE: EBR Effectiveness Evaluation")
    print("=" * 80)
    print()
    print_table(rows)

    names = ["Random\n(theory)", "SoftWait\n(A)", "ITS\n(A')", "HardGate\n(B)", "EBR\n(C)"]
    rates = [1/3, x_a / n_a, x_its / n_its, x_b / n_b, x_c / n_c]
    counts = [(0, 1), (x_a, n_a), (x_its, n_its), (x_b, n_b), (x_c, n_c)]
    colors = ["#999999", "#5DADE2", "#48C9B0", "#E74C3C", "#2ECC71"]

    cis = []
    for x, n in counts:
        if n <= 1:
            cis.append((1/3, 1/3))
        else:
            cis.append(wilson_ci(x, n))

    yerr_low = [r - ci[0] for r, ci in zip(rates, cis)]
    yerr_high = [ci[1] - r for r, ci in zip(rates, cis)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(names))
    bars = ax.bar(
        x_pos, rates, color=colors, edgecolor="black", linewidth=0.8,
        yerr=[yerr_low, yerr_high], capsize=5, error_kw={"linewidth": 1.2},
    )

    ax.axhline(y=1/3, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Random chance (1/3)")
    ax.axhspan(0.60, 1.00, alpha=0.07, color="red", label="Magentic reported range")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Earliest-Arrival Chosen Rate", fontsize=12)
    ax.set_title("EBR Reduces First-Proposal Bias Below Random Chance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)

    for i, (r, name) in enumerate(zip(rates, names)):
        label = f"{r:.3f}" if i > 0 else f"{r:.3f}*"
        ax.text(i, r + yerr_high[i] + 0.025, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    sig_y = 0.85
    ax.annotate("", xy=(3, sig_y), xytext=(4, sig_y),
                arrowprops=dict(arrowstyle="-", lw=1.2, color="black"))
    ax.text(3.5, sig_y + 0.02, "p < 0.001 ***", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "main_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
