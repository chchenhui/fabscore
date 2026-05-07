"""Best-of-j accuracy scaling curves for Qwen2.5-7B on Countdown and Mini Sudoku.
Reuses existing k-sample JSONL outputs to compute best-of-j for j = 1,2,4,...,k_max.
Plots scaling curves with Dream and greedy reference lines.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

SEEDS = [42, 123, 456]

TASKS = {
    "countdown": {
        "file_prefix": "qwen_bok_opt_countdown",
        "k_max": 35,
        "k_median": 35,
        "dream_acc": 0.066,
        "greedy_acc": 0.060,
        "title": "Countdown",
    },
    "sudoku": {
        "file_prefix": "qwen_bok_opt_sudoku",
        "k_max": 39,
        "k_median": 39,
        "dream_acc": 0.776,
        "greedy_acc": 0.168,
        "title": "Mini Sudoku",
    },
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_best_of_j(records, j_values):
    accuracies = {}
    for j in j_values:
        solved = 0
        for rec in records:
            scores = rec["all_scores"][:j]
            if max(scores) >= 1.0:
                solved += 1
        accuracies[j] = solved / len(records)
    return accuracies


def get_j_values(k_max):
    j_vals = []
    j = 1
    while j <= k_max:
        j_vals.append(j)
        j *= 2
    if k_max not in j_vals:
        j_vals.append(k_max)
    return sorted(j_vals)


def plot_task(task_key, cfg, all_seed_accs, output_path):
    j_values = get_j_values(cfg["k_max"])
    means = np.array([np.mean([all_seed_accs[s][j] for s in SEEDS]) for j in j_values])
    stds = np.array([np.std([all_seed_accs[s][j] for s in SEEDS]) for j in j_values])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(j_values, means * 100, "o-", color="#4C72B0", linewidth=2, markersize=6,
            label="Qwen2.5-7B best-of-j", zorder=5)
    ax.fill_between(j_values, (means - stds) * 100, (means + stds) * 100,
                    color="#4C72B0", alpha=0.2)

    ax.axhline(y=cfg["dream_acc"] * 100, color="#55A868", linestyle="--", linewidth=1.5,
               label=f"Dream diffusion ({cfg['dream_acc']*100:.1f}%)")
    ax.axhline(y=cfg["greedy_acc"] * 100, color="#C44E52", linestyle=":", linewidth=1.5,
               label=f"Qwen greedy ({cfg['greedy_acc']*100:.1f}%)")
    ax.axvline(x=cfg["k_median"], color="#8172B2", linestyle="--", linewidth=1.5,
               label=f"Compute-matched k={cfg['k_median']}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(j_values)
    ax.set_xticklabels([str(j) for j in j_values])
    ax.set_xlabel("Number of samples (j)", fontsize=12)
    ax.set_ylabel("Best-of-j accuracy (%)", fontsize=12)
    ax.set_title(f"Qwen2.5-7B Scaling Curve: {cfg['title']}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min = max(0, min(means.min() * 100 - 5, cfg["greedy_acc"] * 100 - 3))
    y_max = min(100, max(means.max() * 100 + 5, cfg["dream_acc"] * 100 + 5))
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    results_summary = {}

    for task_key, cfg in TASKS.items():
        j_values = get_j_values(cfg["k_max"])
        all_seed_accs = {}

        for seed in SEEDS:
            fname = f"{cfg['file_prefix']}_seed{seed}.jsonl"
            path = os.path.join(RAW_DIR, fname)
            records = load_jsonl(path)
            accs = compute_best_of_j(records, j_values)
            all_seed_accs[seed] = accs

        print(f"\n=== {cfg['title']} ===")
        print(f"{'j':>5s}  {'mean':>8s}  {'std':>8s}  " + "  ".join(f"seed{s}" for s in SEEDS))
        print("-" * 60)

        task_data = []
        for j in j_values:
            per_seed = [all_seed_accs[s][j] for s in SEEDS]
            m = np.mean(per_seed)
            sd = np.std(per_seed)
            vals_str = "  ".join(f"{v:.4f}" for v in per_seed)
            print(f"{j:>5d}  {m:>8.4f}  {sd:>8.4f}  {vals_str}")
            task_data.append({
                "j": j,
                "mean": round(float(m), 4),
                "std": round(float(sd), 4),
                "per_seed": {str(s): round(v, 4) for s, v in zip(SEEDS, per_seed)},
            })

        results_summary[task_key] = {
            "j_values": j_values,
            "data": task_data,
            "dream_acc": cfg["dream_acc"],
            "greedy_acc": cfg["greedy_acc"],
            "k_median": cfg["k_median"],
        }

        fig_path = os.path.join(FIGURES_DIR, f"scaling_{task_key}.png")
        plot_task(task_key, cfg, all_seed_accs, fig_path)

    summary_path = os.path.join(TABLES_DIR, "scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
