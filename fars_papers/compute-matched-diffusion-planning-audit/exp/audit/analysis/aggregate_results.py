"""Aggregate results from all three conditions (A: greedy, B: best-of-k, C: diffusion)
across both tasks. Produces main_results.csv and main_results.png grouped bar chart.
"""

import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
TIMING_FILE = os.path.join(BASE_DIR, "timing", "calibration_results.json")


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_timing():
    with open(TIMING_FILE) as f:
        return json.load(f)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    greedy = load_csv(os.path.join(TABLES_DIR, "qwen_greedy_results.csv"))
    dream = load_csv(os.path.join(TABLES_DIR, "dream_diffusion_results.csv"))
    bok_cd = load_csv(os.path.join(TABLES_DIR, "qwen_bok_countdown_results.csv"))
    bok_sdk = load_csv(os.path.join(TABLES_DIR, "qwen_bok_sudoku_results.csv"))
    timing = load_timing()

    greedy_cd = next(r for r in greedy if r["task"] == "countdown")
    greedy_sdk = next(r for r in greedy if r["task"] == "mini_sudoku")
    dream_cd = next(r for r in dream if r["task"] == "countdown")
    dream_sdk = next(r for r in dream if r["task"] == "mini_sudoku")

    bok_cd_mean = next(r for r in bok_cd if r["seed"] == "mean")
    bok_cd_std = next(r for r in bok_cd if r["seed"] == "std")
    bok_sdk_mean = next(r for r in bok_sdk if r["seed"] == "mean")
    bok_sdk_std = next(r for r in bok_sdk if r["seed"] == "std")

    rows = [
        {
            "method": "Greedy (Cond. A)",
            "base_model": "Qwen2.5-7B",
            "task": "Countdown",
            "accuracy": f"{float(greedy_cd['accuracy']):.4f}",
            "accuracy_std": "",
            "wallclock_per_instance_s": f"{timing['countdown']['qwen_median']:.2f}",
            "k": 1,
        },
        {
            "method": "Best-of-k (Cond. B)",
            "base_model": "Qwen2.5-7B",
            "task": "Countdown",
            "accuracy": f"{float(bok_cd_mean['accuracy']):.4f}",
            "accuracy_std": f"{float(bok_cd_std['accuracy']):.4f}",
            "wallclock_per_instance_s": f"~{timing['countdown']['dream_median']:.2f}",
            "k": timing["countdown"]["k_median"],
        },
        {
            "method": "Diffusion (Cond. C)",
            "base_model": "Dream-v0-Base-7B",
            "task": "Countdown",
            "accuracy": f"{float(dream_cd['accuracy']):.4f}",
            "accuracy_std": "",
            "wallclock_per_instance_s": f"{timing['countdown']['dream_median']:.2f}",
            "k": 1,
        },
        {
            "method": "Greedy (Cond. A)",
            "base_model": "Qwen2.5-7B",
            "task": "Mini Sudoku",
            "accuracy": f"{float(greedy_sdk['accuracy']):.4f}",
            "accuracy_std": "",
            "wallclock_per_instance_s": f"{timing['sudoku']['qwen_median']:.2f}",
            "k": 1,
        },
        {
            "method": "Best-of-k (Cond. B)",
            "base_model": "Qwen2.5-7B",
            "task": "Mini Sudoku",
            "accuracy": f"{float(bok_sdk_mean['accuracy']):.4f}",
            "accuracy_std": f"{float(bok_sdk_std['accuracy']):.4f}",
            "wallclock_per_instance_s": f"~{timing['sudoku']['dream_median']:.2f}",
            "k": timing["sudoku"]["k_median"],
        },
        {
            "method": "Diffusion (Cond. C)",
            "base_model": "Dream-v0-Base-7B",
            "task": "Mini Sudoku",
            "accuracy": f"{float(dream_sdk['accuracy']):.4f}",
            "accuracy_std": "",
            "wallclock_per_instance_s": f"{timing['sudoku']['dream_median']:.2f}",
            "k": 1,
        },
    ]

    out_csv = os.path.join(TABLES_DIR, "main_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "base_model", "task", "accuracy", "accuracy_std",
            "wallclock_per_instance_s", "k"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Main results CSV saved to {out_csv}")

    for row in rows:
        acc_str = row["accuracy"]
        if row["accuracy_std"]:
            acc_str += f" +/- {row['accuracy_std']}"
        print(f"  {row['method']:25s} | {row['base_model']:20s} | {row['task']:12s} | "
              f"{acc_str:18s} | {row['wallclock_per_instance_s']:>10s} | k={row['k']}")

    tasks = ["Countdown", "Mini Sudoku"]
    methods = ["Greedy (Cond. A)", "Best-of-k (Cond. B)", "Diffusion (Cond. C)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(tasks))
    width = 0.25

    for i, method in enumerate(methods):
        vals = []
        errs = []
        for task in tasks:
            row = next(r for r in rows if r["method"] == method and r["task"] == task)
            vals.append(float(row["accuracy"]) * 100)
            if row["accuracy_std"]:
                errs.append(float(row["accuracy_std"]) * 100)
            else:
                errs.append(0)
        bars = ax.bar(x + i * width, vals, width, yerr=errs, label=method,
                      color=colors[i], capsize=4, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Compute-Matched Diffusion vs AR Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "main_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
