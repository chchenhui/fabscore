"""Plotting utilities for DUT experiment results.

Generates bar charts for per-family accuracy, condition comparisons,
and coupling diagnostics.
"""

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_accuracy_by_condition(
    results: dict[str, dict[str, Any]],
    output_path: str,
    title: str = "Main Accuracy by Condition",
) -> None:
    conditions = sorted(results.keys())
    accuracies = [results[c]["main_accuracy"] for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(conditions, accuracies, color=sns.color_palette("muted"))
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_family(
    results: dict[str, dict[str, Any]],
    output_path: str,
    title: str = "Accuracy by Family and Condition",
) -> None:
    conditions = sorted(results.keys())
    families = sorted(
        set(f for r in results.values() for f in r.get("per_family", {}).keys())
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(families))
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        accs = [
            results[cond].get("per_family", {}).get(fam, {}).get("accuracy", 0)
            for fam in families
        ]
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], accs, width, label=f"Cond {cond}")

    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
