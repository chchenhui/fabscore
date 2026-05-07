"""Cross-model comparison analysis: LLaMA-3-8B vs Qwen2.5-7B on BGL C0/C1/C2.
Loads Qwen results from existing CSVs and LLaMA results from cross_model_llama.csv.
Generates cross_model_comparison.csv and cross_model_bgl.png bar chart.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from logrules_poisoning.src.evaluation.metrics import evaluate_all, compute_pa

SEEDS = [42, 123, 456]
K_VALUES = [1, 3]
PAYLOAD = "D"
DATASET = "BGL"


def load_predictions(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_split(dataset, seed, split_name):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def compute_metrics_from_predictions(pred_path, dataset, seed):
    pred_recs = load_predictions(pred_path)
    preds = [r["predicted_template"] for r in pred_recs]
    test_recs = load_split(dataset, seed, "test")
    gts = [r["template"] for r in test_recs]
    eids = [r["event_id"] for r in test_recs]
    return evaluate_all(preds, gts, eids)


def load_qwen_c0_bgl():
    rows = []
    for seed in SEEDS:
        pred_path = PROJECT_ROOT / "outputs" / "predictions" / "phase0v2_clean" / DATASET / f"seed_{seed}" / "test_predictions.jsonl"
        metrics = compute_metrics_from_predictions(pred_path, DATASET, seed)
        rows.append({
            "model": "Qwen2.5-7B-Instruct",
            "condition": "C0",
            "k": "-",
            "seed": seed,
            "PA": metrics["PA"],
            "FTA": metrics["FTA"],
            "wildcard_ratio": metrics["wildcard_ratio"],
        })
    return rows


def load_qwen_c1_bgl():
    rows = []
    with open(RESULTS_DIR / "c1_poisoned.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["dataset"] == "BGL" and r["payload"] == PAYLOAD and int(r["k"]) in K_VALUES:
                rows.append({
                    "model": "Qwen2.5-7B-Instruct",
                    "condition": "C1",
                    "k": int(r["k"]),
                    "seed": int(r["seed"]),
                    "PA": float(r["test_PA"]),
                    "FTA": float(r["test_FTA"]),
                    "wildcard_ratio": float(r["test_wildcard_ratio"]),
                })
    return rows


def load_qwen_c2_bgl():
    rows = []
    with open(RESULTS_DIR / "c2_defense.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["dataset"] == "BGL" and r["payload"] == PAYLOAD and int(r["k"]) in K_VALUES:
                rows.append({
                    "model": "Qwen2.5-7B-Instruct",
                    "condition": "C2",
                    "k": int(r["k"]),
                    "seed": int(r["seed"]),
                    "PA": float(r["test_PA"]),
                    "FTA": float(r["test_FTA"]),
                    "wildcard_ratio": float(r["test_wildcard_ratio"]),
                })
    return rows


def load_llama_results():
    rows = []
    with open(RESULTS_DIR / "cross_model_llama.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k_val = r["k"] if r["k"] == "-" else int(r["k"])
            rows.append({
                "model": "LLaMA-3-8B-Instruct",
                "condition": r["condition"],
                "k": k_val,
                "seed": int(r["seed"]),
                "PA": float(r["PA"]),
                "FTA": float(r["FTA"]),
                "wildcard_ratio": float(r["wildcard_ratio"]),
            })
    return rows


def main():
    qwen_c0 = load_qwen_c0_bgl()
    qwen_c1 = load_qwen_c1_bgl()
    qwen_c2 = load_qwen_c2_bgl()
    llama_all = load_llama_results()

    all_rows = qwen_c0 + qwen_c1 + qwen_c2 + llama_all

    csv_path = RESULTS_DIR / "cross_model_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "k", "seed", "PA", "FTA", "wildcard_ratio"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {csv_path} ({len(all_rows)} rows)")

    print("\n" + "=" * 90)
    print("CROSS-MODEL COMPARISON: BGL (Payload D)")
    print("=" * 90)

    def mean_std(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    conditions = [
        ("C0", "-"),
        ("C1", 1),
        ("C1", 3),
        ("C2", 1),
        ("C2", 3),
    ]

    summary = {}
    print(f"\n{'Model':<24} {'Cond':<6} {'k':<4} {'PA_mean':>8} {'PA_std':>8} {'FTA_mean':>8} {'WR_mean':>8}")
    print("-" * 80)

    for cond, k in conditions:
        for model_name in ["Qwen2.5-7B-Instruct", "LLaMA-3-8B-Instruct"]:
            matching = [r for r in all_rows
                       if r["model"] == model_name and r["condition"] == cond and r["k"] == k]
            if not matching:
                continue
            pa_m, pa_s = mean_std([r["PA"] for r in matching])
            fta_m, _ = mean_std([r["FTA"] for r in matching])
            wr_m, _ = mean_std([r["wildcard_ratio"] for r in matching])
            print(f"{model_name:<24} {cond:<6} {str(k):<4} {pa_m:>8.4f} {pa_s:>8.4f} {fta_m:>8.4f} {wr_m:>8.4f}")
            summary[(model_name, cond, k)] = {"PA_mean": pa_m, "PA_std": pa_s, "FTA_mean": fta_m, "WR_mean": wr_m}

    print("\n" + "=" * 90)
    print("KEY ANALYSIS")
    print("=" * 90)

    qwen_c0_pa = summary[("Qwen2.5-7B-Instruct", "C0", "-")]["PA_mean"]
    llama_c0_pa = summary[("LLaMA-3-8B-Instruct", "C0", "-")]["PA_mean"]

    print(f"\n1. Baseline (C0) PA: Qwen={qwen_c0_pa:.4f}, LLaMA={llama_c0_pa:.4f}")

    for k in K_VALUES:
        qwen_c1_pa = summary[("Qwen2.5-7B-Instruct", "C1", k)]["PA_mean"]
        llama_c1_pa = summary[("LLaMA-3-8B-Instruct", "C1", k)]["PA_mean"]
        qwen_drop = qwen_c0_pa - qwen_c1_pa
        llama_drop = llama_c0_pa - llama_c1_pa

        print(f"\n2. C1 PA drop (k={k}):")
        print(f"   Qwen:  C0={qwen_c0_pa:.4f} -> C1={qwen_c1_pa:.4f}  drop={qwen_drop:+.4f}")
        print(f"   LLaMA: C0={llama_c0_pa:.4f} -> C1={llama_c1_pa:.4f}  drop={llama_drop:+.4f}")
        print(f"   Direction consistent: {'YES' if (qwen_drop > 0) == (llama_drop > 0) else 'NO (OPPOSITE)'}")

    for k in K_VALUES:
        qwen_c2_pa = summary.get(("Qwen2.5-7B-Instruct", "C2", k), {}).get("PA_mean", None)
        llama_c2_pa = summary.get(("LLaMA-3-8B-Instruct", "C2", k), {}).get("PA_mean", None)
        if qwen_c2_pa is not None and llama_c2_pa is not None:
            qwen_c1_pa = summary[("Qwen2.5-7B-Instruct", "C1", k)]["PA_mean"]
            llama_c1_pa = summary[("LLaMA-3-8B-Instruct", "C1", k)]["PA_mean"]
            print(f"\n3. C2 defense (k={k}):")
            print(f"   Qwen:  C1={qwen_c1_pa:.4f} -> C2={qwen_c2_pa:.4f}  recovery={qwen_c2_pa - qwen_c1_pa:+.4f}")
            print(f"   LLaMA: C1={llama_c1_pa:.4f} -> C2={llama_c2_pa:.4f}  recovery={llama_c2_pa - llama_c1_pa:+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cond_labels = ["C0", "C1\nk=1", "C1\nk=3", "C2\nk=1", "C2\nk=3"]
    cond_keys = [("C0", "-"), ("C1", 1), ("C1", 3), ("C2", 1), ("C2", 3)]
    x = np.arange(len(cond_labels))
    width = 0.35

    for ax_idx, (metric, ylabel) in enumerate([("PA_mean", "Parsing Accuracy"), ("FTA_mean", "FTA"), ("WR_mean", "Wildcard Ratio")]):
        qwen_vals = []
        llama_vals = []
        qwen_errs = []
        llama_errs = []

        for cond, k in cond_keys:
            q = summary.get(("Qwen2.5-7B-Instruct", cond, k), {})
            l = summary.get(("LLaMA-3-8B-Instruct", cond, k), {})
            qwen_vals.append(q.get(metric, 0))
            llama_vals.append(l.get(metric, 0))
            qwen_errs.append(q.get("PA_std", 0) if metric == "PA_mean" else 0)
            llama_errs.append(l.get("PA_std", 0) if metric == "PA_mean" else 0)

        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, qwen_vals, width, label="Qwen2.5-7B", color="#4C72B0",
                       yerr=qwen_errs if metric == "PA_mean" else None, capsize=3)
        bars2 = ax.bar(x + width/2, llama_vals, width, label="LLaMA-3-8B", color="#DD8452",
                       yerr=llama_errs if metric == "PA_mean" else None, capsize=3)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(qwen_vals), max(llama_vals)) * 1.3 + 0.02)

        for bar in bars1:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                       ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}',
                       ha='center', va='bottom', fontsize=7)

    fig.suptitle("Cross-Model Comparison: BGL (Payload D)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "cross_model_bgl.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
