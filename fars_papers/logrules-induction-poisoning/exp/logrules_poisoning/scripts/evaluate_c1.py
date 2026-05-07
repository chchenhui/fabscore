"""Compute C1 poisoned metrics across all qualifying configs.
Loads C1 test/canary predictions and Phase-0v2 clean baselines (same pipeline
session as the C1 poisoned predictions), computes PA, FTA, wildcard_ratio,
canary_PA, template_disagreement, and delta_wildcard.
Outputs results/c1_poisoned.csv with mean+/-std summaries.
Usage: python scripts/evaluate_c1.py
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.metrics import compute_pa, evaluate_all
from logrules_poisoning.src.evaluation.diagnostics import template_disagreement


def load_predictions(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(dataset, seed, split_name):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    config_path = PROJECT_ROOT / "results" / "c1_config.json"
    with open(config_path) as f:
        config = json.load(f)

    payloads = config["qualifying_payloads"]
    k_values = config["k_values"]
    datasets = config["datasets"]
    seeds = config["seeds"]

    c0_data = {}
    for dataset in datasets:
        for seed in seeds:
            c0_pred_path = (
                PROJECT_ROOT / "outputs" / "predictions" / "phase0v2_clean"
                / dataset / f"seed_{seed}" / "test_predictions.jsonl"
            )
            c0_canary_path = (
                PROJECT_ROOT / "outputs" / "predictions" / "phase0v2_clean"
                / dataset / f"seed_{seed}" / "canary_predictions.jsonl"
            )
            c0_test_recs = load_predictions(c0_pred_path)
            c0_canary_recs = load_predictions(c0_canary_path)
            c0_test_preds = [r["predicted_template"] for r in c0_test_recs]
            c0_test_gts = [r["template"] for r in c0_test_recs]
            c0_test_eids = [r["event_id"] for r in c0_test_recs]
            c0_canary_preds = [r["predicted_template"] for r in c0_canary_recs]
            c0_canary_gts = [r["template"] for r in c0_canary_recs]
            c0_metrics = evaluate_all(c0_test_preds, c0_test_gts, c0_test_eids)
            c0_canary_pa = compute_pa(c0_canary_preds, c0_canary_gts)
            c0_data[(dataset, seed)] = {
                "test_PA": c0_metrics["PA"],
                "test_FTA": c0_metrics["FTA"],
                "test_wildcard_ratio": c0_metrics["wildcard_ratio"],
                "canary_PA": c0_canary_pa,
            }
            print(f"C0 baseline {dataset}/seed_{seed}: PA={c0_metrics['PA']:.4f}")

    rows = []

    for payload in payloads:
        for k in k_values:
            for dataset in datasets:
                for seed in seeds:
                    print(f"Evaluating {payload}/k{k}/{dataset}/seed_{seed}...")

                    c1_pred_dir = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}"
                    )

                    c1_test_recs = load_predictions(c1_pred_dir / "test_predictions.jsonl")
                    c1_canary_recs = load_predictions(c1_pred_dir / "canary_predictions.jsonl")

                    c1_test_preds = [r["predicted_template"] for r in c1_test_recs]
                    c1_test_gts = [r["template"] for r in c1_test_recs]
                    c1_test_eids = [r["event_id"] for r in c1_test_recs]

                    c1_canary_preds = [r["predicted_template"] for r in c1_canary_recs]
                    c1_canary_gts = [r["template"] for r in c1_canary_recs]

                    test_metrics = evaluate_all(c1_test_preds, c1_test_gts, c1_test_eids)
                    canary_pa = compute_pa(c1_canary_preds, c1_canary_gts)

                    c0_pred_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "phase0v2_clean"
                        / dataset / f"seed_{seed}" / "test_predictions.jsonl"
                    )
                    c0_test_recs = load_predictions(c0_pred_path)
                    c0_test_preds = [r["predicted_template"] for r in c0_test_recs]

                    td = template_disagreement(c0_test_preds, c1_test_preds)

                    c0_wr = c0_data[(dataset, seed)]["test_wildcard_ratio"]
                    delta_wr = test_metrics["wildcard_ratio"] - c0_wr

                    row = {
                        "payload": payload,
                        "k": k,
                        "dataset": dataset,
                        "seed": seed,
                        "test_PA": round(test_metrics["PA"], 6),
                        "test_FTA": round(test_metrics["FTA"], 6),
                        "test_wildcard_ratio": round(test_metrics["wildcard_ratio"], 6),
                        "canary_PA": round(canary_pa, 6),
                        "template_disagreement": round(td, 6),
                        "delta_wildcard": round(delta_wr, 6),
                    }
                    rows.append(row)

                    print(f"  test_PA={row['test_PA']:.4f} FTA={row['test_FTA']:.4f} "
                          f"WR={row['test_wildcard_ratio']:.4f} canary_PA={row['canary_PA']:.4f} "
                          f"TD={row['template_disagreement']:.4f} dWR={row['delta_wildcard']:.4f}")

    csv_path = PROJECT_ROOT / "results" / "c1_poisoned.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: Mean +/- Std across 3 seeds")
    print("=" * 80)

    grouped = defaultdict(list)
    for row in rows:
        key = (row["payload"], row["k"], row["dataset"])
        grouped[key].append(row)

    summary_rows = []
    for (payload, k, dataset), seed_rows in sorted(grouped.items()):
        pa_vals = [r["test_PA"] for r in seed_rows]
        fta_vals = [r["test_FTA"] for r in seed_rows]
        wr_vals = [r["test_wildcard_ratio"] for r in seed_rows]
        canary_vals = [r["canary_PA"] for r in seed_rows]
        td_vals = [r["template_disagreement"] for r in seed_rows]
        dwr_vals = [r["delta_wildcard"] for r in seed_rows]

        def fmt(vals):
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:.4f}+/-{s:.4f}"

        c0_pa_mean = statistics.mean([c0_data[(dataset, s)]["test_PA"] for s in [42, 123, 456]])

        print(f"\n{payload}/k{k}/{dataset}:")
        print(f"  test_PA={fmt(pa_vals)}  (C0={c0_pa_mean:.4f})")
        print(f"  test_FTA={fmt(fta_vals)}")
        print(f"  test_WR={fmt(wr_vals)}")
        print(f"  canary_PA={fmt(canary_vals)}")
        print(f"  TD={fmt(td_vals)}")
        print(f"  delta_WR={fmt(dwr_vals)}")

        summary_rows.append({
            "payload": payload,
            "k": k,
            "dataset": dataset,
            "mean_test_PA": round(statistics.mean(pa_vals), 6),
            "std_test_PA": round(statistics.stdev(pa_vals) if len(pa_vals) > 1 else 0.0, 6),
            "mean_test_FTA": round(statistics.mean(fta_vals), 6),
            "mean_canary_PA": round(statistics.mean(canary_vals), 6),
            "mean_TD": round(statistics.mean(td_vals), 6),
            "mean_delta_WR": round(statistics.mean(dwr_vals), 6),
        })

    summary_path = PROJECT_ROOT / "results" / "c1_poisoned_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
