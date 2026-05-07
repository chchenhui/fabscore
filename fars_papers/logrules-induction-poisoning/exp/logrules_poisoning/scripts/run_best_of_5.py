"""Run best-of-5 log parsing on all datasets and seeds.
Generates 5 candidates per log at temperature=0.7, selects first valid template.
Usage: python scripts/run_best_of_5.py [--debug] [--subsample N]
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from logrules_poisoning.src.deduction.best_of_5 import parse_logs_best_of_5
from logrules_poisoning.src.evaluation.metrics import evaluate_all

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]


def load_split(dataset: str, seed: int, split_name: str):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def save_predictions(predictions, records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec, pred in zip(records, predictions):
            out = {**rec, "predicted_template": pred}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def wait_for_server(api_base: str, timeout: int = 600):
    start = time.time()
    url = f"{api_base}/models"
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"vLLM server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM server not ready after {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on 20 logs only")
    parser.add_argument("--subsample", type=int, default=0, help="Subsample N logs from test")
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--n", type=int, default=5, help="Number of candidates per log")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_zero_shot.txt"
    prompt_template = prompt_path.read_text().strip()

    wait_for_server(args.api_base)

    results_rows = []
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset} / seed_{seed}")
            print(f"{'='*60}")

            records = load_split(dataset, seed, "test")

            if args.debug:
                records = records[:20]
            elif args.subsample > 0:
                records = records[:args.subsample]

            logs = [r["raw_log"] for r in records]
            ground_truths = [r["template"] for r in records]
            event_ids = [r["event_id"] for r in records]

            print(f"Parsing {len(logs)} logs (best-of-{args.n}, temp={args.temperature})...")
            t0 = time.time()
            predictions = parse_logs_best_of_5(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                prompt_template=prompt_template,
                logs=logs,
                n=args.n,
                temperature=args.temperature,
                max_concurrency=args.max_concurrency,
            )
            elapsed = time.time() - t0
            print(f"Done in {elapsed:.1f}s ({len(logs)/elapsed:.1f} logs/s)")

            out_path = (PROJECT_ROOT / "outputs" / "predictions" / "best_of_5" /
                        dataset / f"seed_{seed}" / "predictions.jsonl")
            save_predictions(predictions, records, out_path)

            metrics = evaluate_all(predictions, ground_truths, event_ids)
            print(f"  PA={metrics['PA']:.4f}  FTA={metrics['FTA']:.4f}  "
                  f"wildcard_ratio={metrics['wildcard_ratio']:.4f}")

            results_rows.append({
                "dataset": dataset,
                "seed": seed,
                "PA": metrics["PA"],
                "FTA": metrics["FTA"],
                "FTA_precision": metrics["FTA_precision"],
                "FTA_recall": metrics["FTA_recall"],
                "wildcard_ratio": metrics["wildcard_ratio"],
                "num_logs": len(logs),
            })

    csv_path = results_dir / "best_of_5.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nResults saved to {csv_path}")

    print("\n\n=== SUMMARY ===")
    by_dataset = defaultdict(list)
    for row in results_rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in DATASETS:
        rows = by_dataset[dataset]
        pa_vals = [r["PA"] for r in rows]
        fta_vals = [r["FTA"] for r in rows]
        wr_vals = [r["wildcard_ratio"] for r in rows]
        pa_mean = statistics.mean(pa_vals)
        pa_std = statistics.stdev(pa_vals) if len(pa_vals) > 1 else 0
        fta_mean = statistics.mean(fta_vals)
        fta_std = statistics.stdev(fta_vals) if len(fta_vals) > 1 else 0
        wr_mean = statistics.mean(wr_vals)
        wr_std = statistics.stdev(wr_vals) if len(wr_vals) > 1 else 0
        print(f"{dataset}: PA={pa_mean:.4f}+/-{pa_std:.4f}  "
              f"FTA={fta_mean:.4f}+/-{fta_std:.4f}  "
              f"wildcard_ratio={wr_mean:.4f}+/-{wr_std:.4f}")


if __name__ == "__main__":
    main()
