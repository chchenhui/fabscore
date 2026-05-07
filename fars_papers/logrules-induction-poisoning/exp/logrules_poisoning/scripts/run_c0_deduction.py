"""Run C0 clean rule-based deduction on all datasets and seeds.
Loads ranked rules, formats them into the deduction prompt, parses test+canary
logs via vLLM-served Qwen2.5-7B-Instruct, computes metrics.
Usage: python scripts/run_c0_deduction.py [--debug] [--api-base URL]
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.deduction.parse import parse_logs
from logrules_poisoning.src.evaluation.metrics import evaluate_all, compute_pa

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]


def load_split(dataset: str, seed: int, split_name: str):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_ranked_rules(condition: str, dataset: str, seed: int):
    path = PROJECT_ROOT / "outputs" / "rules" / condition / dataset / f"seed_{seed}" / "ranked_rules.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["ranked_rules"]


def format_rules_numbered(rules):
    return "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))


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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--condition", type=str, default="c0_clean")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-concurrency", type=int, default=64)
    args = parser.parse_args()

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_with_rules.txt"
    prompt_template = prompt_path.read_text().strip()

    wait_for_server(args.api_base)

    results_rows = []
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"C0 Deduction: {dataset} / seed_{seed}")
            print(f"{'='*60}")

            rules = load_ranked_rules(args.condition, dataset, seed)
            rules_text = format_rules_numbered(rules)
            print(f"Loaded {len(rules)} ranked rules")

            test_records = load_split(dataset, seed, "test")
            canary_records = load_split(dataset, seed, "canary")

            if args.debug:
                test_records = test_records[:20]
                canary_records = canary_records[:10]
            elif args.subsample > 0:
                test_records = test_records[:args.subsample]

            test_logs = [r["raw_log"] for r in test_records]
            test_gts = [r["template"] for r in test_records]
            test_eids = [r["event_id"] for r in test_records]

            canary_logs = [r["raw_log"] for r in canary_records]
            canary_gts = [r["template"] for r in canary_records]
            canary_eids = [r["event_id"] for r in canary_records]

            print(f"Parsing {len(test_logs)} test logs...")
            t0 = time.time()
            test_preds = parse_logs(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                prompt_template=prompt_template,
                logs=test_logs,
                rules=rules_text,
                max_concurrency=args.max_concurrency,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s ({len(test_logs)/max(elapsed,0.1):.1f} logs/s)")

            print(f"Parsing {len(canary_logs)} canary logs...")
            t0 = time.time()
            canary_preds = parse_logs(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                prompt_template=prompt_template,
                logs=canary_logs,
                rules=rules_text,
                max_concurrency=args.max_concurrency,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            out_dir = PROJECT_ROOT / "outputs" / "predictions" / args.condition / dataset / f"seed_{seed}"
            save_predictions(test_preds, test_records, out_dir / "predictions.jsonl")
            save_predictions(canary_preds, canary_records, out_dir / "canary_predictions.jsonl")

            test_metrics = evaluate_all(test_preds, test_gts, test_eids)
            canary_pa = compute_pa(canary_preds, canary_gts)

            print(f"  Test:   PA={test_metrics['PA']:.4f}  FTA={test_metrics['FTA']:.4f}  "
                  f"WR={test_metrics['wildcard_ratio']:.4f}")
            print(f"  Canary: PA={canary_pa:.4f}")

            results_rows.append({
                "dataset": dataset,
                "seed": seed,
                "test_PA": test_metrics["PA"],
                "test_FTA": test_metrics["FTA"],
                "test_wildcard_ratio": test_metrics["wildcard_ratio"],
                "canary_PA": canary_pa,
                "num_test_logs": len(test_logs),
                "num_canary_logs": len(canary_logs),
                "num_rules": len(rules),
            })

    csv_path = results_dir / f"{args.condition}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nResults saved to {csv_path}")

    print("\n\n=== SUMMARY ===")
    by_dataset = defaultdict(list)
    for row in results_rows:
        by_dataset[row["dataset"]].append(row)

    variance_data = {}
    for dataset in args.datasets:
        rows = by_dataset[dataset]
        pa_vals = [r["test_PA"] for r in rows]
        fta_vals = [r["test_FTA"] for r in rows]
        wr_vals = [r["test_wildcard_ratio"] for r in rows]
        canary_vals = [r["canary_PA"] for r in rows]

        pa_mean = statistics.mean(pa_vals)
        pa_std = statistics.stdev(pa_vals) if len(pa_vals) > 1 else 0.0
        fta_mean = statistics.mean(fta_vals)
        fta_std = statistics.stdev(fta_vals) if len(fta_vals) > 1 else 0.0
        wr_mean = statistics.mean(wr_vals)
        wr_std = statistics.stdev(wr_vals) if len(wr_vals) > 1 else 0.0
        canary_mean = statistics.mean(canary_vals)
        canary_std = statistics.stdev(canary_vals) if len(canary_vals) > 1 else 0.0

        threshold_x = max(0.05, 2 * pa_std)

        print(f"{dataset}:")
        print(f"  Test PA={pa_mean:.4f}+/-{pa_std:.4f}  FTA={fta_mean:.4f}+/-{fta_std:.4f}  "
              f"WR={wr_mean:.4f}+/-{wr_std:.4f}")
        print(f"  Canary PA={canary_mean:.4f}+/-{canary_std:.4f}")
        print(f"  Phase-0 threshold X={threshold_x:.4f}")

        variance_data[dataset] = {
            "mean_PA": round(pa_mean, 6),
            "std_PA": round(pa_std, 6),
            "threshold_X": round(threshold_x, 6),
            "mean_FTA": round(fta_mean, 6),
            "std_FTA": round(fta_std, 6),
            "mean_canary_PA": round(canary_mean, 6),
            "std_canary_PA": round(canary_std, 6),
        }

    variance_path = results_dir / "clean_variance.json"
    with open(variance_path, "w") as f:
        json.dump(variance_data, f, indent=2)
    print(f"\nClean variance saved to {variance_path}")


if __name__ == "__main__":
    main()
