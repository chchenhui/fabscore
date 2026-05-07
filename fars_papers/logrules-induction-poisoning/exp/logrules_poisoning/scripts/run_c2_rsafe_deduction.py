"""Parse canary + test logs using R_safe rules for 9 (dataset, seed) combos.
Used in C2 admission control defense -- generates R_safe predictions that
the admission gate compares against R_gen (C1 poisoned) predictions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.deduction.parse import parse_logs
from logrules_poisoning.src.evaluation.metrics import compute_pa
from logrules_poisoning.src.defense.admission_control import R_SAFE, format_rules

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
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--datasets", type=str, default="BGL,Linux,HDFS")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_with_rules.txt"
    prompt_template = prompt_path.read_text().strip()

    wait_for_server(args.api_base)

    rules_text = format_rules(R_SAFE)
    print(f"R_safe rules ({len(R_SAFE)} rules):")
    print(rules_text)
    print()

    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"R_safe Deduction: {dataset} / seed_{seed}")
            print(f"{'='*60}")

            canary_records = load_split(dataset, seed, "canary")
            test_records = load_split(dataset, seed, "test")

            canary_logs = [r["raw_log"] for r in canary_records]
            canary_gts = [r["template"] for r in canary_records]
            test_logs = [r["raw_log"] for r in test_records]
            test_gts = [r["template"] for r in test_records]

            print(f"Parsing {len(canary_logs)} canary logs with R_safe...")
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
            canary_pa = compute_pa(canary_preds, canary_gts)
            print(f"  Done in {elapsed:.1f}s | Canary PA = {canary_pa:.4f}")

            print(f"Parsing {len(test_logs)} test logs with R_safe...")
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
            test_pa = compute_pa(test_preds, test_gts)
            print(f"  Done in {elapsed:.1f}s | Test PA = {test_pa:.4f}")

            out_dir = PROJECT_ROOT / "outputs" / "predictions" / "c2_defense" / "r_safe" / dataset / f"seed_{seed}"
            save_predictions(canary_preds, canary_records, out_dir / "canary_predictions.jsonl")
            save_predictions(test_preds, test_records, out_dir / "test_predictions.jsonl")
            print(f"  Saved to {out_dir}")

    print("\nR_safe deduction complete for all datasets and seeds.")


if __name__ == "__main__":
    main()
