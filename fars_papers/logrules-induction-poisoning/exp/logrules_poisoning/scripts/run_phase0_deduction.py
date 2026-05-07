"""Phase-0 deduction: parse 50-log canary set for clean + 6 poisoned configs on BGL seed=0.
Expects a running vLLM server. Saves predictions to outputs/predictions/phase0/.
Usage: python scripts/run_phase0_deduction.py [--api-base URL] [--model MODEL]
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

DATASET = "BGL"
SEED = 0


def load_split(dataset, seed, split_name):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_ranked_rules(condition_dir, dataset, seed):
    path = PROJECT_ROOT / "outputs" / "rules" / condition_dir / dataset / f"seed_{seed}" / "ranked_rules.json"
    with open(path) as f:
        data = json.load(f)
    return data["ranked_rules"]


def format_rules_numbered(rules):
    return "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))


def save_predictions(predictions, records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec, pred in zip(records, predictions):
            out = {**rec, "predicted_template": pred}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def wait_for_server(api_base, timeout=600):
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
    parser.add_argument("--api-base", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-concurrency", type=int, default=64)
    args = parser.parse_args()

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_with_rules.txt"
    prompt_template = prompt_path.read_text().strip()

    wait_for_server(args.api_base)

    canary_records = load_split(DATASET, SEED, "canary")
    canary_logs = [r["raw_log"] for r in canary_records]
    canary_gts = [r["template"] for r in canary_records]
    print(f"Loaded {len(canary_records)} canary records for {DATASET}/seed_{SEED}")

    configs = [
        ("c0_clean", "c0_clean"),
    ]
    for payload in ["A", "B", "C"]:
        for k in [1, 3]:
            configs.append((f"{payload}_k{k}", f"phase0/{payload}_k{k}"))

    results = {}

    for config_name, rules_dir in configs:
        print(f"\n--- {config_name} ---")
        rules = load_ranked_rules(rules_dir, DATASET, SEED)
        rules_text = format_rules_numbered(rules)
        print(f"  {len(rules)} rules loaded")

        t0 = time.time()
        preds = parse_logs(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            prompt_template=prompt_template,
            logs=canary_logs,
            rules=rules_text,
            max_concurrency=args.max_concurrency,
        )
        elapsed = time.time() - t0
        print(f"  Parsed {len(preds)} logs in {elapsed:.1f}s")

        pa = compute_pa(preds, canary_gts)
        print(f"  Canary PA = {pa:.4f}")

        if config_name == "c0_clean":
            out_dir = PROJECT_ROOT / "outputs" / "predictions" / "c0_clean" / DATASET / f"seed_{SEED}"
        else:
            out_dir = PROJECT_ROOT / "outputs" / "predictions" / "phase0" / config_name / DATASET / f"seed_{SEED}"
        save_predictions(preds, canary_records, out_dir / "canary_predictions.jsonl")

        results[config_name] = {
            "canary_PA": pa,
            "num_canary": len(preds),
            "num_rules": len(rules),
        }

    summary_path = PROJECT_ROOT / "outputs" / "phase0" / "deduction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDeduction summary saved to {summary_path}")
    print("\nPhase-0 deduction complete.")


if __name__ == "__main__":
    main()
