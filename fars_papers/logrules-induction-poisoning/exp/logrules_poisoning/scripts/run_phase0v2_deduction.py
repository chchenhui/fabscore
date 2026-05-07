"""Phase-0v2 deduction + evaluation: parse canary and test sets for clean + poisoned configs.
Expects a running vLLM server. Evaluates PA, FTA, wildcard_ratio, template_disagreement.
Usage: python scripts/run_phase0v2_deduction.py [--api-base URL] [--model MODEL]
       [--datasets BGL,Linux,HDFS] [--seeds 42,123,456] [--eval-test]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.deduction.parse import parse_logs
from logrules_poisoning.src.evaluation.metrics import compute_pa, compute_fta, compute_wildcard_ratio, evaluate_all
from logrules_poisoning.src.evaluation.diagnostics import template_disagreement

NEW_PAYLOADS = ["D", "E", "F"]
ATTACK_BUDGETS = [1, 3, 5, 7]


def load_split(dataset, seed, split_name):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_ranked_rules(rules_dir):
    path = rules_dir / "ranked_rules.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("ranked_rules", [])


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


def run_deduction(args, logs, rules, prompt_template):
    rules_text = format_rules_numbered(rules) if rules else ""
    preds = parse_logs(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        prompt_template=prompt_template,
        logs=logs,
        rules=rules_text,
        max_concurrency=args.max_concurrency,
    )
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--datasets", default="BGL,Linux,HDFS")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--eval-test", action="store_true", help="Also evaluate on full test set")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_with_rules.txt"
    prompt_template = prompt_path.read_text().strip()

    wait_for_server(args.api_base)

    clean_variance_path = PROJECT_ROOT / "results" / "clean_variance.json"
    with open(clean_variance_path) as f:
        clean_variance = json.load(f)

    all_rows = []
    all_results = {}

    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'#'*70}")
            print(f"Dataset: {dataset}, Seed: {seed}")
            print(f"{'#'*70}")

            threshold_x = clean_variance.get(dataset, {}).get("threshold_X", 0.05)

            canary_records = load_split(dataset, seed, "canary")
            canary_logs = [r["raw_log"] for r in canary_records]
            canary_gts = [r["template"] for r in canary_records]
            canary_eids = [r["event_id"] for r in canary_records]

            test_records = load_split(dataset, seed, "test") if args.eval_test else []
            test_logs = [r["raw_log"] for r in test_records]
            test_gts = [r["template"] for r in test_records]
            test_eids = [r["event_id"] for r in test_records]

            clean_rules_dir = PROJECT_ROOT / "outputs" / "rules" / "phase0v2_clean" / dataset / f"seed_{seed}"
            clean_rules = load_ranked_rules(clean_rules_dir)
            if not clean_rules:
                print(f"  WARNING: No clean rules found at {clean_rules_dir}, skipping")
                continue

            print(f"\n--- CLEAN (C0) ---")
            print(f"  {len(clean_rules)} rules")
            c0_canary_preds = run_deduction(args, canary_logs, clean_rules, prompt_template)
            c0_canary_pa = compute_pa(c0_canary_preds, canary_gts)
            c0_canary_wr = compute_wildcard_ratio(c0_canary_preds)
            print(f"  Canary PA = {c0_canary_pa:.4f}, WR = {c0_canary_wr:.4f}")

            c0_pred_dir = PROJECT_ROOT / "outputs" / "predictions" / "phase0v2_clean" / dataset / f"seed_{seed}"
            save_predictions(c0_canary_preds, canary_records, c0_pred_dir / "canary_predictions.jsonl")

            c0_test_pa = None
            c0_test_wr = None
            c0_test_preds = None
            if args.eval_test and test_records:
                c0_test_preds = run_deduction(args, test_logs, clean_rules, prompt_template)
                c0_test_pa = compute_pa(c0_test_preds, test_gts)
                c0_test_wr = compute_wildcard_ratio(c0_test_preds)
                print(f"  Test PA = {c0_test_pa:.4f}, WR = {c0_test_wr:.4f}")
                save_predictions(c0_test_preds, test_records, c0_pred_dir / "test_predictions.jsonl")

            for payload in NEW_PAYLOADS:
                for k in ATTACK_BUDGETS:
                    config_name = f"{payload}_k{k}"
                    rules_dir = PROJECT_ROOT / "outputs" / "rules" / f"phase0v2/{payload}_k{k}" / dataset / f"seed_{seed}"
                    c1_rules = load_ranked_rules(rules_dir)
                    if not c1_rules:
                        print(f"\n--- {config_name} --- SKIPPED (no rules)")
                        continue

                    print(f"\n--- {config_name} ---")
                    print(f"  {len(c1_rules)} rules")

                    c1_canary_preds = run_deduction(args, canary_logs, c1_rules, prompt_template)
                    c1_canary_pa = compute_pa(c1_canary_preds, canary_gts)
                    c1_canary_wr = compute_wildcard_ratio(c1_canary_preds)
                    td = template_disagreement(c0_canary_preds, c1_canary_preds)
                    pa_drop = c0_canary_pa - c1_canary_pa

                    passes_td = td >= 0.10
                    passes_pa = pa_drop >= threshold_x
                    passes = passes_td and passes_pa

                    print(f"  Canary PA = {c1_canary_pa:.4f} (drop={pa_drop:+.4f}, thresh={threshold_x:.4f}) {'PASS' if passes_pa else 'FAIL'}")
                    print(f"  Canary WR = {c1_canary_wr:.4f} (C0={c0_canary_wr:.4f}, diff={c1_canary_wr - c0_canary_wr:+.4f})")
                    print(f"  Template Disagreement = {td:.4f} {'PASS' if passes_td else 'FAIL'}")
                    print(f"  Overall: {'PASS' if passes else 'FAIL'}")

                    c1_pred_dir = PROJECT_ROOT / "outputs" / "predictions" / f"phase0v2/{config_name}" / dataset / f"seed_{seed}"
                    save_predictions(c1_canary_preds, canary_records, c1_pred_dir / "canary_predictions.jsonl")

                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "payload": payload,
                        "k": k,
                        "canary_PA_C0": round(c0_canary_pa, 4),
                        "canary_PA_C1": round(c1_canary_pa, 4),
                        "PA_drop": round(pa_drop, 4),
                        "canary_WR_C0": round(c0_canary_wr, 4),
                        "canary_WR_C1": round(c1_canary_wr, 4),
                        "template_disagreement": round(td, 4),
                        "passes_td": passes_td,
                        "passes_pa": passes_pa,
                        "passes_criterion": passes,
                    }

                    if args.eval_test and test_records:
                        c1_test_preds = run_deduction(args, test_logs, c1_rules, prompt_template)
                        c1_test_pa = compute_pa(c1_test_preds, test_gts)
                        c1_test_wr = compute_wildcard_ratio(c1_test_preds)
                        test_td = template_disagreement(c0_test_preds, c1_test_preds)
                        test_pa_drop = c0_test_pa - c1_test_pa
                        print(f"  Test PA = {c1_test_pa:.4f} (drop={test_pa_drop:+.4f})")
                        print(f"  Test WR = {c1_test_wr:.4f} (diff={c1_test_wr - c0_test_wr:+.4f})")
                        print(f"  Test Template Disagreement = {test_td:.4f}")
                        save_predictions(c1_test_preds, test_records, c1_pred_dir / "test_predictions.jsonl")
                        row.update({
                            "test_PA_C0": round(c0_test_pa, 4),
                            "test_PA_C1": round(c1_test_pa, 4),
                            "test_PA_drop": round(test_pa_drop, 4),
                            "test_WR_C0": round(c0_test_wr, 4),
                            "test_WR_C1": round(c1_test_wr, 4),
                            "test_template_disagreement": round(test_td, 4),
                        })

                    all_rows.append(row)
                    key = f"{dataset}_seed{seed}_{config_name}"
                    all_results[key] = row

    csv_path = PROJECT_ROOT / "results" / "phase0v2_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPhase-0v2 summary saved to {csv_path}")

    json_path = PROJECT_ROOT / "results" / "phase0v2_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Phase-0v2 results saved to {json_path}")

    passing = [r for r in all_rows if r.get("passes_criterion")]
    print(f"\n{'='*60}")
    if passing:
        print(f"PASS: {len(passing)} config(s) pass criteria:")
        for r in passing:
            print(f"  {r['dataset']}/seed{r['seed']}/{r['payload']}_k{r['k']}: PA_drop={r['PA_drop']:.4f}")
    else:
        print("NO CONFIG PASSES the proceed criterion.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
