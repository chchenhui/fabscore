"""Cross-model deduction: run C0/C1/C2 on BGL with an alternative deduction model.
Reuses induced rules from main experiments; only the deduction model changes.
Conditions: C0 (clean), C1 (poisoned D k=1,3), R_safe (for C2 admission control).
Usage: python scripts/run_cross_model_deduction.py --api-base URL --model MODEL
"""

import argparse
import csv
import json
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
from logrules_poisoning.src.defense.admission_control import R_SAFE, format_rules, admission_control

SEEDS = [42, 123, 456]
DATASET = "BGL"
PAYLOAD = "D"
K_VALUES = [1, 3]
DELTA = 2.0


def load_split(dataset, seed, split_name):
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / f"{split_name}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_ranked_rules(rules_path):
    with open(rules_path) as f:
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


def run_parsing(api_base, api_key, model, prompt_template, logs, rules_text, max_concurrency):
    return parse_logs(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt_template=prompt_template,
        logs=logs,
        rules=rules_text,
        max_concurrency=max_concurrency,
    )


def sanity_check(api_base, api_key, model, prompt_template, max_concurrency):
    print("\n" + "=" * 60)
    print("SANITY CHECK: Parsing 5 BGL test logs with C0 rules (seed=42)")
    print("=" * 60)

    rules_path = PROJECT_ROOT / "outputs" / "rules" / "phase0v2_clean" / DATASET / "seed_42" / "ranked_rules.json"
    rules = load_ranked_rules(rules_path)
    rules_text = format_rules_numbered(rules)

    test_records = load_split(DATASET, 42, "test")[:5]
    test_logs = [r["raw_log"] for r in test_records]
    test_gts = [r["template"] for r in test_records]

    preds = run_parsing(api_base, api_key, model, prompt_template, test_logs, rules_text, max_concurrency)

    for i, (log, gt, pred) in enumerate(zip(test_logs, test_gts, preds)):
        print(f"\n--- Log {i+1} ---")
        print(f"  Raw:  {log[:120]}...")
        print(f"  GT:   {gt}")
        print(f"  Pred: {pred}")

    pa = compute_pa(preds, test_gts)
    print(f"\nSanity check PA (5 logs): {pa:.4f}")

    if all(p == "" for p in preds):
        print("ERROR: All predictions empty. Chat template may not work. Aborting.")
        sys.exit(1)

    print("Sanity check passed. Proceeding with full experiment.\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    prompt_path = PROJECT_ROOT / "prompts" / "deduction_with_rules.txt"
    prompt_template = prompt_path.read_text().strip()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    if "llama" in model_short:
        model_tag = "llama"
    elif "qwen" in model_short:
        model_tag = "qwen"
    else:
        model_tag = model_short

    wait_for_server(args.api_base)

    if not args.skip_sanity:
        sanity_check(args.api_base, args.api_key, args.model, prompt_template, args.max_concurrency)

    all_results = []

    for seed in SEEDS:
        test_records = load_split(DATASET, seed, "test")
        canary_records = load_split(DATASET, seed, "canary")

        test_logs = [r["raw_log"] for r in test_records]
        test_gts = [r["template"] for r in test_records]
        test_eids = [r["event_id"] for r in test_records]
        canary_logs = [r["raw_log"] for r in canary_records]
        canary_gts = [r["template"] for r in canary_records]

        print(f"\n{'='*60}")
        print(f"C0 CLEAN: {DATASET} / seed_{seed}")
        print(f"{'='*60}")

        c0_rules_path = PROJECT_ROOT / "outputs" / "rules" / "phase0v2_clean" / DATASET / f"seed_{seed}" / "ranked_rules.json"
        c0_rules = load_ranked_rules(c0_rules_path)
        c0_rules_text = format_rules_numbered(c0_rules)
        print(f"Loaded {len(c0_rules)} clean rules")

        t0 = time.time()
        c0_test_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, test_logs, c0_rules_text, args.max_concurrency)
        print(f"  Test: {len(test_logs)} logs in {time.time()-t0:.1f}s")

        t0 = time.time()
        c0_canary_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, canary_logs, c0_rules_text, args.max_concurrency)
        print(f"  Canary: {len(canary_logs)} logs in {time.time()-t0:.1f}s")

        out_dir = PROJECT_ROOT / "outputs" / "predictions" / f"c0_{model_tag}" / DATASET / f"seed_{seed}"
        save_predictions(c0_test_preds, test_records, out_dir / "predictions.jsonl")
        save_predictions(c0_canary_preds, canary_records, out_dir / "canary_predictions.jsonl")

        c0_metrics = evaluate_all(c0_test_preds, test_gts, test_eids)
        c0_canary_pa = compute_pa(c0_canary_preds, canary_gts)
        print(f"  Test PA={c0_metrics['PA']:.4f}  FTA={c0_metrics['FTA']:.4f}  WR={c0_metrics['wildcard_ratio']:.4f}")
        print(f"  Canary PA={c0_canary_pa:.4f}")

        all_results.append({
            "model": args.model, "condition": "C0", "k": "-", "seed": seed,
            "PA": c0_metrics["PA"], "FTA": c0_metrics["FTA"],
            "wildcard_ratio": c0_metrics["wildcard_ratio"],
            "canary_PA": c0_canary_pa,
        })

        for k in K_VALUES:
            print(f"\n{'='*60}")
            print(f"C1 POISONED: {DATASET} / seed_{seed} / payload={PAYLOAD} k={k}")
            print(f"{'='*60}")

            c1_rules_path = PROJECT_ROOT / "outputs" / "rules" / "c1_poisoned" / PAYLOAD / DATASET / f"k{k}" / f"seed_{seed}" / "ranked_rules.json"
            c1_rules = load_ranked_rules(c1_rules_path)
            c1_rules_text = format_rules_numbered(c1_rules)
            print(f"Loaded {len(c1_rules)} poisoned rules")

            t0 = time.time()
            c1_test_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, test_logs, c1_rules_text, args.max_concurrency)
            print(f"  Test: {len(test_logs)} logs in {time.time()-t0:.1f}s")

            t0 = time.time()
            c1_canary_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, canary_logs, c1_rules_text, args.max_concurrency)
            print(f"  Canary: {len(canary_logs)} logs in {time.time()-t0:.1f}s")

            out_dir = PROJECT_ROOT / "outputs" / "predictions" / f"c1_{model_tag}" / PAYLOAD / DATASET / f"k{k}" / f"seed_{seed}"
            save_predictions(c1_test_preds, test_records, out_dir / "predictions.jsonl")
            save_predictions(c1_canary_preds, canary_records, out_dir / "canary_predictions.jsonl")

            c1_metrics = evaluate_all(c1_test_preds, test_gts, test_eids)
            c1_canary_pa = compute_pa(c1_canary_preds, canary_gts)
            print(f"  Test PA={c1_metrics['PA']:.4f}  FTA={c1_metrics['FTA']:.4f}  WR={c1_metrics['wildcard_ratio']:.4f}")
            print(f"  Canary PA={c1_canary_pa:.4f}")

            all_results.append({
                "model": args.model, "condition": "C1", "k": k, "seed": seed,
                "PA": c1_metrics["PA"], "FTA": c1_metrics["FTA"],
                "wildcard_ratio": c1_metrics["wildcard_ratio"],
                "canary_PA": c1_canary_pa,
            })

        print(f"\n{'='*60}")
        print(f"R_SAFE: {DATASET} / seed_{seed}")
        print(f"{'='*60}")

        rsafe_rules_text = format_rules(R_SAFE)
        print(f"Using {len(R_SAFE)} R_safe rules")

        t0 = time.time()
        rsafe_test_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, test_logs, rsafe_rules_text, args.max_concurrency)
        print(f"  Test: {len(test_logs)} logs in {time.time()-t0:.1f}s")

        t0 = time.time()
        rsafe_canary_preds = run_parsing(args.api_base, args.api_key, args.model, prompt_template, canary_logs, rsafe_rules_text, args.max_concurrency)
        print(f"  Canary: {len(canary_logs)} logs in {time.time()-t0:.1f}s")

        out_dir = PROJECT_ROOT / "outputs" / "predictions" / f"c2_{model_tag}" / "r_safe" / DATASET / f"seed_{seed}"
        save_predictions(rsafe_test_preds, test_records, out_dir / "test_predictions.jsonl")
        save_predictions(rsafe_canary_preds, canary_records, out_dir / "canary_predictions.jsonl")

        rsafe_metrics = evaluate_all(rsafe_test_preds, test_gts, test_eids)
        rsafe_canary_pa = compute_pa(rsafe_canary_preds, canary_gts)
        print(f"  Test PA={rsafe_metrics['PA']:.4f}  FTA={rsafe_metrics['FTA']:.4f}  WR={rsafe_metrics['wildcard_ratio']:.4f}")
        print(f"  Canary PA={rsafe_canary_pa:.4f}")

        all_results.append({
            "model": args.model, "condition": "R_safe", "k": "-", "seed": seed,
            "PA": rsafe_metrics["PA"], "FTA": rsafe_metrics["FTA"],
            "wildcard_ratio": rsafe_metrics["wildcard_ratio"],
            "canary_PA": rsafe_canary_pa,
        })

    print("\n\n" + "=" * 80)
    print("C2 ADMISSION CONTROL")
    print("=" * 80)

    c2_results = []
    for seed in SEEDS:
        rsafe_canary_path = PROJECT_ROOT / "outputs" / "predictions" / f"c2_{model_tag}" / "r_safe" / DATASET / f"seed_{seed}" / "canary_predictions.jsonl"
        rsafe_canary_recs = []
        with open(rsafe_canary_path) as f:
            for line in f:
                rsafe_canary_recs.append(json.loads(line.strip()))
        rsafe_canary_preds_list = [r["predicted_template"] for r in rsafe_canary_recs]
        canary_records = load_split(DATASET, seed, "canary")
        canary_gts = [r["template"] for r in canary_records]
        rsafe_canary_pa = compute_pa(rsafe_canary_preds_list, canary_gts)

        for k in K_VALUES:
            c1_canary_path = PROJECT_ROOT / "outputs" / "predictions" / f"c1_{model_tag}" / PAYLOAD / DATASET / f"k{k}" / f"seed_{seed}" / "canary_predictions.jsonl"
            c1_canary_recs = []
            with open(c1_canary_path) as f:
                for line in f:
                    c1_canary_recs.append(json.loads(line.strip()))
            c1_canary_preds_list = [r["predicted_template"] for r in c1_canary_recs]
            c1_canary_pa = compute_pa(c1_canary_preds_list, canary_gts)

            ac = admission_control(c1_canary_pa, rsafe_canary_pa, DELTA)
            decision = ac["decision"]

            if decision == "r_gen":
                c1_test_path = PROJECT_ROOT / "outputs" / "predictions" / f"c1_{model_tag}" / PAYLOAD / DATASET / f"k{k}" / f"seed_{seed}" / "predictions.jsonl"
            else:
                c1_test_path = PROJECT_ROOT / "outputs" / "predictions" / f"c2_{model_tag}" / "r_safe" / DATASET / f"seed_{seed}" / "test_predictions.jsonl"

            test_recs = []
            with open(c1_test_path) as f:
                for line in f:
                    test_recs.append(json.loads(line.strip()))
            test_preds_list = [r["predicted_template"] for r in test_recs]
            test_records = load_split(DATASET, seed, "test")
            test_gts_list = [r["template"] for r in test_records]
            test_eids = [r["event_id"] for r in test_records]

            c2_metrics = evaluate_all(test_preds_list, test_gts_list, test_eids)

            c0_pa = [r["PA"] for r in all_results if r["condition"] == "C0" and r["seed"] == seed][0]
            c1_pa = [r["PA"] for r in all_results if r["condition"] == "C1" and r["k"] == k and r["seed"] == seed][0]
            pa_drop = c0_pa - c1_pa
            pa_recovery = ((c2_metrics["PA"] - c1_pa) / pa_drop * 100) if pa_drop > 0.001 else 0.0

            print(f"  seed={seed} k={k}: decision={decision}, "
                  f"C1_canary_PA={c1_canary_pa:.4f}, Rsafe_canary_PA={rsafe_canary_pa:.4f}, "
                  f"C2_test_PA={c2_metrics['PA']:.4f}, recovery={pa_recovery:.1f}%")

            all_results.append({
                "model": args.model, "condition": "C2", "k": k, "seed": seed,
                "PA": c2_metrics["PA"], "FTA": c2_metrics["FTA"],
                "wildcard_ratio": c2_metrics["wildcard_ratio"],
                "canary_PA": c1_canary_pa,
                "admission_decision": decision,
                "r_gen_canary_PA": c1_canary_pa,
                "r_safe_canary_PA": rsafe_canary_pa,
                "pa_recovery_pct": pa_recovery,
            })

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"cross_model_{model_tag}.csv"
    fieldnames = ["model", "condition", "k", "seed", "PA", "FTA", "wildcard_ratio",
                  "canary_PA", "admission_decision", "r_gen_canary_PA", "r_safe_canary_PA", "pa_recovery_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<12} {'k':<4} {'Seed':<6} {'PA':<8} {'FTA':<8} {'WR':<8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['condition']:<12} {str(r['k']):<4} {r['seed']:<6} {r['PA']:.4f}  {r['FTA']:.4f}  {r['wildcard_ratio']:.4f}")


if __name__ == "__main__":
    main()
