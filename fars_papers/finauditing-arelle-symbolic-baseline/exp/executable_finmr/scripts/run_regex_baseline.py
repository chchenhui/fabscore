# Run the regex message-only baseline on all 332 FinMR instances,
# evaluate with the deterministic judge, and save results.

import json
from collections import Counter
from pathlib import Path

from executable_finmr.baselines.regex_baseline import (
    run_regex_baseline,
    run_regex_baseline_single,
)
from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR
from executable_finmr.data.load_finmr import load_finmr
from executable_finmr.evaluation.deterministic_judge import evaluate


def main():
    print("Loading FinMR dataset...")
    instances = load_finmr()
    print(f"Loaded {len(instances)} instances")

    dqc_counts = Counter(i.dqc_id for i in instances)
    print(f"DQC distribution: {dict(sorted(dqc_counts.items()))}")

    print("\nRunning regex baseline on all instances...")
    results = run_regex_baseline(instances)

    print("Evaluating with deterministic judge...")
    eval_output = evaluate(results)
    metrics = eval_output["metrics"]
    per_instance = eval_output["per_instance"]

    print(f"\n{'='*50}")
    print("AGGREGATE METRICS")
    print(f"{'='*50}")
    print(f"  N       = {metrics['N']}")
    print(f"  ACC     = {metrics['ACC']:.4f}  ({metrics['N_A']}/{metrics['N']})")
    print(f"  SER     = {metrics['SER']:.4f}  ({metrics['N_S']}/{metrics['N']})")
    print(f"  EER     = {metrics['EER']:.4f}  ({metrics['N_E']}/{metrics['N']})")
    print(f"  CER     = {metrics['CER']:.4f}  ({metrics['N_C']}/{metrics['N']})")

    per_dqc = {}
    for r in per_instance:
        dqc = r["dqc_id"]
        if dqc not in per_dqc:
            per_dqc[dqc] = {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0}
        per_dqc[dqc][r["label"]] += 1
        per_dqc[dqc]["total"] += 1

    print(f"\n{'='*50}")
    print("PER-DQC BREAKDOWN")
    print(f"{'='*50}")
    for dqc in sorted(per_dqc):
        d = per_dqc[dqc]
        n = d["total"]
        acc = d["A"] / n if n else 0
        ser = d["S"] / n if n else 0
        eer = d["E"] / n if n else 0
        cer = d["C"] / n if n else 0
        print(f"  {dqc}: N={n} ACC={acc:.4f} SER={ser:.4f} EER={eer:.4f} CER={cer:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "regex_baseline_results.jsonl"
    with open(output_path, "w") as f:
        for r in per_instance:
            row = {
                "id": r["id"],
                "dqc_id": r["dqc_id"],
                "prediction": r["prediction"],
                "gold": r["gold"],
                "label": r["label"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"\nPer-instance results saved to: {output_path}")

    metrics_output = {
        "method": "regex_message_only_baseline",
        "dataset": "TheFinAI/FinMR",
        "n_instances": metrics["N"],
        "aggregate": {
            "ACC": round(metrics["ACC"], 4),
            "SER": round(metrics["SER"], 4),
            "EER": round(metrics["EER"], 4),
            "CER": round(metrics["CER"], 4),
        },
        "counts": {
            "N_A": metrics["N_A"],
            "N_S": metrics["N_S"],
            "N_E": metrics["N_E"],
            "N_C": metrics["N_C"],
        },
        "per_dqc": {},
    }
    for dqc in sorted(per_dqc):
        d = per_dqc[dqc]
        n = d["total"]
        metrics_output["per_dqc"][dqc] = {
            "N": n,
            "ACC": round(d["A"] / n, 4) if n else 0,
            "SER": round(d["S"] / n, 4) if n else 0,
            "EER": round(d["E"] / n, 4) if n else 0,
            "CER": round(d["C"] / n, 4) if n else 0,
            "N_A": d["A"],
            "N_S": d["S"],
            "N_E": d["E"],
            "N_C": d["C"],
        }

    metrics_path = RESULTS_DIR / "regex_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Aggregate metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
