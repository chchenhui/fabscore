# Run the self-consistency LLM baseline on a 50-instance stratified FinMR subset.
# Evaluates with 3 seeds, computes ACC/SER/EER/CER per seed and mean +/- std.

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from executable_finmr.baselines.self_consistency_baseline import run_sc_baseline
from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR, PROJECT_ROOT
from executable_finmr.data.load_finmr import load_finmr
from executable_finmr.evaluation.deterministic_judge import evaluate

SEEDS = [42, 123, 456]
SUBSET_IDS_PATH = PROJECT_ROOT / "configs" / "sc_subset_ids.json"
MODEL = "gpt-4.1"
N_SAMPLES = 4


def load_subset(instances, subset_ids):
    id_set = set(subset_ids)
    return [inst for inst in instances if inst.id in id_set]


def main(sanity_n=None):
    print("Loading FinMR dataset...")
    instances = load_finmr()
    print(f"Loaded {len(instances)} instances")

    with open(SUBSET_IDS_PATH) as f:
        subset_ids = json.load(f)
    print(f"Loaded {len(subset_ids)} subset IDs from {SUBSET_IDS_PATH}")

    subset = load_subset(instances, subset_ids)
    print(f"Filtered to {len(subset)} instances")

    if sanity_n is not None:
        subset = subset[:sanity_n]
        print(f"Sanity check mode: using first {sanity_n} instances")

    family_counts = Counter(i.dqc_rule_family for i in subset)
    print(f"DQC family distribution: {dict(sorted(family_counts.items()))}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_metrics = []
    all_per_instance = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"RUNNING SEED={seed}")
        print(f"{'='*60}")

        results = run_sc_baseline(subset, model=MODEL, n_samples=N_SAMPLES, seed=seed)

        eval_input = []
        for r in results:
            pred = r["prediction"]
            if isinstance(pred, str) and pred == "":
                pred = {}
            eval_input.append({
                "id": r["id"],
                "dqc_id": r["dqc_id"],
                "prediction": pred,
                "gold": r["gold"],
            })

        eval_output = evaluate(eval_input)
        metrics = eval_output["metrics"]
        per_instance = eval_output["per_instance"]

        print(f"\n  SEED={seed} METRICS:")
        print(f"    N   = {metrics['N']}")
        print(f"    ACC = {metrics['ACC']:.4f}  ({metrics['N_A']}/{metrics['N']})")
        print(f"    SER = {metrics['SER']:.4f}  ({metrics['N_S']}/{metrics['N']})")
        print(f"    EER = {metrics['EER']:.4f}  ({metrics['N_E']}/{metrics['N']})")
        print(f"    CER = {metrics['CER']:.4f}  ({metrics['N_C']}/{metrics['N']})")

        all_seed_metrics.append(metrics)

        for pi, r in zip(per_instance, results):
            all_per_instance.append({
                "seed": seed,
                "id": pi["id"],
                "dqc_id": pi["dqc_id"],
                "label": pi["label"],
                "prediction": pi["prediction"],
                "gold": pi["gold"],
                "raw_completions": r["raw_completions"],
                "parsed_answers": r["parsed_answers"],
                "majority_answer": r["majority_answer"],
            })

    accs = [m["ACC"] for m in all_seed_metrics]
    sers = [m["SER"] for m in all_seed_metrics]
    eers = [m["EER"] for m in all_seed_metrics]
    cers = [m["CER"] for m in all_seed_metrics]

    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS SEEDS (mean +/- std)")
    print(f"{'='*60}")
    print(f"  ACC = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  SER = {np.mean(sers):.4f} +/- {np.std(sers):.4f}")
    print(f"  EER = {np.mean(eers):.4f} +/- {np.std(eers):.4f}")
    print(f"  CER = {np.mean(cers):.4f} +/- {np.std(cers):.4f}")

    per_dqc_by_seed = {}
    for pi in all_per_instance:
        key = (pi["seed"], pi["dqc_id"])
        if key not in per_dqc_by_seed:
            per_dqc_by_seed[key] = {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0}
        per_dqc_by_seed[key][pi["label"]] += 1
        per_dqc_by_seed[key]["total"] += 1

    output_path = OUTPUT_DIR / "sc_llm_baseline_results.jsonl"
    with open(output_path, "w") as f:
        for r in all_per_instance:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nPer-instance results saved to: {output_path}")

    metrics_output = {
        "method": "self_consistency_llm_baseline",
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "seeds": SEEDS,
        "dataset": "TheFinAI/FinMR",
        "subset_size": len(subset),
        "subset_ids_path": str(SUBSET_IDS_PATH),
        "per_seed": [],
        "aggregate": {
            "ACC_mean": round(float(np.mean(accs)), 4),
            "ACC_std": round(float(np.std(accs)), 4),
            "SER_mean": round(float(np.mean(sers)), 4),
            "SER_std": round(float(np.std(sers)), 4),
            "EER_mean": round(float(np.mean(eers)), 4),
            "EER_std": round(float(np.std(eers)), 4),
            "CER_mean": round(float(np.mean(cers)), 4),
            "CER_std": round(float(np.std(cers)), 4),
        },
    }

    for seed, m in zip(SEEDS, all_seed_metrics):
        per_dqc_seed = {}
        for (s, dqc), counts in per_dqc_by_seed.items():
            if s != seed:
                continue
            n = counts["total"]
            per_dqc_seed[dqc] = {
                "N": n,
                "ACC": round(counts["A"] / n, 4) if n else 0,
                "SER": round(counts["S"] / n, 4) if n else 0,
                "EER": round(counts["E"] / n, 4) if n else 0,
                "CER": round(counts["C"] / n, 4) if n else 0,
            }
        metrics_output["per_seed"].append({
            "seed": seed,
            "ACC": round(m["ACC"], 4),
            "SER": round(m["SER"], 4),
            "EER": round(m["EER"], 4),
            "CER": round(m["CER"], 4),
            "N_A": m["N_A"],
            "N_S": m["N_S"],
            "N_E": m["N_E"],
            "N_C": m["N_C"],
            "per_dqc": per_dqc_seed,
        })

    metrics_path = RESULTS_DIR / "sc_llm_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Aggregate metrics saved to: {metrics_path}")


if __name__ == "__main__":
    sanity_n = None
    if len(sys.argv) > 1 and sys.argv[1] == "--sanity":
        sanity_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(sanity_n=sanity_n)
