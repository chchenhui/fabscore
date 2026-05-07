"""Evaluate C2 canary-based admission control defense.
For each (payload, k, dataset, seed) config:
1. Load R_gen canary PA from C1 canary predictions
2. Load R_safe canary PA from c2_defense/r_safe predictions
3. Run admission control gate (delta=2pp)
4. Select test predictions (R_gen or R_safe) based on decision
5. Compute test metrics and PA recovery
6. Save to results/c2_defense.csv
"""

import csv
import json
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.metrics import compute_pa, compute_fta, compute_wildcard_ratio
from logrules_poisoning.src.defense.admission_control import admission_control

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]
PAYLOADS = ["D", "E", "F"]
K_VALUES = [1, 3, 5, 7]
DELTA = 2.0


def load_predictions(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def get_c0_pa(dataset: str, seed: int, phase0v2_data: dict) -> float:
    key = (dataset, seed)
    return phase0v2_data[key]


def load_phase0v2_c0_pa():
    csv_path = PROJECT_ROOT / "results" / "phase0v2_summary.csv"
    c0_pa = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], int(row["seed"]))
            c0_pa[key] = float(row["test_PA_C0"])
    return c0_pa


def main():
    c0_pa_map = load_phase0v2_c0_pa()

    results_rows = []

    for payload in PAYLOADS:
        for k in K_VALUES:
            for dataset in DATASETS:
                for seed in SEEDS:
                    c1_canary_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}" / "canary_predictions.jsonl"
                    )
                    rsafe_canary_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c2_defense"
                        / "r_safe" / dataset / f"seed_{seed}" / "canary_predictions.jsonl"
                    )
                    c1_test_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}" / "test_predictions.jsonl"
                    )
                    rsafe_test_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c2_defense"
                        / "r_safe" / dataset / f"seed_{seed}" / "test_predictions.jsonl"
                    )

                    c1_canary_recs = load_predictions(c1_canary_path)
                    rsafe_canary_recs = load_predictions(rsafe_canary_path)

                    r_gen_canary_pa = compute_pa(
                        [r["predicted_template"] for r in c1_canary_recs],
                        [r["template"] for r in c1_canary_recs],
                    )
                    r_safe_canary_pa = compute_pa(
                        [r["predicted_template"] for r in rsafe_canary_recs],
                        [r["template"] for r in rsafe_canary_recs],
                    )

                    result = admission_control(r_gen_canary_pa, r_safe_canary_pa, delta=DELTA)
                    decision = result["decision"]

                    if decision == "r_gen":
                        test_recs = load_predictions(c1_test_path)
                    else:
                        test_recs = load_predictions(rsafe_test_path)

                    out_dir = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c2_defense"
                        / payload / dataset / f"k{k}" / f"seed_{seed}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "test_predictions.jsonl", "w") as f:
                        for rec in test_recs:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    test_preds = [r["predicted_template"] for r in test_recs]
                    test_gts = [r["template"] for r in test_recs]
                    test_eids = [r["event_id"] for r in test_recs]

                    test_pa = compute_pa(test_preds, test_gts)
                    _, _, test_fta = compute_fta(test_preds, test_gts, test_eids)
                    test_wr = compute_wildcard_ratio(test_preds)

                    c0_pa = get_c0_pa(dataset, seed, c0_pa_map)

                    c1_test_recs_full = load_predictions(c1_test_path)
                    c1_test_pa = compute_pa(
                        [r["predicted_template"] for r in c1_test_recs_full],
                        [r["template"] for r in c1_test_recs_full],
                    )

                    denom = c0_pa - c1_test_pa
                    if abs(denom) < 0.01:
                        pa_recovery = 0.0
                    else:
                        pa_recovery = (test_pa - c1_test_pa) / denom * 100.0

                    row = {
                        "payload": payload,
                        "k": k,
                        "dataset": dataset,
                        "seed": seed,
                        "test_PA": round(test_pa, 6),
                        "test_FTA": round(test_fta, 6),
                        "test_wildcard_ratio": round(test_wr, 6),
                        "admission_decision": decision,
                        "r_gen_canary_PA": round(r_gen_canary_pa, 4),
                        "r_safe_canary_PA": round(r_safe_canary_pa, 4),
                        "pa_recovery_pct": round(pa_recovery, 2),
                    }
                    results_rows.append(row)

                    print(
                        f"{payload}/k{k}/{dataset}/seed_{seed}: "
                        f"decision={decision} | "
                        f"r_gen_canary={r_gen_canary_pa:.4f} r_safe_canary={r_safe_canary_pa:.4f} | "
                        f"C2_PA={test_pa:.4f} C1_PA={c1_test_pa:.4f} C0_PA={c0_pa:.4f} | "
                        f"recovery={pa_recovery:.1f}%"
                    )

    csv_path = PROJECT_ROOT / "results" / "c2_defense.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nResults saved to {csv_path}")

    print(f"\n\n{'='*60}")
    print("=== C2 DEFENSE SUMMARY ===")
    print(f"{'='*60}")

    total_configs = len(results_rows)
    rsafe_count = sum(1 for r in results_rows if r["admission_decision"] == "r_safe")
    rgen_count = total_configs - rsafe_count
    print(f"\nAdmission decisions: R_safe={rsafe_count}/{total_configs} ({rsafe_count/total_configs*100:.1f}%), "
          f"R_gen={rgen_count}/{total_configs} ({rgen_count/total_configs*100:.1f}%)")

    print("\n--- Mean +/- Std across 3 seeds ---")
    by_config = defaultdict(list)
    for row in results_rows:
        key = (row["payload"], row["k"], row["dataset"])
        by_config[key].append(row)

    for (payload, k, dataset), rows in sorted(by_config.items()):
        pa_vals = [r["test_PA"] for r in rows]
        recovery_vals = [r["pa_recovery_pct"] for r in rows]
        decisions = [r["admission_decision"] for r in rows]
        rsafe_frac = sum(1 for d in decisions if d == "r_safe") / len(decisions)

        pa_mean = statistics.mean(pa_vals)
        pa_std = statistics.stdev(pa_vals) if len(pa_vals) > 1 else 0.0
        rec_mean = statistics.mean(recovery_vals)
        rec_std = statistics.stdev(recovery_vals) if len(recovery_vals) > 1 else 0.0

        print(
            f"  {payload}/k{k}/{dataset}: "
            f"PA={pa_mean:.4f}+/-{pa_std:.4f} | "
            f"recovery={rec_mean:.1f}+/-{rec_std:.1f}% | "
            f"R_safe rate={rsafe_frac:.0%}"
        )


if __name__ == "__main__":
    main()
