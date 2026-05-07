"""Delta ablation: compare admission control with delta=0 vs delta=2.
delta=0 uses >= semantics (R_gen selected if equal or better on canary PA).
delta=2 uses existing strict > semantics from c2_defense.csv.
Restricted to k in {1,3} and payloads D/E/F.
"""

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.metrics import compute_pa, compute_fta, compute_wildcard_ratio

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]
PAYLOADS = ["D", "E", "F"]
K_VALUES = [1, 3]
DELTAS = [0, 2]


def load_predictions(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_c2_defense_csv():
    csv_path = PROJECT_ROOT / "results" / "c2_defense.csv"
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_phase0v2_c0_pa():
    csv_path = PROJECT_ROOT / "results" / "phase0v2_summary.csv"
    c0_pa = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], int(row["seed"]))
            c0_pa[key] = float(row["test_PA_C0"])
    return c0_pa


def get_c1_test_pa(payload, k, dataset, seed):
    c1_test_path = (
        PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned"
        / payload / dataset / f"k{k}" / f"seed_{seed}" / "test_predictions.jsonl"
    )
    recs = load_predictions(c1_test_path)
    return compute_pa(
        [r["predicted_template"] for r in recs],
        [r["template"] for r in recs],
    )


def compute_test_metrics(test_recs):
    preds = [r["predicted_template"] for r in test_recs]
    gts = [r["template"] for r in test_recs]
    eids = [r["event_id"] for r in test_recs]
    pa = compute_pa(preds, gts)
    _, _, fta = compute_fta(preds, gts, eids)
    wr = compute_wildcard_ratio(preds)
    return pa, fta, wr


def main():
    c2_rows = load_c2_defense_csv()
    c0_pa_map = load_phase0v2_c0_pa()

    c2_lookup = {}
    for row in c2_rows:
        key = (row["payload"], int(row["k"]), row["dataset"], int(row["seed"]))
        c2_lookup[key] = row

    results = []

    for payload in PAYLOADS:
        for k in K_VALUES:
            for dataset in DATASETS:
                for seed in SEEDS:
                    key = (payload, k, dataset, seed)
                    c2_row = c2_lookup[key]

                    r_gen_canary_pa = float(c2_row["r_gen_canary_PA"])
                    r_safe_canary_pa = float(c2_row["r_safe_canary_PA"])

                    c0_pa = c0_pa_map[(dataset, seed)]
                    c1_test_pa = get_c1_test_pa(payload, k, dataset, seed)

                    c1_test_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}" / "test_predictions.jsonl"
                    )
                    rsafe_test_path = (
                        PROJECT_ROOT / "outputs" / "predictions" / "c2_defense"
                        / "r_safe" / dataset / f"seed_{seed}" / "test_predictions.jsonl"
                    )

                    for delta in DELTAS:
                        if delta == 0:
                            decision = "r_gen" if r_gen_canary_pa >= r_safe_canary_pa else "r_safe"
                        else:
                            decision = c2_row["admission_decision"]

                        if decision == "r_gen":
                            test_recs = load_predictions(c1_test_path)
                        else:
                            test_recs = load_predictions(rsafe_test_path)

                        test_pa, test_fta, test_wr = compute_test_metrics(test_recs)

                        denom = c0_pa - c1_test_pa
                        if abs(denom) < 0.01:
                            pa_recovery = 0.0
                        else:
                            pa_recovery = (test_pa - c1_test_pa) / denom * 100.0

                        results.append({
                            "payload": payload,
                            "k": k,
                            "dataset": dataset,
                            "seed": seed,
                            "delta": delta,
                            "admission_decision": decision,
                            "test_PA": round(test_pa, 6),
                            "test_FTA": round(test_fta, 6),
                            "test_wildcard_ratio": round(test_wr, 6),
                            "pa_recovery_pct": round(pa_recovery, 2),
                        })

    csv_path = PROJECT_ROOT / "results" / "delta_ablation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {csv_path}")

    # --- Verification: spot-check decision flips ---
    print(f"\n{'='*70}")
    print("VERIFICATION: Decision Flips (delta=0 vs delta=2)")
    print(f"{'='*70}")

    flip_configs = []
    d0_lookup = {}
    d2_lookup = {}
    for row in results:
        key = (row["payload"], row["k"], row["dataset"], row["seed"])
        if row["delta"] == 0:
            d0_lookup[key] = row
        else:
            d2_lookup[key] = row

    for key in d0_lookup:
        d0 = d0_lookup[key]
        d2 = d2_lookup[key]
        if d0["admission_decision"] != d2["admission_decision"]:
            c2_row = c2_lookup[key]
            flip_configs.append({
                "config": f"{key[0]}/k{key[1]}/{key[2]}/seed_{key[3]}",
                "r_gen_canary_PA": float(c2_row["r_gen_canary_PA"]),
                "r_safe_canary_PA": float(c2_row["r_safe_canary_PA"]),
                "delta0_decision": d0["admission_decision"],
                "delta2_decision": d2["admission_decision"],
                "delta0_test_PA": d0["test_PA"],
                "delta2_test_PA": d2["test_PA"],
                "pa_change": round(d0["test_PA"] - d2["test_PA"], 6),
            })

    if flip_configs:
        print(f"\n{len(flip_configs)} configs flipped decision:")
        for fc in flip_configs:
            print(f"  {fc['config']}: "
                  f"r_gen_canary={fc['r_gen_canary_PA']:.4f} r_safe_canary={fc['r_safe_canary_PA']:.4f} | "
                  f"delta=0->{fc['delta0_decision']} delta=2->{fc['delta2_decision']} | "
                  f"PA: {fc['delta0_test_PA']:.4f} vs {fc['delta2_test_PA']:.4f} (change={fc['pa_change']:+.4f})")

        wrong_direction = [fc for fc in flip_configs if fc["delta0_decision"] == "r_safe" and fc["delta2_decision"] == "r_gen"]
        if wrong_direction:
            print(f"\n  WARNING: {len(wrong_direction)} configs flipped r_gen->r_safe (unexpected!)")
        else:
            print(f"\n  All flips are r_safe->r_gen (expected with lower delta)")
    else:
        print("\n  No decision flips between delta=0 and delta=2")

    boundary_configs = []
    for key in d0_lookup:
        c2_row = c2_lookup[key]
        rg = float(c2_row["r_gen_canary_PA"])
        rs = float(c2_row["r_safe_canary_PA"])
        if rs <= rg <= rs + 0.02:
            boundary_configs.append({
                "config": f"{key[0]}/k{key[1]}/{key[2]}/seed_{key[3]}",
                "r_gen_canary_PA": rg,
                "r_safe_canary_PA": rs,
                "gap": rg - rs,
            })
    if boundary_configs:
        print(f"\n  Boundary configs (r_gen_canary in [r_safe_canary, r_safe_canary+0.02]):")
        for bc in boundary_configs:
            print(f"    {bc['config']}: r_gen={bc['r_gen_canary_PA']:.4f} r_safe={bc['r_safe_canary_PA']:.4f} gap={bc['gap']:.4f}")

    # --- Summary Statistics ---
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    for delta in DELTAS:
        delta_rows = [r for r in results if r["delta"] == delta]
        total = len(delta_rows)
        rgen_count = sum(1 for r in delta_rows if r["admission_decision"] == "r_gen")
        rsafe_count = total - rgen_count
        mean_recovery = statistics.mean([r["pa_recovery_pct"] for r in delta_rows])
        mean_pa = statistics.mean([r["test_PA"] for r in delta_rows])

        print(f"\n  delta={delta}:")
        print(f"    R_gen deployed (false acceptance): {rgen_count}/{total} ({rgen_count/total*100:.1f}%)")
        print(f"    R_safe fallback: {rsafe_count}/{total} ({rsafe_count/total*100:.1f}%)")
        print(f"    Mean test PA: {mean_pa:.4f}")
        print(f"    Mean PA recovery: {mean_recovery:.1f}%")

        for ds in DATASETS:
            ds_rows = [r for r in delta_rows if r["dataset"] == ds]
            ds_rgen = sum(1 for r in ds_rows if r["admission_decision"] == "r_gen")
            ds_recovery = statistics.mean([r["pa_recovery_pct"] for r in ds_rows])
            ds_pa = statistics.mean([r["test_PA"] for r in ds_rows])
            print(f"    {ds}: R_gen={ds_rgen}/{len(ds_rows)} PA={ds_pa:.4f} recovery={ds_recovery:.1f}%")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        d0_recoveries = []
        d2_recoveries = []
        labels = []

        for payload in PAYLOADS:
            for k in K_VALUES:
                label = f"{payload}_k{k}"
                labels.append(label)
                d0_vals = [r["pa_recovery_pct"] for r in results
                           if r["dataset"] == ds and r["payload"] == payload
                           and r["k"] == k and r["delta"] == 0]
                d2_vals = [r["pa_recovery_pct"] for r in results
                           if r["dataset"] == ds and r["payload"] == payload
                           and r["k"] == k and r["delta"] == 2]
                d0_recoveries.append(statistics.mean(d0_vals))
                d2_recoveries.append(statistics.mean(d2_vals))

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, d0_recoveries, width, label="delta=0", color="#1f77b4", alpha=0.8)
        ax.bar(x + width/2, d2_recoveries, width, label="delta=2", color="#ff7f0e", alpha=0.8)
        ax.set_title(ds, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        if idx == 0:
            ax.set_ylabel("PA Recovery (%)", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Delta Ablation: PA Recovery by delta across Datasets (k=1,3)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "delta_ablation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to {fig_path}")

    # --- Build structured JSON results ---
    summary = {
        "task": "delta_ablation",
        "description": "Ablation of admission control margin delta (0 vs 2) at k in {1,3}",
        "k_values": K_VALUES,
        "payloads": PAYLOADS,
        "datasets": DATASETS,
        "total_configs_per_delta": len(results) // 2,
    }

    for delta in DELTAS:
        delta_rows = [r for r in results if r["delta"] == delta]
        rgen_count = sum(1 for r in delta_rows if r["admission_decision"] == "r_gen")
        total = len(delta_rows)
        summary[f"delta_{delta}"] = {
            "false_acceptance_rate": round(rgen_count / total, 4),
            "r_gen_count": rgen_count,
            "r_safe_count": total - rgen_count,
            "mean_test_PA": round(statistics.mean([r["test_PA"] for r in delta_rows]), 4),
            "mean_pa_recovery_pct": round(statistics.mean([r["pa_recovery_pct"] for r in delta_rows]), 2),
            "per_dataset": {},
        }
        for ds in DATASETS:
            ds_rows = [r for r in delta_rows if r["dataset"] == ds]
            ds_rgen = sum(1 for r in ds_rows if r["admission_decision"] == "r_gen")
            summary[f"delta_{delta}"]["per_dataset"][ds] = {
                "false_acceptance_rate": round(ds_rgen / len(ds_rows), 4),
                "r_gen_count": ds_rgen,
                "mean_test_PA": round(statistics.mean([r["test_PA"] for r in ds_rows]), 4),
                "mean_pa_recovery_pct": round(statistics.mean([r["pa_recovery_pct"] for r in ds_rows]), 2),
            }

    summary["decision_flips"] = {
        "count": len(flip_configs),
        "configs": flip_configs,
    }

    json_path = PROJECT_ROOT / "results" / "delta_ablation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Structured results saved to {json_path}")

    return summary, flip_configs, results


if __name__ == "__main__":
    main()
