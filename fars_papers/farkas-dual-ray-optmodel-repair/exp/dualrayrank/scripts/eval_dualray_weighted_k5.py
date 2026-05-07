"""Evaluate already-generated DualRay-TopK+Weights K=5 repair outputs.
Runs locally without GPU -- just solves the LP files with HiGHS.
"""

import json
from pathlib import Path

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.solver.highs_wrapper import HiGHSWrapper


def main():
    attempt0_dir = Path("dualrayrank/outputs/attempt0")
    output_dir = Path("dualrayrank/outputs/repair_dualray_weighted_k5")
    results_dir = Path("dualrayrank/results")
    iis_results_path = Path("dualrayrank/results/iis_topk_results.json")
    K = 5

    with open(attempt0_dir / "infeasible_instances.json", "r") as f:
        infeasible_list = json.load(f)
    print(f"Loaded {len(infeasible_list)} infeasible instances")

    iis_size_map = {}
    if iis_results_path.exists():
        with open(iis_results_path, "r") as f:
            iis_data = json.load(f)
        for item in iis_data.get("per_instance", []):
            iis_size_map[f"{item['difficulty']}_{item['instance_id']}"] = item.get("iis_size")

    all_instances = load_mamo()
    instance_map = {}
    for inst in all_instances:
        instance_map[f"{inst['difficulty']}_{inst['id']}"] = inst

    eval_wrapper = HiGHSWrapper()
    repair_successes = 0
    per_instance = []

    for entry in infeasible_list:
        iid = entry["instance_id"]
        difficulty = entry["difficulty"]
        key = f"{difficulty}_{iid}"
        inst = instance_map.get(key)

        repaired_path = output_dir / f"{key}.lp"
        if not repaired_path.exists():
            per_instance.append({
                "instance_id": iid,
                "difficulty": difficulty,
                "attempt0_status": "fail-infeasible",
                "repair_status": "skipped",
                "repair_objective": None,
                "ground_truth": None,
                "dual_ray_valid": None,
                "num_nonzero_multipliers": None,
                "feedback_source": None,
                "iis_size": iis_size_map.get(key),
                "skip_reason": "no_repaired_file",
            })
            continue

        gt = parse_ground_truth(inst["Answer"]) if inst else None
        record = classify_result(repaired_path, gt, eval_wrapper)
        repair_status = record["classification"]
        if repair_status == "pass":
            repair_successes += 1

        per_instance.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "attempt0_status": "fail-infeasible",
            "repair_status": repair_status,
            "repair_objective": record.get("objective"),
            "ground_truth": gt,
            "dual_ray_valid": None,
            "num_nonzero_multipliers": None,
            "feedback_source": "dualray_weighted",
            "iis_size": iis_size_map.get(key),
        })

    infeasible_total = len(infeasible_list)
    repaired_total = sum(1 for r in per_instance if r["repair_status"] != "skipped")

    easy_repair = [r for r in per_instance if r["difficulty"] == "EasyLP" and r["repair_status"] == "pass"]
    complex_repair = [r for r in per_instance if r["difficulty"] == "ComplexLP" and r["repair_status"] == "pass"]

    easy_pass_attempt0 = 464
    complex_pass_attempt0 = 37
    easy_total = 652
    complex_total = 211
    attempt0_pass_count = 501
    total_instances = 863

    truncation_subset = [r for r in per_instance if r.get("iis_size") is not None and r["iis_size"] > K]
    truncation_successes = sum(1 for r in truncation_subset if r["repair_status"] == "pass")
    truncation_total = len(truncation_subset)

    results_summary = {
        "experiment": "dualray_weighted_k5_repair",
        "condition": "C (DualRay-TopK+Weights, K=5, greedy)",
        "k": K,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "infeasible_instances": infeasible_total,
        "repairable_instances": repaired_total,
        "skipped_instances": infeasible_total - repaired_total,
        "repair_successes": repair_successes,
        "repair_success_rate": repair_successes / infeasible_total if infeasible_total > 0 else 0,
        "repair_success_rate_on_repairable": repair_successes / repaired_total if repaired_total > 0 else 0,
        "pass_at_1_after_repair": {
            "overall": (attempt0_pass_count + repair_successes) / total_instances,
            "overall_count": attempt0_pass_count + repair_successes,
            "overall_total": total_instances,
            "EasyLP": (easy_pass_attempt0 + len(easy_repair)) / easy_total,
            "EasyLP_count": easy_pass_attempt0 + len(easy_repair),
            "EasyLP_total": easy_total,
            "ComplexLP": (complex_pass_attempt0 + len(complex_repair)) / complex_total,
            "ComplexLP_count": complex_pass_attempt0 + len(complex_repair),
            "ComplexLP_total": complex_total,
        },
        "truncation_regime": {
            "description": "Instances where IIS size > K (truncation regime)",
            "total_instances": truncation_total,
            "repair_successes": truncation_successes,
            "repair_success_rate": truncation_successes / truncation_total if truncation_total > 0 else 0,
        },
        "per_instance": per_instance,
    }

    results_path = results_dir / "dualray_weighted_k5_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")
    print(f"\n{'='*60}")
    print(f"DualRay-TopK+Weights REPAIR RESULTS (Condition C, K={K}, greedy)")
    print(f"{'='*60}")
    print(f"Infeasible instances: {infeasible_total}")
    print(f"Repairable: {repaired_total}")
    print(f"Repair successes: {repair_successes}")
    print(f"Repair success rate (on infeasible): {repair_successes}/{infeasible_total} = {results_summary['repair_success_rate']:.4f}")
    p = results_summary["pass_at_1_after_repair"]
    print(f"\nPass@1 after <=2 attempts:")
    print(f"  Overall:   {p['overall']:.4f} ({p['overall_count']}/{p['overall_total']})")
    print(f"  EasyLP:    {p['EasyLP']:.4f} ({p['EasyLP_count']}/{p['EasyLP_total']})")
    print(f"  ComplexLP: {p['ComplexLP']:.4f} ({p['ComplexLP_count']}/{p['ComplexLP_total']})")
    t = results_summary["truncation_regime"]
    print(f"\nTruncation regime (IIS size > {K}):")
    print(f"  Instances: {t['total_instances']}")
    print(f"  Repair successes: {t['repair_successes']}/{t['total_instances']} = {t['repair_success_rate']:.4f}")

    for r in per_instance:
        if r["repair_status"] == "pass":
            print(f"\nRepaired: {r['difficulty']}_{r['instance_id']} (obj={r['repair_objective']}, gt={r['ground_truth']})")


if __name__ == "__main__":
    main()
