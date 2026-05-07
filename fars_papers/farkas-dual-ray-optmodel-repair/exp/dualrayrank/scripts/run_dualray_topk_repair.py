"""DualRay-TopK repair experiment (Condition B): repair infeasible attempt-0 instances
using Farkas dual ray ranking with K=5 constraints (names only, no weights).
Falls back to IIS-TopK feedback if dual ray extraction fails.

Usage:
    python -m dualrayrank.scripts.run_dualray_topk_repair [--output-dir DIR] [--model MODEL]
"""

import argparse
import json
import sys
from pathlib import Path

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.inference.vllm_runner import (
    DEFAULT_SAMPLING_PARAMS,
    create_llm,
    extract_lp_content,
    generate_batch,
)
from dualrayrank.prompts.dualray_feedback import build_dualray_topk_feedback, verify_dual_ray
from dualrayrank.prompts.iis_feedback import build_iis_topk_feedback
from dualrayrank.prompts.repair_prompt import build_repair_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper
from dualrayrank.solver.lp_parser import parse_lp_string, strip_integrality


def main():
    parser = argparse.ArgumentParser(description="Run DualRay-TopK repair on infeasible instances")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attempt0-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/repair_dualray_topk")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    parser.add_argument("--iis-results", default="dualrayrank/results/iis_topk_results.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    attempt0_dir = Path(args.attempt0_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    infeasible_path = attempt0_dir / "infeasible_instances.json"
    with open(infeasible_path, "r", encoding="utf-8") as f:
        infeasible_list = json.load(f)
    print(f"Loaded {len(infeasible_list)} infeasible instances")

    iis_results_path = Path(args.iis_results)
    iis_size_map = {}
    if iis_results_path.exists():
        with open(iis_results_path, "r", encoding="utf-8") as f:
            iis_data = json.load(f)
        for item in iis_data.get("per_instance", []):
            iid = item["instance_id"]
            diff = item["difficulty"]
            iis_size_map[f"{diff}_{iid}"] = item.get("iis_size")
        print(f"Loaded IIS sizes for {len(iis_size_map)} instances from {iis_results_path}")

    all_instances = load_mamo()
    instance_map = {}
    for inst in all_instances:
        key = f"{inst['difficulty']}_{inst['id']}"
        instance_map[key] = inst

    wrapper = HiGHSWrapper(presolve="off", iis_strategy=2)

    repair_items = []
    skip_log = []

    for entry in infeasible_list:
        iid = entry["instance_id"]
        difficulty = entry["difficulty"]
        key = f"{difficulty}_{iid}"
        inst = instance_map.get(key)
        if inst is None:
            print(f"WARNING: instance {key} not found in MAMO data, skipping")
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "not_found_in_mamo"})
            continue

        lp_path = attempt0_dir / f"{key}.lp"
        if not lp_path.exists():
            print(f"WARNING: LP file {lp_path} not found, skipping")
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "lp_file_missing"})
            continue

        attempt0_lp = lp_path.read_text(encoding="utf-8").strip()
        if not attempt0_lp:
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "lp_file_empty"})
            continue

        solve_result, iis, dual_ray = wrapper.solve_and_diagnose(attempt0_lp)

        if solve_result.status != "infeasible":
            print(f"  {key}: LP relaxation is {solve_result.status} (MILP-infeasible-but-LP-relaxation-feasible), skipping")
            skip_log.append({
                "instance_id": iid,
                "difficulty": difficulty,
                "reason": "lp_relaxation_not_infeasible",
                "lp_relaxation_status": solve_result.status,
            })
            continue

        lp_model = parse_lp_string(strip_integrality(attempt0_lp))

        dr_check = verify_dual_ray(dual_ray)
        feedback_source = "dualray"
        feedback = build_dualray_topk_feedback(dual_ray, lp_model, k=args.k)

        if feedback is None:
            print(f"  {key}: Dual ray invalid ({dr_check['reason']}), falling back to IIS-TopK")
            feedback_source = "iis_fallback"
            if iis is not None and iis.success:
                feedback = build_iis_topk_feedback(iis, lp_model, k=args.k)
            else:
                print(f"  {key}: IIS also failed, skipping")
                skip_log.append({
                    "instance_id": iid,
                    "difficulty": difficulty,
                    "reason": "both_dualray_and_iis_failed",
                    "dual_ray_reason": dr_check["reason"],
                })
                continue

        prompt = build_repair_prompt(inst["Question"], attempt0_lp, feedback)

        top_k_names = []
        if dual_ray and dr_check["valid"]:
            entries_sorted = sorted(
                [(n, abs(m)) for n, m in zip(dual_ray.row_names, dual_ray.multipliers) if abs(m) > 1e-10],
                key=lambda x: x[1], reverse=True,
            )
            top_k_names = [n for n, _ in entries_sorted[:args.k]]

        repair_items.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "key": key,
            "prompt": prompt,
            "dual_ray_extracted": dr_check["valid"],
            "num_nonzero_multipliers": dr_check["num_nonzero"],
            "top_k_constraints": top_k_names,
            "feedback_text": feedback,
            "feedback_source": feedback_source,
        })

    print(f"\nRepairable instances: {len(repair_items)}")
    print(f"Skipped instances: {len(skip_log)}")
    dualray_count = sum(1 for item in repair_items if item["feedback_source"] == "dualray")
    fallback_count = sum(1 for item in repair_items if item["feedback_source"] == "iis_fallback")
    print(f"  Using dual ray feedback: {dualray_count}")
    print(f"  Fell back to IIS-TopK:  {fallback_count}")

    if not repair_items:
        print("No repairable instances, exiting.")
        sys.exit(0)

    print(f"\nInitializing vLLM with model={args.model}, tp={args.tensor_parallel_size}")
    llm = create_llm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    prompts = [item["prompt"] for item in repair_items]
    print(f"Running inference on {len(prompts)} repair prompts...")
    raw_outputs = generate_batch(llm, prompts, DEFAULT_SAMPLING_PARAMS)
    print("Inference complete")

    log_path = output_dir / "repair_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as log_f:
        for item, raw_output in zip(repair_items, raw_outputs):
            lp_content = extract_lp_content(raw_output)
            lp_out_path = output_dir / f"{item['key']}.lp"
            lp_out_path.write_text(lp_content, encoding="utf-8")

            log_entry = {
                "instance_id": item["instance_id"],
                "difficulty": item["difficulty"],
                "dual_ray_extracted": item["dual_ray_extracted"],
                "num_nonzero_multipliers": item["num_nonzero_multipliers"],
                "top_k_constraints": item["top_k_constraints"],
                "feedback_text": item["feedback_text"],
                "feedback_source": item["feedback_source"],
                "raw_output": raw_output,
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        for entry in skip_log:
            entry["skipped"] = True
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(repair_items)} repaired .lp files to {output_dir}")
    print(f"Repair log: {log_path}")

    print("\n=== Evaluating repaired files ===")
    eval_wrapper = HiGHSWrapper()
    per_instance = []
    repair_successes = 0

    attempt0_pass_count = 501
    total_instances = 863

    for item in repair_items:
        key = item["key"]
        inst = instance_map[key]
        gt = parse_ground_truth(inst["Answer"])

        repaired_path = output_dir / f"{key}.lp"
        record = classify_result(repaired_path, gt, eval_wrapper)

        repair_status = record["classification"]
        is_success = repair_status == "pass"
        if is_success:
            repair_successes += 1

        per_instance.append({
            "instance_id": item["instance_id"],
            "difficulty": item["difficulty"],
            "attempt0_status": "fail-infeasible",
            "repair_status": repair_status,
            "repair_objective": record.get("objective"),
            "ground_truth": gt,
            "dual_ray_valid": item["dual_ray_extracted"],
            "num_nonzero_multipliers": item["num_nonzero_multipliers"],
            "feedback_source": item["feedback_source"],
            "iis_size": iis_size_map.get(key),
        })

    for entry in skip_log:
        per_instance.append({
            "instance_id": entry["instance_id"],
            "difficulty": entry["difficulty"],
            "attempt0_status": "fail-infeasible",
            "repair_status": "skipped",
            "repair_objective": None,
            "ground_truth": None,
            "dual_ray_valid": None,
            "num_nonzero_multipliers": None,
            "feedback_source": None,
            "iis_size": iis_size_map.get(f"{entry['difficulty']}_{entry['instance_id']}"),
            "skip_reason": entry.get("reason"),
        })

    repaired_total = len(repair_items)
    infeasible_total = len(infeasible_list)

    easy_repair = [r for r in per_instance if r["difficulty"] == "EasyLP" and r["repair_status"] == "pass"]
    complex_repair = [r for r in per_instance if r["difficulty"] == "ComplexLP" and r["repair_status"] == "pass"]

    easy_pass_attempt0 = 464
    complex_pass_attempt0 = 37
    easy_total = 652
    complex_total = 211

    truncation_subset = [r for r in per_instance if r.get("iis_size") is not None and r["iis_size"] > args.k]
    truncation_successes = sum(1 for r in truncation_subset if r["repair_status"] == "pass")
    truncation_total = len(truncation_subset)

    results_summary = {
        "experiment": "dualray_topk_repair",
        "condition": "B (DualRay-TopK, names only)",
        "k": args.k,
        "model": args.model,
        "infeasible_instances": infeasible_total,
        "repairable_instances": repaired_total,
        "skipped_instances": len(skip_log),
        "dualray_feedback_count": dualray_count,
        "iis_fallback_count": fallback_count,
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

    results_path = results_dir / "dualray_topk_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"DualRay-TopK REPAIR RESULTS (Condition B, K={args.k})")
    print(f"{'='*60}")
    print(f"Infeasible instances: {infeasible_total}")
    print(f"Repairable (LP-relaxation infeasible): {repaired_total}")
    print(f"Skipped: {len(skip_log)}")
    print(f"Dual ray feedback: {dualray_count}, IIS fallback: {fallback_count}")
    print(f"Repair successes: {repair_successes}")
    print(f"Repair success rate (on infeasible): {repair_successes}/{infeasible_total} = {results_summary['repair_success_rate']:.4f}")
    print(f"Repair success rate (on repairable): {repair_successes}/{repaired_total} = {results_summary['repair_success_rate_on_repairable']:.4f}")
    print(f"\nPass@1 after <=2 attempts:")
    p = results_summary["pass_at_1_after_repair"]
    print(f"  Overall:   {p['overall']:.4f} ({p['overall_count']}/{p['overall_total']})")
    print(f"  EasyLP:    {p['EasyLP']:.4f} ({p['EasyLP_count']}/{p['EasyLP_total']})")
    print(f"  ComplexLP: {p['ComplexLP']:.4f} ({p['ComplexLP_count']}/{p['ComplexLP_total']})")
    t = results_summary["truncation_regime"]
    print(f"\nTruncation regime (IIS size > {args.k}):")
    print(f"  Instances: {t['total_instances']}")
    print(f"  Repair successes: {t['repair_successes']}/{t['total_instances']} = {t['repair_success_rate']:.4f}")


if __name__ == "__main__":
    main()
