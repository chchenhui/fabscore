"""IIS-TopK repair experiment (Condition A): repair infeasible attempt-0 instances
using IIS-based feedback with K=5 constraints.

Usage:
    python -m dualrayrank.scripts.run_iis_topk_repair [--output-dir DIR] [--model MODEL]
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
from dualrayrank.prompts.iis_feedback import build_iis_topk_feedback
from dualrayrank.prompts.repair_prompt import build_repair_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper
from dualrayrank.solver.lp_parser import parse_lp_string, strip_integrality


def main():
    parser = argparse.ArgumentParser(description="Run IIS-TopK repair on infeasible instances")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attempt0-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/repair_iis_topk")
    parser.add_argument("--results-dir", default="dualrayrank/results")
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

        if iis is None or not iis.success:
            print(f"  {key}: IIS extraction failed, skipping")
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "iis_extraction_failed"})
            continue

        lp_model = parse_lp_string(strip_integrality(attempt0_lp))
        feedback = build_iis_topk_feedback(iis, lp_model, k=args.k)
        prompt = build_repair_prompt(inst["Question"], attempt0_lp, feedback)

        repair_items.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "key": key,
            "prompt": prompt,
            "iis_size": len(iis.row_names),
            "selected_constraints": [n for n in sorted(iis.row_names)[:args.k]],
            "feedback_text": feedback,
            "was_truncated": len(iis.row_names) > args.k,
        })

    print(f"\nRepairable instances: {len(repair_items)}")
    print(f"Skipped instances: {len(skip_log)}")

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
                "iis_size": item["iis_size"],
                "selected_constraints": item["selected_constraints"],
                "was_truncated": item["was_truncated"],
                "feedback_text": item["feedback_text"],
                "raw_output": raw_output,
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    for entry in skip_log:
        with open(log_path, "a", encoding="utf-8") as log_f:
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
            "iis_size": item["iis_size"],
            "was_truncated": item["was_truncated"],
            "selected_constraints": item["selected_constraints"],
        })

    for entry in skip_log:
        per_instance.append({
            "instance_id": entry["instance_id"],
            "difficulty": entry["difficulty"],
            "attempt0_status": "fail-infeasible",
            "repair_status": "skipped",
            "repair_objective": None,
            "ground_truth": None,
            "iis_size": None,
            "was_truncated": None,
            "selected_constraints": None,
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

    results_summary = {
        "experiment": "iis_topk_repair",
        "condition": "A (IIS-TopK)",
        "k": args.k,
        "model": args.model,
        "infeasible_instances": infeasible_total,
        "repairable_instances": repaired_total,
        "skipped_instances": len(skip_log),
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
        "per_instance": per_instance,
    }

    results_path = results_dir / "iis_topk_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"IIS-TopK REPAIR RESULTS (Condition A, K={args.k})")
    print(f"{'='*60}")
    print(f"Infeasible instances: {infeasible_total}")
    print(f"Repairable (LP-relaxation infeasible): {repaired_total}")
    print(f"Skipped: {len(skip_log)}")
    print(f"Repair successes: {repair_successes}")
    print(f"Repair success rate (on infeasible): {repair_successes}/{infeasible_total} = {results_summary['repair_success_rate']:.4f}")
    print(f"Repair success rate (on repairable): {repair_successes}/{repaired_total} = {results_summary['repair_success_rate_on_repairable']:.4f}")
    print(f"\nPass@1 after <=2 attempts:")
    p = results_summary["pass_at_1_after_repair"]
    print(f"  Overall:   {p['overall']:.4f} ({p['overall_count']}/{p['overall_total']})")
    print(f"  EasyLP:    {p['EasyLP']:.4f} ({p['EasyLP_count']}/{p['EasyLP_total']})")
    print(f"  ComplexLP: {p['ComplexLP']:.4f} ({p['ComplexLP_count']}/{p['ComplexLP_total']})")


if __name__ == "__main__":
    main()
