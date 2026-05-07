"""Optimized DualRay-TopK+Weights repair (Condition C) with multi-attempt iterative repair.

Improvements over Condition B:
  1. Condition C: includes normalized Farkas multiplier weights in feedback
  2. Enhanced repair prompt with actionable repair strategies
  3. Multi-attempt iterative repair (up to 3 attempts per instance)
  4. Increased K=10 for more complete diagnostic info
  5. Sampling with temperature>0 on retry attempts for diversity

Usage:
    python -m dualrayrank.scripts.run_dualray_weighted_repair [--k 10] [--max-attempts 3]
"""

import argparse
import json
import sys
from pathlib import Path

from vllm import SamplingParams

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.inference.vllm_runner import create_llm, extract_lp_content, generate_batch
from dualrayrank.prompts.dualray_weighted_feedback import build_dualray_weighted_feedback
from dualrayrank.prompts.dualray_feedback import verify_dual_ray
from dualrayrank.prompts.iis_feedback import build_iis_topk_feedback
from dualrayrank.prompts.repair_prompt import build_enhanced_repair_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper
from dualrayrank.solver.lp_parser import parse_lp_string, strip_integrality


def diagnose_and_build_feedback(lp_text, wrapper, k, iis_fallback=True):
    solve_result, iis, dual_ray = wrapper.solve_and_diagnose(lp_text)
    if solve_result.status != "infeasible":
        return solve_result, None, None, "not_infeasible", None

    lp_model = parse_lp_string(strip_integrality(lp_text))
    dr_check = verify_dual_ray(dual_ray)

    feedback_source = "dualray_weighted"
    feedback = build_dualray_weighted_feedback(dual_ray, lp_model, k=k)
    if feedback is None and iis_fallback:
        feedback_source = "iis_fallback"
        if iis is not None and iis.success:
            feedback = build_iis_topk_feedback(iis, lp_model, k=k)

    return solve_result, feedback, dr_check, feedback_source, dual_ray


def main():
    parser = argparse.ArgumentParser(description="Optimized DualRay-TopK+Weights multi-attempt repair")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attempt0-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/repair_dualray_weighted")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    parser.add_argument("--iis-results", default="dualrayrank/results/iis_topk_results.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
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

    all_instances = load_mamo()
    instance_map = {}
    for inst in all_instances:
        key = f"{inst['difficulty']}_{inst['id']}"
        instance_map[key] = inst

    wrapper = HiGHSWrapper(presolve="off", iis_strategy=2)
    eval_wrapper = HiGHSWrapper()

    candidates = []
    skip_log = []

    for entry in infeasible_list:
        iid = entry["instance_id"]
        difficulty = entry["difficulty"]
        key = f"{difficulty}_{iid}"
        inst = instance_map.get(key)
        if inst is None:
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "not_found_in_mamo"})
            continue

        lp_path = attempt0_dir / f"{key}.lp"
        if not lp_path.exists():
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "lp_file_missing"})
            continue

        attempt0_lp = lp_path.read_text(encoding="utf-8").strip()
        if not attempt0_lp:
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "lp_file_empty"})
            continue

        solve_result, feedback, dr_check, feedback_source, dual_ray = diagnose_and_build_feedback(
            attempt0_lp, wrapper, args.k
        )
        if feedback_source == "not_infeasible":
            skip_log.append({
                "instance_id": iid, "difficulty": difficulty,
                "reason": "lp_relaxation_not_infeasible",
                "lp_relaxation_status": solve_result.status,
            })
            continue

        if feedback is None:
            skip_log.append({
                "instance_id": iid, "difficulty": difficulty,
                "reason": "both_dualray_and_iis_failed",
            })
            continue

        candidates.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "key": key,
            "question": inst["Question"],
            "attempt0_lp": attempt0_lp,
            "ground_truth": parse_ground_truth(inst["Answer"]),
            "initial_feedback": feedback,
            "initial_feedback_source": feedback_source,
            "initial_dr_check": dr_check,
            "initial_dual_ray": dual_ray,
            "iis_size": iis_size_map.get(key),
        })

    print(f"\nRepairable instances: {len(candidates)}")
    print(f"Skipped instances: {len(skip_log)}")

    if not candidates:
        print("No repairable instances, exiting.")
        sys.exit(0)

    print(f"\nInitializing vLLM with model={args.model}, tp={args.tensor_parallel_size}")
    llm = create_llm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    greedy_params = SamplingParams(temperature=0.0, max_tokens=4096, top_p=1.0)
    sampling_params_list = [
        SamplingParams(temperature=0.6, max_tokens=4096, top_p=0.95, seed=42),
        SamplingParams(temperature=0.6, max_tokens=4096, top_p=0.95, seed=123),
    ]

    per_instance_results = []
    repair_successes = 0

    active = list(range(len(candidates)))

    for attempt_num in range(args.max_attempts):
        if not active:
            break

        params = greedy_params if attempt_num == 0 else sampling_params_list[min(attempt_num - 1, len(sampling_params_list) - 1)]
        print(f"\n{'='*60}")
        print(f"ATTEMPT {attempt_num + 1}/{args.max_attempts} ({len(active)} instances, temp={params.temperature})")
        print(f"{'='*60}")

        prompts = []
        for idx in active:
            cand = candidates[idx]
            if attempt_num == 0:
                feedback = cand["initial_feedback"]
            else:
                prev_lp = cand.get("latest_lp", cand["attempt0_lp"])
                _, feedback, _, fb_source, _ = diagnose_and_build_feedback(prev_lp, wrapper, args.k)
                if feedback is None:
                    feedback = cand["initial_feedback"]

            lp_for_prompt = cand.get("latest_lp", cand["attempt0_lp"])
            prompt = build_enhanced_repair_prompt(cand["question"], lp_for_prompt, feedback)
            prompts.append(prompt)

        raw_outputs = generate_batch(llm, prompts, params)
        print(f"Inference complete for attempt {attempt_num + 1}")

        still_active = []
        for i, idx in enumerate(active):
            cand = candidates[idx]
            raw_output = raw_outputs[i]
            lp_content = extract_lp_content(raw_output)

            attempt_dir = output_dir / f"attempt{attempt_num + 1}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            lp_out_path = attempt_dir / f"{cand['key']}.lp"
            lp_out_path.write_text(lp_content, encoding="utf-8")

            record = classify_result(lp_out_path, cand["ground_truth"], eval_wrapper)
            status = record["classification"]
            print(f"  {cand['key']}: attempt {attempt_num+1} -> {status}" +
                  (f" (obj={record.get('objective')})" if record.get("objective") else ""))

            if status == "pass":
                cand["final_status"] = "pass"
                cand["final_attempt"] = attempt_num + 1
                cand["final_objective"] = record.get("objective")
                final_path = output_dir / f"{cand['key']}.lp"
                final_path.write_text(lp_content, encoding="utf-8")
                repair_successes += 1
            elif attempt_num < args.max_attempts - 1 and status in ("fail-infeasible",):
                cand["latest_lp"] = lp_content
                still_active.append(idx)
            else:
                cand["final_status"] = status
                cand["final_attempt"] = attempt_num + 1
                cand["final_objective"] = record.get("objective")
                final_path = output_dir / f"{cand['key']}.lp"
                final_path.write_text(lp_content, encoding="utf-8")

            cand.setdefault("attempt_log", []).append({
                "attempt": attempt_num + 1,
                "status": status,
                "objective": record.get("objective"),
                "temperature": params.temperature,
            })

        active = still_active
        print(f"  Successes so far: {repair_successes}, still infeasible: {len(active)}")

    for idx in active:
        cand = candidates[idx]
        last_log = cand["attempt_log"][-1] if cand.get("attempt_log") else {}
        cand["final_status"] = last_log.get("status", "fail-infeasible")
        cand["final_attempt"] = args.max_attempts
        cand["final_objective"] = last_log.get("objective")

    dualray_count = sum(1 for c in candidates if c.get("initial_feedback_source") == "dualray_weighted")
    fallback_count = sum(1 for c in candidates if c.get("initial_feedback_source") == "iis_fallback")

    for cand in candidates:
        dr_check = cand.get("initial_dr_check") or {}
        per_instance_results.append({
            "instance_id": cand["instance_id"],
            "difficulty": cand["difficulty"],
            "attempt0_status": "fail-infeasible",
            "repair_status": cand.get("final_status", "fail-infeasible"),
            "repair_objective": cand.get("final_objective"),
            "ground_truth": cand["ground_truth"],
            "dual_ray_valid": dr_check.get("valid"),
            "num_nonzero_multipliers": dr_check.get("num_nonzero"),
            "feedback_source": cand.get("initial_feedback_source"),
            "iis_size": cand.get("iis_size"),
            "attempts_used": cand.get("final_attempt", args.max_attempts),
            "attempt_log": cand.get("attempt_log", []),
        })

    for entry in skip_log:
        per_instance_results.append({
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

    infeasible_total = len(infeasible_list)
    repaired_total = len(candidates)

    easy_repair = [r for r in per_instance_results if r["difficulty"] == "EasyLP" and r["repair_status"] == "pass"]
    complex_repair = [r for r in per_instance_results if r["difficulty"] == "ComplexLP" and r["repair_status"] == "pass"]
    easy_pass_attempt0 = 464
    complex_pass_attempt0 = 37
    easy_total = 652
    complex_total = 211
    attempt0_pass_count = 501
    total_instances = 863

    truncation_subset = [r for r in per_instance_results if r.get("iis_size") is not None and r["iis_size"] > 5]
    truncation_successes = sum(1 for r in truncation_subset if r["repair_status"] == "pass")
    truncation_total = len(truncation_subset)

    results_summary = {
        "experiment": "dualray_weighted_repair",
        "condition": "C (DualRay-TopK+Weights, enhanced prompt, multi-attempt)",
        "k": args.k,
        "max_attempts": args.max_attempts,
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
            "description": "Instances where IIS size > 5 (original K=5 truncation regime)",
            "total_instances": truncation_total,
            "repair_successes": truncation_successes,
            "repair_success_rate": truncation_successes / truncation_total if truncation_total > 0 else 0,
        },
        "per_instance": per_instance_results,
    }

    results_path = results_dir / "dualray_weighted_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"DualRay-TopK+Weights REPAIR RESULTS (Condition C, K={args.k}, max_attempts={args.max_attempts})")
    print(f"{'='*60}")
    print(f"Infeasible instances: {infeasible_total}")
    print(f"Repairable: {repaired_total}")
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
    print(f"\nTruncation regime (IIS size > 5):")
    print(f"  Instances: {t['total_instances']}")
    print(f"  Repair successes: {t['repair_successes']}/{t['total_instances']} = {t['repair_success_rate']:.4f}")

    for r in per_instance_results:
        if r["repair_status"] == "pass":
            print(f"\nRepaired: {r['difficulty']}_{r['instance_id']} (attempt {r['attempts_used']}, obj={r['repair_objective']}, gt={r['ground_truth']})")


if __name__ == "__main__":
    main()
