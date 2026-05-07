"""Best-of-N IIS-TopK repair: control baseline for comparison with DualRay best-of-N.

Same setup as DualRay best-of-N but using IIS-TopK feedback (Condition A) with the
enhanced repair prompt and K=10. This isolates the effect of the feedback type.

Usage:
    python -m dualrayrank.scripts.run_iis_bestofn_repair [--n 5] [--k 10]
"""

import argparse
import json
import sys
from pathlib import Path

from vllm import SamplingParams

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.inference.vllm_runner import create_llm, extract_lp_content
from dualrayrank.prompts.iis_feedback import build_iis_topk_feedback
from dualrayrank.prompts.repair_prompt import build_enhanced_repair_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper
from dualrayrank.solver.lp_parser import parse_lp_string, strip_integrality


def main():
    parser = argparse.ArgumentParser(description="Best-of-N IIS-TopK repair baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attempt0-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/repair_iis_bestofn")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=5, help="Number of diverse samples per instance")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
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

        solve_result, iis, dual_ray = wrapper.solve_and_diagnose(attempt0_lp)
        if solve_result.status != "infeasible":
            skip_log.append({
                "instance_id": iid, "difficulty": difficulty,
                "reason": "lp_relaxation_not_infeasible",
            })
            continue

        if iis is None or not iis.success:
            skip_log.append({"instance_id": iid, "difficulty": difficulty, "reason": "iis_extraction_failed"})
            continue

        lp_model = parse_lp_string(strip_integrality(attempt0_lp))
        feedback = build_iis_topk_feedback(iis, lp_model, k=args.k)
        prompt = build_enhanced_repair_prompt(inst["Question"], attempt0_lp, feedback)

        candidates.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "key": key,
            "prompt": prompt,
            "ground_truth": parse_ground_truth(inst["Answer"]),
            "iis_size": len(iis.row_names),
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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=4096,
        top_p=0.95,
        n=args.n,
        seed=args.seed,
    )
    greedy_params = SamplingParams(temperature=0.0, max_tokens=4096, top_p=1.0)

    prompts = [c["prompt"] for c in candidates]

    print(f"Running greedy inference on {len(prompts)} prompts...")
    greedy_outputs = llm.generate(prompts, greedy_params)

    print(f"Running best-of-{args.n} inference (temp={args.temperature}) on {len(prompts)} prompts...")
    sampled_outputs = llm.generate(prompts, sampling_params)

    per_instance_results = []
    repair_successes = 0

    for i, cand in enumerate(candidates):
        key = cand["key"]
        gt = cand["ground_truth"]

        all_outputs = []
        greedy_text = greedy_outputs[i].outputs[0].text
        all_outputs.append(("greedy", greedy_text))
        for j, out in enumerate(sampled_outputs[i].outputs):
            all_outputs.append((f"sample_{j}", out.text))

        best_status = None
        best_objective = None
        best_idx = None
        sample_results = []

        sample_dir = output_dir / "samples" / key
        sample_dir.mkdir(parents=True, exist_ok=True)

        for idx, (label, raw_output) in enumerate(all_outputs):
            lp_content = extract_lp_content(raw_output)
            sample_path = sample_dir / f"{label}.lp"
            sample_path.write_text(lp_content, encoding="utf-8")

            record = classify_result(sample_path, gt, eval_wrapper)
            status = record["classification"]
            sample_results.append({"label": label, "status": status, "objective": record.get("objective")})

            if status == "pass" and best_status != "pass":
                best_status = "pass"
                best_objective = record.get("objective")
                best_idx = idx

            if best_status != "pass" and status not in ("fail-infeasible", "fail-error"):
                if best_status is None or best_status in ("fail-infeasible", "fail-error"):
                    best_status = status
                    best_objective = record.get("objective")
                    best_idx = idx

        if best_idx is None:
            best_idx = 0
            best_status = sample_results[0]["status"]
            best_objective = sample_results[0].get("objective")

        best_lp = extract_lp_content(all_outputs[best_idx][1])
        final_path = output_dir / f"{key}.lp"
        final_path.write_text(best_lp, encoding="utf-8")

        is_pass = best_status == "pass"
        if is_pass:
            repair_successes += 1

        print(f"  {key}: best={all_outputs[best_idx][0]} -> {best_status}" +
              (f" (obj={best_objective})" if best_objective else "") +
              f" [{sum(1 for s in sample_results if s['status'] == 'pass')}/{len(sample_results)} passed]")

        per_instance_results.append({
            "instance_id": cand["instance_id"],
            "difficulty": cand["difficulty"],
            "repair_status": best_status,
            "repair_objective": best_objective,
            "ground_truth": gt,
            "iis_size": cand.get("iis_size"),
            "best_sample": all_outputs[best_idx][0],
            "total_samples": len(all_outputs),
            "samples_passed": sum(1 for s in sample_results if s["status"] == "pass"),
            "sample_results": sample_results,
        })

    for entry in skip_log:
        per_instance_results.append({
            "instance_id": entry["instance_id"],
            "difficulty": entry["difficulty"],
            "repair_status": "skipped",
            "skip_reason": entry.get("reason"),
        })

    infeasible_total = len(infeasible_list)
    easy_repair = [r for r in per_instance_results if r.get("difficulty") == "EasyLP" and r.get("repair_status") == "pass"]
    complex_repair = [r for r in per_instance_results if r.get("difficulty") == "ComplexLP" and r.get("repair_status") == "pass"]
    easy_pass_attempt0 = 464
    complex_pass_attempt0 = 37
    easy_total = 652
    complex_total = 211
    attempt0_pass_count = 501
    total_instances = 863

    truncation_subset = [r for r in per_instance_results if r.get("iis_size") is not None and r["iis_size"] > 5]
    truncation_successes = sum(1 for r in truncation_subset if r.get("repair_status") == "pass")

    results_summary = {
        "experiment": "iis_bestofn_repair",
        "condition": "A+ (IIS-TopK, enhanced prompt, best-of-N)",
        "k": args.k,
        "n": args.n,
        "temperature": args.temperature,
        "model": args.model,
        "infeasible_instances": infeasible_total,
        "repairable_instances": len(candidates),
        "repair_successes": repair_successes,
        "repair_success_rate": repair_successes / infeasible_total if infeasible_total > 0 else 0,
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
            "total_instances": len(truncation_subset),
            "repair_successes": truncation_successes,
            "repair_success_rate": truncation_successes / len(truncation_subset) if truncation_subset else 0,
        },
        "per_instance": per_instance_results,
    }

    results_path = results_dir / "iis_bestofn_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"IIS-TopK BEST-OF-{args.n} REPAIR (K={args.k}, temp={args.temperature})")
    print(f"{'='*60}")
    print(f"Repair successes: {repair_successes}/{infeasible_total} = {results_summary['repair_success_rate']:.4f}")
    p = results_summary["pass_at_1_after_repair"]
    print(f"  Overall:   {p['overall']:.4f} ({p['overall_count']}/{p['overall_total']})")
    print(f"  EasyLP:    {p['EasyLP']:.4f} ({p['EasyLP_count']}/{p['EasyLP_total']})")
    print(f"  ComplexLP: {p['ComplexLP']:.4f} ({p['ComplexLP_count']}/{p['ComplexLP_total']})")


if __name__ == "__main__":
    main()
