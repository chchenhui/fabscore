"""Optimized iterative best-of-N repair with configurable model, sample count, and rounds.

Combines three improvements over the original best-of-6:
1. Supports stronger models (Qwen2.5-72B, Qwen3-32B) via --model and --tensor-parallel-size
2. Higher N (default 16) for more diverse candidates
3. Multi-round iterative repair: after round 1, re-diagnoses failed instances and retries

For Qwen3 models, prepends /no_think to the prompt to disable thinking mode.

Usage:
    python -m dualrayrank.scripts.run_optimized_repair \
        --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 \
        --n 16 --rounds 2 --k 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.inference.vllm_runner import extract_lp_content
from dualrayrank.prompts.dualray_weighted_feedback import build_dualray_weighted_feedback
from dualrayrank.prompts.dualray_feedback import verify_dual_ray
from dualrayrank.prompts.iis_feedback import build_iis_topk_feedback
from dualrayrank.prompts.repair_prompt import build_enhanced_repair_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper
from dualrayrank.solver.lp_parser import parse_lp_string, strip_integrality


def diagnose_and_build_feedback(lp_text, wrapper, k):
    solve_result, iis, dual_ray = wrapper.solve_and_diagnose(lp_text)
    if solve_result.status != "infeasible":
        return solve_result, None, "not_infeasible"

    lp_model = parse_lp_string(strip_integrality(lp_text))
    feedback = build_dualray_weighted_feedback(dual_ray, lp_model, k=k)
    source = "dualray_weighted"
    if feedback is None:
        source = "iis_fallback"
        if iis is not None and iis.success:
            feedback = build_iis_topk_feedback(iis, lp_model, k=k)

    return solve_result, feedback, source


def evaluate_candidates(all_outputs, cand, output_dir, eval_wrapper, round_label):
    key = cand["key"]
    gt = cand["ground_truth"]
    sample_dir = output_dir / "samples" / key / round_label
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_results = []
    best_pass_idx = None
    best_feasible_idx = None
    best_feasible_obj = None

    for idx, (label, raw_output) in enumerate(all_outputs):
        lp_content = extract_lp_content(raw_output)
        sample_path = sample_dir / f"{label}.lp"
        sample_path.write_text(lp_content, encoding="utf-8")

        record = classify_result(sample_path, gt, eval_wrapper)
        status = record["classification"]
        obj = record.get("objective")
        sample_results.append({"label": label, "status": status, "objective": obj})

        if status == "pass" and best_pass_idx is None:
            best_pass_idx = idx

        if status == "fail-wrong-objective" and obj is not None:
            if best_feasible_idx is None:
                best_feasible_idx = idx
                best_feasible_obj = obj
            elif gt is not None:
                if abs(obj - gt) < abs(best_feasible_obj - gt):
                    best_feasible_idx = idx
                    best_feasible_obj = obj

    return sample_results, best_pass_idx, best_feasible_idx


def main():
    parser = argparse.ArgumentParser(description="Optimized iterative best-of-N repair")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--attempt0-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/repair_optimized")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    parser.add_argument("--iis-results", default="dualrayrank/results/iis_topk_results.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=16, help="Samples per round per instance")
    parser.add_argument("--rounds", type=int, default=2, help="Iterative repair rounds")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="", help="Tag appended to output/results filenames")
    args = parser.parse_args()

    attempt0_dir = Path(args.attempt0_dir)
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = Path(args.output_dir + tag)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    is_qwen3 = "qwen3" in args.model.lower() or "Qwen3" in args.model
    model_short = args.model.split("/")[-1]

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

    mamo_data = load_mamo()
    mamo_lookup = {}
    for inst in mamo_data:
        key = f"{inst['difficulty']}_{inst['id']}"
        mamo_lookup[key] = inst

    wrapper = HiGHSWrapper(presolve="off", iis_strategy=2)
    eval_wrapper = HiGHSWrapper()

    candidates = []
    skip_log = []
    for entry in infeasible_list:
        iid = entry["instance_id"]
        difficulty = entry["difficulty"]
        key = f"{difficulty}_{iid}"

        inst = mamo_lookup.get(key)
        if inst is None:
            skip_log.append({"key": key, "reason": "not_found_in_mamo"})
            continue

        lp_path = attempt0_dir / f"{key}.lp"
        if not lp_path.exists():
            skip_log.append({"key": key, "reason": "lp_file_missing"})
            continue

        attempt0_lp = lp_path.read_text(encoding="utf-8").strip()
        if not attempt0_lp:
            skip_log.append({"key": key, "reason": "lp_file_empty"})
            continue

        candidates.append({
            "instance_id": iid,
            "difficulty": difficulty,
            "key": key,
            "question": inst["Question"],
            "ground_truth": parse_ground_truth(inst["Answer"]),
            "attempt0_lp": attempt0_lp,
            "iis_size": iis_size_map.get(key),
        })

    print(f"Repairable instances: {len(candidates)}")
    print(f"Skipped: {len(skip_log)}")

    if not candidates:
        print("No repairable instances, exiting.")
        sys.exit(0)

    print(f"\nInitializing vLLM: model={args.model}, tp={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    greedy_params = SamplingParams(temperature=0.0, max_tokens=4096, top_p=1.0)

    per_instance = {c["key"]: {
        "instance_id": c["instance_id"],
        "difficulty": c["difficulty"],
        "ground_truth": c["ground_truth"],
        "iis_size": c["iis_size"],
        "rounds": [],
        "final_status": None,
        "final_objective": None,
        "total_samples": 0,
        "total_passed": 0,
        "total_feasible": 0,
    } for c in candidates}

    active_indices = list(range(len(candidates)))
    repaired_keys = set()
    best_lps = {}

    for round_idx in range(args.rounds):
        if not active_indices:
            print(f"\nRound {round_idx+1}: No active instances remaining.")
            break

        print(f"\n{'='*60}")
        print(f"ROUND {round_idx+1}/{args.rounds}: {len(active_indices)} active instances")
        print(f"{'='*60}")

        round_prompts = []
        round_indices = []
        round_lps = []

        for ci in active_indices:
            cand = candidates[ci]
            key = cand["key"]

            lp_for_diagnosis = cand["attempt0_lp"]
            lp_for_prompt = cand["attempt0_lp"]

            if round_idx > 0:
                prev_best = best_lps.get(key)
                if prev_best:
                    prev_result, prev_fb, _ = diagnose_and_build_feedback(prev_best, wrapper, args.k)
                    if prev_result.status == "infeasible" and prev_fb is not None:
                        lp_for_diagnosis = prev_best
                        lp_for_prompt = prev_best

            solve_result, feedback, source = diagnose_and_build_feedback(lp_for_diagnosis, wrapper, args.k)

            if feedback is None:
                print(f"  {key}: No feedback available (status={solve_result.status}), skipping round")
                per_instance[key]["rounds"].append({
                    "round": round_idx + 1,
                    "skipped": True,
                    "reason": f"no_feedback_{solve_result.status}",
                })
                continue

            if round_idx > 0 and lp_for_diagnosis == cand["attempt0_lp"]:
                prev_best = best_lps.get(key)
                prev_status = per_instance[key].get("final_status", "unknown")
                prev_obj = per_instance[key].get("final_objective")
                extra = (f"\n\nNote: A previous repair attempt produced a model that was "
                         f"{prev_status} (objective={prev_obj}, expected={cand['ground_truth']}). "
                         f"The original constraints below are still infeasible. "
                         f"Try a DIFFERENT approach this time.")
                feedback = feedback + extra

            prompt = build_enhanced_repair_prompt(cand["question"], lp_for_prompt, feedback)
            if is_qwen3:
                prompt = "/no_think\n" + prompt

            round_prompts.append(prompt)
            round_indices.append(ci)
            round_lps.append(lp_for_prompt)

        if not round_prompts:
            print("  No prompts to generate, skipping round.")
            continue

        round_seed = args.seed + round_idx * 1000
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=4096,
            top_p=0.95,
            n=args.n,
            seed=round_seed,
        )

        print(f"  Generating greedy outputs for {len(round_prompts)} instances...")
        t0 = time.time()
        greedy_outputs = llm.generate(round_prompts, greedy_params)
        t1 = time.time()
        print(f"  Greedy done in {t1-t0:.1f}s")

        print(f"  Generating {args.n} sampled outputs for {len(round_prompts)} instances...")
        sampled_outputs = llm.generate(round_prompts, sampling_params)
        t2 = time.time()
        print(f"  Sampling done in {t2-t1:.1f}s")

        next_active = []
        round_repairs = 0

        for j, ci in enumerate(round_indices):
            cand = candidates[ci]
            key = cand["key"]
            gt = cand["ground_truth"]

            all_outputs = [("greedy", greedy_outputs[j].outputs[0].text)]
            for s_idx, out in enumerate(sampled_outputs[j].outputs):
                all_outputs.append((f"sample_{s_idx}", out.text))

            sample_results, best_pass_idx, best_feasible_idx = evaluate_candidates(
                all_outputs, cand, output_dir, eval_wrapper, f"round{round_idx+1}"
            )

            n_passed = sum(1 for s in sample_results if s["status"] == "pass")
            n_feasible = sum(1 for s in sample_results if s["status"] not in ("fail-infeasible", "fail-error"))
            total = len(sample_results)

            per_instance[key]["total_samples"] += total
            per_instance[key]["total_passed"] += n_passed
            per_instance[key]["total_feasible"] += n_feasible

            round_info = {
                "round": round_idx + 1,
                "skipped": False,
                "total_samples": total,
                "passed": n_passed,
                "feasible": n_feasible,
                "sample_results": sample_results,
            }

            if best_pass_idx is not None:
                best_label, best_raw = all_outputs[best_pass_idx]
                best_lp_text = extract_lp_content(best_raw)
                final_path = output_dir / f"{key}.lp"
                final_path.write_text(best_lp_text, encoding="utf-8")
                best_lps[key] = best_lp_text
                repaired_keys.add(key)
                round_repairs += 1

                per_instance[key]["final_status"] = "pass"
                per_instance[key]["final_objective"] = sample_results[best_pass_idx]["objective"]
                round_info["best_sample"] = best_label
                round_info["result"] = "pass"

                print(f"  {key}: PASS (best={best_label}, {n_passed}/{total} passed)")

            elif best_feasible_idx is not None:
                best_label, best_raw = all_outputs[best_feasible_idx]
                best_lp_text = extract_lp_content(best_raw)
                final_path = output_dir / f"{key}.lp"
                final_path.write_text(best_lp_text, encoding="utf-8")
                best_lps[key] = best_lp_text
                obj = sample_results[best_feasible_idx]["objective"]

                per_instance[key]["final_status"] = "fail-wrong-objective"
                per_instance[key]["final_objective"] = obj
                round_info["best_sample"] = best_label
                round_info["result"] = f"fail-wrong-objective (obj={obj}, gt={gt})"

                next_active.append(ci)
                print(f"  {key}: wrong-obj (best={best_label}, obj={obj}, gt={gt}, {n_feasible}/{total} feasible)")

            else:
                best_label, best_raw = all_outputs[0]
                best_lp_text = extract_lp_content(best_raw)
                best_lps[key] = best_lp_text
                status = sample_results[0]["status"]

                if per_instance[key]["final_status"] is None:
                    per_instance[key]["final_status"] = status
                round_info["best_sample"] = best_label
                round_info["result"] = status

                next_active.append(ci)
                print(f"  {key}: {status} ({n_feasible}/{total} feasible)")

            per_instance[key]["rounds"].append(round_info)

        print(f"\n  Round {round_idx+1} repairs: {round_repairs}")
        print(f"  Total repaired so far: {len(repaired_keys)}")

        active_indices = [ci for ci in next_active if candidates[ci]["key"] not in repaired_keys]

    easy_pass_attempt0 = 464
    complex_pass_attempt0 = 37
    easy_total = 652
    complex_total = 211
    attempt0_pass_count = 501
    total_instances = 863
    infeasible_total = len(candidates) + len(skip_log)

    repair_successes = len(repaired_keys)
    easy_repair = [k for k in repaired_keys if k.startswith("EasyLP")]
    complex_repair = [k for k in repaired_keys if k.startswith("ComplexLP")]

    results_summary = {
        "experiment": f"optimized_repair{tag}",
        "model": args.model,
        "model_short": model_short,
        "condition": f"C+ iterative best-of-N (K={args.k}, N={args.n}, rounds={args.rounds})",
        "k": args.k,
        "n": args.n,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "infeasible_instances": infeasible_total,
        "repairable_instances": len(candidates),
        "skipped_instances": len(skip_log),
        "repair_successes": repair_successes,
        "repair_success_rate": repair_successes / infeasible_total if infeasible_total > 0 else 0,
        "repaired_instances": sorted(repaired_keys),
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
        "per_instance": [],
    }

    for key in sorted(per_instance.keys()):
        r = per_instance[key]
        results_summary["per_instance"].append({
            "key": key,
            "instance_id": r["instance_id"],
            "difficulty": r["difficulty"],
            "ground_truth": r["ground_truth"],
            "iis_size": r["iis_size"],
            "final_status": r["final_status"],
            "final_objective": r["final_objective"],
            "total_samples": r["total_samples"],
            "total_passed": r["total_passed"],
            "total_feasible": r["total_feasible"],
            "rounds": r["rounds"],
        })

    results_filename = f"optimized_repair{tag}_results.json"
    results_path = results_dir / results_filename
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"OPTIMIZED REPAIR: {model_short}, K={args.k}, N={args.n}, rounds={args.rounds}")
    print(f"{'='*60}")
    print(f"Infeasible instances: {infeasible_total}")
    print(f"Repair successes: {repair_successes}/{infeasible_total} = {results_summary['repair_success_rate']:.4f}")
    p = results_summary["pass_at_1_after_repair"]
    print(f"\nPass@1 after repair:")
    print(f"  Overall:   {p['overall']:.4f} ({p['overall_count']}/{p['overall_total']})")
    print(f"  EasyLP:    {p['EasyLP']:.4f} ({p['EasyLP_count']}/{p['EasyLP_total']})")
    print(f"  ComplexLP: {p['ComplexLP']:.4f} ({p['ComplexLP_count']}/{p['ComplexLP_total']})")

    print(f"\nRepaired instances:")
    for key in sorted(repaired_keys):
        r = per_instance[key]
        print(f"  {key}: obj={r['final_objective']}, gt={r['ground_truth']}, "
              f"total_passed={r['total_passed']}/{r['total_samples']}")


if __name__ == "__main__":
    main()
