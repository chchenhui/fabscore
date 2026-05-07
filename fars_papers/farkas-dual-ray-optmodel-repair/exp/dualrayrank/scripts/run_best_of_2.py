"""Best-of-2 stochastic inference scaling control: generate 2 samples per instance
with temperature=0.7, top_p=0.95, pick first solver-correct one. Runs 3 seeds,
reports mean +/- std pass@1. This controls whether a second inference call alone
(without solver feedback) improves pass@1.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.evaluation.evaluate import classify_result, parse_ground_truth
from dualrayrank.inference.vllm_runner import create_llm, extract_lp_content
from dualrayrank.prompts.generation_prompt import build_generation_prompt
from dualrayrank.solver.highs_wrapper import HiGHSWrapper


SEEDS = [42, 123, 456]


def run_seed(llm, prompts, instances, seed, output_base, wrapper):
    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        seed=seed,
        n=2,
    )

    print(f"\n{'='*60}")
    print(f"Seed {seed}: generating 2 samples for {len(prompts)} instances")
    print(f"{'='*60}")

    raw_outputs = llm.generate(prompts, params)

    seed_dir = output_base / f"seed_{seed}"
    sample_dirs = [seed_dir / "sample_0", seed_dir / "sample_1"]
    for d in sample_dirs:
        d.mkdir(parents=True, exist_ok=True)

    per_instance = []
    for inst, output in zip(instances, raw_outputs):
        iid = inst["id"]
        diff = inst["difficulty"]
        gt = parse_ground_truth(inst["Answer"])
        fname = f"{diff}_{iid}.lp"

        samples_lp = []
        for si in range(2):
            raw_text = output.outputs[si].text
            lp_text = extract_lp_content(raw_text)
            lp_path = sample_dirs[si] / fname
            lp_path.write_text(lp_text, encoding="utf-8")
            samples_lp.append(lp_path)

        best_class = None
        for si, lp_path in enumerate(samples_lp):
            record = classify_result(lp_path, gt, wrapper)
            if record["classification"] == "pass":
                best_class = "pass"
                break
            if best_class is None:
                best_class = record["classification"]

        per_instance.append({
            "instance_id": iid,
            "difficulty": diff,
            "classification": best_class,
        })

    total = len(per_instance)
    pass_count = sum(1 for r in per_instance if r["classification"] == "pass")
    easy = [r for r in per_instance if r["difficulty"] == "EasyLP"]
    complex_ = [r for r in per_instance if r["difficulty"] == "ComplexLP"]
    easy_pass = sum(1 for r in easy if r["classification"] == "pass")
    complex_pass = sum(1 for r in complex_ if r["classification"] == "pass")

    seed_result = {
        "seed": seed,
        "total": total,
        "pass_count": pass_count,
        "pass_at_1": pass_count / total if total > 0 else 0,
        "EasyLP": {
            "total": len(easy),
            "pass_count": easy_pass,
            "pass_at_1": easy_pass / len(easy) if easy else 0,
        },
        "ComplexLP": {
            "total": len(complex_),
            "pass_count": complex_pass,
            "pass_at_1": complex_pass / len(complex_) if complex_ else 0,
        },
        "status_distribution": dict(Counter(r["classification"] for r in per_instance)),
    }

    print(f"Seed {seed}: pass@1={seed_result['pass_at_1']:.4f} ({pass_count}/{total})")
    print(f"  EasyLP:    {seed_result['EasyLP']['pass_at_1']:.4f} ({easy_pass}/{len(easy)})")
    print(f"  ComplexLP: {seed_result['ComplexLP']['pass_at_1']:.4f} ({complex_pass}/{len(complex_)})")

    detail_path = seed_dir / "per_instance.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_instance, f, indent=2, ensure_ascii=False)

    return seed_result


def main():
    parser = argparse.ArgumentParser(description="Best-of-2 inference scaling control")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/best_of_2")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MAMO instances...")
    instances = load_mamo()
    if args.limit:
        instances = instances[:args.limit]
    print(f"Loaded {len(instances)} instances")

    print("Building prompts...")
    prompts = [build_generation_prompt(inst["Question"]) for inst in instances]

    print(f"Initializing vLLM: model={args.model}, tp={args.tensor_parallel_size}")
    llm = create_llm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    wrapper = HiGHSWrapper()

    seed_results = []
    for seed in SEEDS:
        sr = run_seed(llm, prompts, instances, seed, output_base, wrapper)
        seed_results.append(sr)

    overall_pass = [s["pass_at_1"] for s in seed_results]
    easy_pass = [s["EasyLP"]["pass_at_1"] for s in seed_results]
    complex_pass = [s["ComplexLP"]["pass_at_1"] for s in seed_results]

    summary = {
        "experiment": "best_of_2_inference_scaling_control",
        "model": args.model,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096, "n": 2},
        "seeds": SEEDS,
        "total_instances": len(instances),
        "overall": {
            "pass_at_1_mean": float(np.mean(overall_pass)),
            "pass_at_1_std": float(np.std(overall_pass)),
            "pass_at_1_per_seed": overall_pass,
        },
        "EasyLP": {
            "total": seed_results[0]["EasyLP"]["total"],
            "pass_at_1_mean": float(np.mean(easy_pass)),
            "pass_at_1_std": float(np.std(easy_pass)),
            "pass_at_1_per_seed": easy_pass,
        },
        "ComplexLP": {
            "total": seed_results[0]["ComplexLP"]["total"],
            "pass_at_1_mean": float(np.mean(complex_pass)),
            "pass_at_1_std": float(np.std(complex_pass)),
            "pass_at_1_per_seed": complex_pass,
        },
        "per_seed_details": seed_results,
        "comparison_with_attempt0": {
            "attempt0_pass_at_1": 0.5805,
            "attempt0_easy_pass_at_1": 0.7117,
            "attempt0_complex_pass_at_1": 0.1754,
        },
    }

    results_path = results_dir / "best_of_2_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("BEST-OF-2 AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Overall pass@1: {np.mean(overall_pass):.4f} +/- {np.std(overall_pass):.4f}")
    print(f"EasyLP pass@1:  {np.mean(easy_pass):.4f} +/- {np.std(easy_pass):.4f}")
    print(f"ComplexLP pass@1: {np.mean(complex_pass):.4f} +/- {np.std(complex_pass):.4f}")
    print(f"\nAttempt-0 greedy: {summary['comparison_with_attempt0']['attempt0_pass_at_1']:.4f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
