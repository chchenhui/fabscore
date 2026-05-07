"""Attempt-0 generation: run Qwen2.5-7B-Instruct on all MAMO instances to produce .lp files.

Usage:
    python -m dualrayrank.scripts.run_attempt0 [--model MODEL] [--output-dir DIR] [--limit N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.prompts.generation_prompt import build_generation_prompt, get_system_prompt
from dualrayrank.inference.vllm_runner import create_llm, generate_batch, extract_lp_content, DEFAULT_SAMPLING_PARAMS


def main():
    parser = argparse.ArgumentParser(description="Run attempt-0 LP generation on MAMO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances (for debugging)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MAMO instances...")
    instances = load_mamo()
    if args.limit:
        instances = instances[:args.limit]
    print(f"Loaded {len(instances)} instances")

    print(f"Building prompts...")
    prompts = []
    for inst in instances:
        prompt_text = build_generation_prompt(inst["Question"])
        prompts.append(prompt_text)

    print(f"Initializing vLLM with model={args.model}, tp={args.tensor_parallel_size}")
    llm = create_llm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    print(f"Running inference on {len(prompts)} prompts...")
    raw_outputs = generate_batch(llm, prompts, DEFAULT_SAMPLING_PARAMS)
    print(f"Inference complete")

    log_path = output_dir / "generation_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as log_f:
        for inst, raw_output in zip(instances, raw_outputs):
            instance_id = inst["id"]
            difficulty = inst["difficulty"]

            lp_content = extract_lp_content(raw_output)

            lp_path = output_dir / f"{difficulty}_{instance_id}.lp"
            with open(lp_path, "w", encoding="utf-8") as lp_f:
                lp_f.write(lp_content)

            log_entry = {
                "instance_id": instance_id,
                "difficulty": difficulty,
                "raw_output": raw_output,
                "extracted_lp_path": str(lp_path),
            }
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(instances)} .lp files to {output_dir}")
    print(f"Generation log: {log_path}")


if __name__ == "__main__":
    main()
