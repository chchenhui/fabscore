"""
Best-of-2 inference with AST-parseability filtering for LLaDA-8B-Instruct on BFCL-v3.
Generates 2 independent diffusion samples per example using different sub-seeds,
then selects the parseable one (if exactly one parses); otherwise selects sample 1.
Reuses model loading and decoding from run_unconstrained.py.
"""
import argparse
import ast
import json
import os
import sys
import time
import random

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
CD4DLLM_ROOT = EXP_ROOT / "CD4dLLM"
sys.path.insert(0, str(CD4DLLM_ROOT))

from bfcl_cfg_diffusion.inference.run_unconstrained import load_model, run_inference


def check_ast_parseable(result_str: str) -> bool:
    cleaned = result_str.strip("`\n ")
    if not cleaned.startswith("["):
        cleaned = "[" + cleaned
    if not cleaned.endswith("]"):
        cleaned = cleaned + "]"
    try:
        ast.parse(cleaned)
        return True
    except (SyntaxError, ValueError):
        return False


def select_best(sample_1: str, sample_2: str):
    p1 = check_ast_parseable(sample_1)
    p2 = check_ast_parseable(sample_2)
    if p1 and not p2:
        return sample_1, 1, p1, p2
    elif p2 and not p1:
        return sample_2, 2, p1, p2
    else:
        return sample_1, 1, p1, p2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--data_path", type=str, default=str(EXP_ROOT / "bfcl_cfg_diffusion" / "data" / "bfcl_nonlive_300.json"))
    args = parser.parse_args()

    output_dir = EXP_ROOT / "bfcl_cfg_diffusion" / "results" / "best_of_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{args.seed}.jsonl"

    with open(args.data_path) as f:
        examples = json.load(f)

    if args.limit > 0:
        examples = examples[:args.limit]

    already_done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                already_done.add(entry["id"])
        print(f"Resuming: {len(already_done)} already done")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    print(f"Model loaded. Running best-of-2 inference on {len(examples)} examples with seed={args.seed}")

    sub_seed_1 = args.seed * 2
    sub_seed_2 = args.seed * 2 + 1
    print(f"Sub-seeds: {sub_seed_1}, {sub_seed_2}")

    from bfcl_cfg_diffusion.inference.prompt_formatter import build_messages

    times = []
    selection_stats = {"selected_1": 0, "selected_2": 0, "both_parse": 0, "neither_parse": 0, "one_parses": 0}

    for i, ex in enumerate(tqdm(examples, desc=f"Seed {args.seed}")):
        if ex["id"] in already_done:
            continue

        messages = build_messages(ex)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages.pop(0)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        start_time = time.monotonic()

        sample_1, t1 = run_inference(
            model, tokenizer, prompt_text, sub_seed_1,
            max_tokens=args.max_tokens, steps=args.steps,
            block_length=args.block_length, temperature=args.temperature,
        )

        sample_2, t2 = run_inference(
            model, tokenizer, prompt_text, sub_seed_2,
            max_tokens=args.max_tokens, steps=args.steps,
            block_length=args.block_length, temperature=args.temperature,
        )

        wall_time = time.monotonic() - start_time

        selected, selected_idx, p1, p2 = select_best(sample_1, sample_2)

        if p1 and p2:
            selection_stats["both_parse"] += 1
        elif not p1 and not p2:
            selection_stats["neither_parse"] += 1
        else:
            selection_stats["one_parses"] += 1

        if selected_idx == 1:
            selection_stats["selected_1"] += 1
        else:
            selection_stats["selected_2"] += 1

        times.append(wall_time)

        result = {
            "id": ex["id"],
            "category": ex["category"],
            "result": selected,
            "sample_1": sample_1,
            "sample_2": sample_2,
            "sample_1_parseable": p1,
            "sample_2_parseable": p2,
            "selected_sample": selected_idx,
            "wall_time": wall_time,
            "seed": args.seed,
        }
        with open(output_path, "a") as f:
            print(json.dumps(result), file=f, flush=True)

        if (i + 1) % 10 == 0:
            avg_time = sum(times[-10:]) / min(10, len(times[-10:]))
            done = len(already_done) + len(times)
            print(f"  [{done}/{len(examples)}] Last 10 avg time: {avg_time:.2f}s | "
                  f"Stats: {selection_stats}")

    print(f"\nDone. Total examples: {len(examples)}, Mean time: {sum(times)/max(len(times),1):.2f}s")
    print(f"Selection stats: {selection_stats}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
