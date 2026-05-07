"""Wall-clock timing calibration for compute-matched best-of-k.
Times Dream diffusion and Qwen greedy on 50-instance calibration subsets (both tasks).
Computes k = floor(dream_median / qwen_median), clipped to [1, 64].
Also computes k using p75 for sensitivity analysis.

Both models are timed on the same GPU with bfloat16, using torch.cuda.synchronize()
for accurate GPU timing. Models are loaded/unloaded sequentially to avoid memory contention.
"""

import json
import os
import time
import math
import argparse
import gc
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARMUP_COUNT = 3


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_template(path):
    with open(path) as f:
        return f.read()


def build_prompts(template, instances):
    prompts = []
    for inst in instances:
        question = inst["question"].strip()
        full_prompt = template + "\n" + question + "\nOutput:"
        prompts.append(full_prompt)
    return prompts


def time_dream(model, tokenizer, prompts, max_new_tokens=64):
    times = []
    all_prompts = prompts

    for i, prompt in enumerate(all_prompts):
        inputs = tokenizer(
            [prompt],
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model.diffusion_generate(
                inputs=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                diffusion_steps=max_new_tokens,
                temperature=0,
                top_p=1,
                alg="entropy",
                alg_temp=0,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i < WARMUP_COUNT:
            print(f"  [Dream warmup {i+1}/{WARMUP_COUNT}] {elapsed:.4f}s")
        else:
            times.append(elapsed)
            if (i - WARMUP_COUNT) % 10 == 0:
                print(f"  [Dream {i - WARMUP_COUNT + 1}/{len(all_prompts) - WARMUP_COUNT}] {elapsed:.4f}s")

    return times


def time_qwen(model, tokenizer, prompts, max_new_tokens=64):
    times = []
    all_prompts = prompts

    for i, prompt in enumerate(all_prompts):
        inputs = tokenizer(
            [prompt],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i < WARMUP_COUNT:
            print(f"  [Qwen warmup {i+1}/{WARMUP_COUNT}] {elapsed:.4f}s")
        else:
            times.append(elapsed)
            if (i - WARMUP_COUNT) % 10 == 0:
                print(f"  [Qwen {i - WARMUP_COUNT + 1}/{len(all_prompts) - WARMUP_COUNT}] {elapsed:.4f}s")

    return times


def compute_k(dream_times, qwen_times):
    dream_median = float(np.median(dream_times))
    qwen_median = float(np.median(qwen_times))
    dream_p75 = float(np.percentile(dream_times, 75))
    qwen_p75 = float(np.percentile(qwen_times, 75))

    k_median = max(1, min(64, math.floor(dream_median / qwen_median)))
    k_p75 = max(1, min(64, math.floor(dream_p75 / qwen_p75)))

    return {
        "dream_median": round(dream_median, 6),
        "qwen_median": round(qwen_median, 6),
        "k_median": k_median,
        "dream_p75": round(dream_p75, 6),
        "qwen_p75": round(qwen_p75, 6),
        "k_p75": k_p75,
        "dream_times": [round(t, 6) for t in dream_times],
        "qwen_times": [round(t, 6) for t in qwen_times],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "timing", "calibration_results.json"))
    args = parser.parse_args()

    task_configs = {
        "countdown": {
            "cal_file": os.path.join(BASE_DIR, "data", "countdown_cal.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "countdown_8shot.txt"),
            "max_new_tokens": 64,
        },
        "sudoku": {
            "cal_file": os.path.join(BASE_DIR, "data", "sudoku_cal.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "sudoku_8shot.txt"),
            "max_new_tokens": 64,
        },
    }

    results = {}

    print("=" * 60)
    print("Loading Dream-org/Dream-v0-Base-7B ...")
    dream_model = AutoModel.from_pretrained(
        "Dream-org/Dream-v0-Base-7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    dream_tokenizer = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Base-7B",
        trust_remote_code=True,
    )
    print(f"Dream model loaded on {next(dream_model.parameters()).device}")

    dream_times = {}
    for task_name, cfg in task_configs.items():
        instances = load_jsonl(cfg["cal_file"])
        template = load_template(cfg["template_file"])
        prompts = build_prompts(template, instances)
        warmup_prompts = prompts[:WARMUP_COUNT]
        all_prompts = warmup_prompts + prompts

        print(f"\n--- Timing Dream on {task_name} ({len(prompts)} cal + {WARMUP_COUNT} warmup) ---")
        dream_times[task_name] = time_dream(
            dream_model, dream_tokenizer, all_prompts, cfg["max_new_tokens"]
        )
        print(f"  Dream {task_name}: median={np.median(dream_times[task_name]):.4f}s")

    del dream_model, dream_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDream model unloaded, GPU memory freed.")

    print("=" * 60)
    print("Loading Qwen/Qwen2.5-7B ...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    print(f"Qwen model loaded on {next(qwen_model.parameters()).device}")

    qwen_times = {}
    for task_name, cfg in task_configs.items():
        instances = load_jsonl(cfg["cal_file"])
        template = load_template(cfg["template_file"])
        prompts = build_prompts(template, instances)
        warmup_prompts = prompts[:WARMUP_COUNT]
        all_prompts = warmup_prompts + prompts

        print(f"\n--- Timing Qwen on {task_name} ({len(prompts)} cal + {WARMUP_COUNT} warmup) ---")
        qwen_times[task_name] = time_qwen(
            qwen_model, qwen_tokenizer, all_prompts, cfg["max_new_tokens"]
        )
        print(f"  Qwen {task_name}: median={np.median(qwen_times[task_name]):.4f}s")

    del qwen_model, qwen_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\nQwen model unloaded.")

    print("=" * 60)
    print("Computing k values ...")
    for task_name in task_configs:
        results[task_name] = compute_k(dream_times[task_name], qwen_times[task_name])
        print(f"\n{task_name}:")
        print(f"  Dream median: {results[task_name]['dream_median']:.4f}s")
        print(f"  Qwen median:  {results[task_name]['qwen_median']:.4f}s")
        print(f"  k (median):   {results[task_name]['k_median']}")
        print(f"  Dream p75:    {results[task_name]['dream_p75']:.4f}s")
        print(f"  Qwen p75:     {results[task_name]['qwen_p75']:.4f}s")
        print(f"  k (p75):      {results[task_name]['k_p75']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
