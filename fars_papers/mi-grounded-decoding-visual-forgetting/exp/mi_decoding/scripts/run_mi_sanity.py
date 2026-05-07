# Sanity checks for MI decoding: memory, masked-model fluency/accuracy, full MI decoding.
import argparse
import json
import os
import sys
import time
import random

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mi_decoding.models.load_model import load_model, VLAA_THINKER_SYSTEM_PROMPT, VLAA_THINKER_IDS
from mi_decoding.decoding.mi_decoding import generate_mi_decoding, generate_masked_only
from mi_decoding.evaluation.extract_answer import extract_mc_answer


def load_mmstar_items():
    from mi_decoding.data.mmstar import load_mmstar
    return load_mmstar()


def check_memory(model, processor, items, system_prompt, max_new_tokens=512):
    print("\n=== Memory Check (3 items, MI decoding, max_new_tokens=512) ===")
    torch.cuda.reset_peak_memory_stats()
    for i, item in enumerate(items[:3]):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        text = generate_mi_decoding(
            model, processor, item["image"], item["question"],
            max_new_tokens=max_new_tokens, system_prompt=system_prompt,
        )
        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  Item {i}: peak={peak_mb:.0f} MB, time={elapsed:.1f}s, len={len(text)} chars")
        print(f"    Output (first 200 chars): {text[:200]}")
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Overall peak GPU memory: {peak_gb:.2f} GB (A100-80GB limit)")
    return peak_gb


def check_masked_fluency(model, processor, items, system_prompt, n=10):
    print(f"\n=== Masked Model Fluency Check ({n} items) ===")
    for i, item in enumerate(items[:n]):
        text = generate_masked_only(
            model, processor, item["image"], item["question"],
            max_new_tokens=256, system_prompt=system_prompt,
        )
        is_degenerate = len(text.strip()) < 10 or len(set(text[-100:])) < 5
        status = "DEGENERATE" if is_degenerate else "OK"
        print(f"  Item {i} [{status}] len={len(text)}: {text[:150]}...")


def check_masked_accuracy(model, processor, items, system_prompt, n=50):
    print(f"\n=== Masked Model Accuracy Check ({n} items, expect ~25%) ===")
    random.seed(42)
    sample = random.sample(items, min(n, len(items)))
    correct = 0
    for i, item in enumerate(sample):
        text = generate_masked_only(
            model, processor, item["image"], item["question"],
            max_new_tokens=256, system_prompt=system_prompt,
        )
        pred = extract_mc_answer(text)
        gt = item["answer"]
        if pred == gt:
            correct += 1
        if i < 5:
            print(f"  Item {i}: pred={pred} gt={gt} match={pred==gt}")
    acc = correct / len(sample) * 100
    print(f"  Accuracy: {correct}/{len(sample)} = {acc:.1f}% (expected ~25%)")
    return acc


def check_mi_decoding(model, processor, items, system_prompt, n=10):
    print(f"\n=== Full MI Decoding Check ({n} items) ===")
    correct = 0
    for i, item in enumerate(items[:n]):
        t0 = time.time()
        text = generate_mi_decoding(
            model, processor, item["image"], item["question"],
            max_new_tokens=512, system_prompt=system_prompt,
        )
        elapsed = time.time() - t0
        pred = extract_mc_answer(text)
        gt = item["answer"]
        match = pred == gt
        if match:
            correct += 1
        print(f"  Item {i}: pred={pred} gt={gt} match={match} time={elapsed:.1f}s")
        if i < 3:
            print(f"    Output (first 300 chars): {text[:300]}")
    acc = correct / n * 100
    print(f"  Accuracy: {correct}/{n} = {acc:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B")
    args = parser.parse_args()

    system_prompt = None
    if args.model_id in VLAA_THINKER_IDS:
        system_prompt = VLAA_THINKER_SYSTEM_PROMPT

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)
    print(f"Model loaded on {model.device}")

    print("Loading MMStar dataset...")
    items = load_mmstar_items()
    print(f"Loaded {len(items)} items")

    peak_gb = check_memory(model, processor, items, system_prompt)
    check_masked_fluency(model, processor, items, system_prompt, n=10)
    check_masked_accuracy(model, processor, items, system_prompt, n=50)
    check_mi_decoding(model, processor, items, system_prompt, n=10)

    print("\n=== All sanity checks complete ===")
    print(f"Peak GPU memory: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
