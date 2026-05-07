# CLI entry point for offline CA profiling.
# Loads Qwen3-8B, runs forward passes on calibration data, computes CA scores,
# generates masks, and saves results.
# Usage: python -m fcboost.profiling.run_profiling [--num_sequences N] [--max_seq_len L]

import argparse
import os
import sys
import time

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_calibration_texts(num_sequences: int = 4, max_seq_len: int = 4096, tokenizer=None):
    """Load calibration texts from WikiText-2."""
    from datasets import load_dataset

    print(f"Loading WikiText-2 calibration data ({num_sequences} sequences, max {max_seq_len} tokens)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    all_text = " ".join([t for t in dataset["text"] if len(t.strip()) > 100])

    texts = []
    tokens_so_far = 0
    chunk_size_chars = max_seq_len * 5

    for start in range(0, len(all_text), chunk_size_chars):
        if len(texts) >= num_sequences:
            break
        chunk = all_text[start:start + chunk_size_chars]
        if len(chunk.strip()) < 500:
            continue

        if tokenizer is not None:
            toks = tokenizer(chunk, return_tensors="pt", max_length=max_seq_len, truncation=True)
            if toks["input_ids"].shape[1] < 512:
                continue

        texts.append(chunk)

    print(f"  Loaded {len(texts)} calibration sequences")
    return texts


def main():
    parser = argparse.ArgumentParser(description="FCBoost CA Profiling")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_sequences", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--max_sample_tokens", type=int, default=256)
    parser.add_argument("--top_f", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="fcboost/masks")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Quick run with 1 sequence, 512 tokens")
    args = parser.parse_args()

    if args.sanity_check:
        args.num_sequences = 1
        args.max_seq_len = 512
        args.max_sample_tokens = 32
        print("=== SANITY CHECK MODE ===")

    print("=" * 80)
    print("FCBoost CA Profiling")
    print(f"  Model: {args.model}")
    print(f"  Sequences: {args.num_sequences}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  TopK: {args.topk}")
    print(f"  Max sample tokens: {args.max_sample_tokens}")
    print(f"  Top-F (RoPE pairs): {args.top_f}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    calibration_texts = load_calibration_texts(
        num_sequences=args.num_sequences,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
    )

    if not calibration_texts:
        print("ERROR: No valid calibration sequences found")
        sys.exit(1)

    print(f"\nLoading model {args.model} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded. Config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} QH, {model.config.num_key_value_heads} KVH, "
          f"head_dim={model.config.head_dim}")

    from fcboost.profiling.ca_profiler import CAProfiler
    from fcboost.profiling.mask_generator import (
        aggregate_ca_for_gqa,
        generate_masks,
        save_profiling_results,
    )

    profiler = CAProfiler(
        model_name=args.model,
        topk=args.topk,
        max_sample_tokens=args.max_sample_tokens,
    )

    start_time = time.time()
    ca_scores_qh = profiler.profile(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        max_seq_len=args.max_seq_len,
    )
    elapsed = time.time() - start_time
    print(f"\nProfiling took {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("\nAggregating CA scores for GQA...")
    kv_ca_scores = aggregate_ca_for_gqa(
        ca_scores_qh,
        num_kv_heads=model.config.num_key_value_heads,
    )

    print(f"\nGenerating masks (top-{args.top_f} RoPE pairs)...")
    masks = generate_masks(
        kv_ca_scores,
        top_f=args.top_f,
        head_dim=model.config.head_dim,
    )

    save_profiling_results(
        ca_scores_qh=ca_scores_qh,
        kv_ca_scores=kv_ca_scores,
        masks=masks,
        output_dir=args.output_dir,
        top_f=args.top_f,
    )

    print(f"\nDone! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
