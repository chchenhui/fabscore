# Debiased one-pass sorting pipeline with minimal-swap strategy.
# Sort by raw attention (same as k=1), then use debiased scores to identify
# the top-1 document. If debiased top-1 differs from raw top-1, swap ONLY
# that document to the last position, preserving the rest of the k=1 ordering.
# This avoids distractor reordering noise while correcting recency-bias errors.
# Total prefill passes = 2 (same as k=1 Attention Sorting).

import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_ROOT, "synthwiki"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_pipeline import (
    MODEL_ID,
    JUNK_SIZE,
    PROMPT_TYPE,
    MAX_NEW_TOKENS,
    OUTPUT_BASE,
    RESULTS_BASE,
    load_data,
    prepare_context,
    generate_answer,
    sort_docs_by_attention,
)
from llama_utils import generateAttnPrompt
from attention_utils import extract_first_token_attention
from bias_estimation import debias_scores

ALPHA = 0.05
NUM_BINS = 20


def run_debiased_k1(args):
    """Debiased one-pass attention sorting with minimal-swap strategy.

    For each example:
      1. Build prompt with original doc ordering (same seed-determined order as baselines).
      2. Extract per-document attention masses (flash prefill + eager decode).
      3. Sort docs by RAW attention scores (ascending, best-last) -- same as k=1.
      4. Compute debiased scores. If debiased top-1 != raw top-1, swap ONLY the
         debiased top-1 to the last position, preserving the rest of the ordering.
      5. Generate answer and record rankings, bias curve, swap status, correctness.
    """
    load_dotenv(os.path.join(PROJ_ROOT, ".env"))
    import wandb

    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "calib-attnsort-onepass")
    run_name = f"debiased_k1_seed{args.seed}_n{args.num_examples}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "mode": "debiased_k1",
            "seed": args.seed,
            "num_examples": args.num_examples,
            "model": MODEL_ID,
            "junk_size": JUNK_SIZE,
            "prompt_type": PROMPT_TYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prefill_passes_per_query": 2,
            "alpha": ALPHA,
            "num_bins": NUM_BINS,
        },
    )

    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples, {len(all_contexts)} unique contexts")

    out_dir = os.path.join(OUTPUT_BASE, "debiased_k1_llama2_32k_instruct", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    data_seed_dir = os.path.join(OUTPUT_BASE, "data_seeds", f"seed_{args.seed}")
    os.makedirs(data_seed_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded (flash_attention_2).")

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    correct_count = 0
    raw_ranks = []
    debiased_ranks = []

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context = row["context"]
        question = row["question"]
        true_answer = row["answer"]

        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)
        num_docs = len(supp_docs)

        seed_info = {
            "idx": idx,
            "question": question,
            "gold_doc_position": int(gold_pos),
            "total_docs": num_docs,
        }
        with open(os.path.join(data_seed_dir, f"example_{idx}.json"), "w") as f:
            json.dump(seed_info, f)

        t0 = time.time()
        prompt, doc_idx_range, prompt_len = generateAttnPrompt(
            docs=supp_docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE
        )
        prompt_device = {k: v.to(model.device) for k, v in prompt.items()}
        attn_scores, _ = extract_first_token_attention(
            model, tokenizer, prompt_device, doc_idx_range
        )
        attn_time = time.time() - t0

        positions = np.arange(num_docs, dtype=np.float64)
        debiased_scores, bias_curve = debias_scores(attn_scores, positions, alpha=ALPHA, num_bins=NUM_BINS)

        sorted_docs, raw_gold_pos, raw_order = sort_docs_by_attention(
            supp_docs, attn_scores, gold_pos, best_doc_last=True
        )

        raw_top1_orig = int(np.argmax(attn_scores))
        deb_top1_orig = int(np.argmax(debiased_scores))
        did_swap = (raw_top1_orig != deb_top1_orig)

        if did_swap:
            gold_doc = supp_docs[gold_pos]
            deb_doc = supp_docs[deb_top1_orig]
            deb_sorted_idx = sorted_docs.index(deb_doc)
            last_idx = len(sorted_docs) - 1
            sorted_docs[deb_sorted_idx], sorted_docs[last_idx] = sorted_docs[last_idx], sorted_docs[deb_sorted_idx]
            debiased_gold_pos = sorted_docs.index(gold_doc)
        else:
            debiased_gold_pos = raw_gold_pos

        t1 = time.time()
        model_answer, gen_prompt_len, _ = generate_answer(
            model, tokenizer, sorted_docs, question
        )
        gen_time = time.time() - t1

        correct = __import__("synthwiki_utils").checkCorrectness(model_answer, true_answer)
        correct_count += correct
        running_acc = correct_count / (idx + 1)

        raw_ranks.append(int(raw_gold_pos))
        debiased_ranks.append(int(debiased_gold_pos))

        result = {
            "idx": idx,
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "correct": correct,
            "seed": args.seed,
            "gold_doc_position": int(gold_pos),
            "raw_rank": int(raw_gold_pos),
            "debiased_rank": int(debiased_gold_pos),
            "total_docs": num_docs,
            "prompt_tokens": gen_prompt_len,
            "raw_attention_scores": attn_scores.tolist(),
            "debiased_scores": debiased_scores.tolist(),
            "bias_curve": bias_curve.tolist(),
            "did_swap": did_swap,
            "attn_extraction_time_s": round(attn_time, 2),
            "generation_time_s": round(gen_time, 2),
        }
        results.append(result)

        wandb.log({
            "example_idx": idx,
            "correct": correct,
            "running_accuracy": running_acc,
            "gold_doc_position": int(gold_pos),
            "raw_rank": int(raw_gold_pos),
            "debiased_rank": int(debiased_gold_pos),
            "did_swap": int(did_swap),
            "generation_time_s": gen_time,
        })

        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->raw:{raw_gold_pos}/deb:{debiased_gold_pos}/{num_docs} | "
              f"attn={attn_time:.1f}s gen={gen_time:.1f}s")

    accuracy = correct_count / len(subset)
    mean_raw_rank = float(np.mean(raw_ranks))
    mean_debiased_rank = float(np.mean(debiased_ranks))
    improved_frac = float(np.mean([d > r for d, r in zip(debiased_ranks, raw_ranks)]))
    swap_count = sum(1 for r in results if r["did_swap"])

    print(f"\n=== FINAL: seed={args.seed} mode=debiased_k1 accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")
    print(f"  Mean raw rank: {mean_raw_rank:.1f}, Mean debiased rank: {mean_debiased_rank:.1f}")
    print(f"  Debiasing improved gold rank: {improved_frac:.3f}")
    print(f"  Swaps performed: {swap_count}/{len(subset)}")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.summary["mean_raw_rank"] = mean_raw_rank
    wandb.summary["mean_debiased_rank"] = mean_debiased_rank
    wandb.summary["debiasing_improved_frac"] = improved_frac
    wandb.summary["swap_count"] = swap_count
    wandb.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model": MODEL_ID,
        "benchmark": "SynthWiki",
        "mode": "debiased_k1",
        "seed": args.seed,
        "num_examples": len(subset),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "prefill_passes_per_query": 2,
        "junk_size": JUNK_SIZE,
        "alpha": ALPHA,
        "num_bins": NUM_BINS,
        "mean_raw_rank": mean_raw_rank,
        "mean_debiased_rank": mean_debiased_rank,
        "debiasing_improved_frac": improved_frac,
        "swap_count": swap_count,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
