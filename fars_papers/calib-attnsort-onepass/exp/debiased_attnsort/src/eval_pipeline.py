# Unified evaluation pipeline for SynthWiki baselines and attention sorting.
# Supports: no_sorting, random_reorder, attnsort_k1 (uncalibrated one-pass sorting).
# attnsort_k1 uses a two-phase approach: eager model for attention extraction,
# then flash-attention model for answer generation from reordered context.
# Usage:
#   python eval_pipeline.py --mode no_sorting --seed 42 --num_examples 200
#   python eval_pipeline.py --mode random_reorder --seed 42 --num_examples 200
#   python eval_pipeline.py --mode attnsort_k1 --seed 42 --num_examples 200

import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ_ROOT, "synthwiki"))

from synthwiki_utils import genJunkContext, insertIntoJunk, checkCorrectness
from llama_utils import generateAttnPrompt

MODEL_ID = "togethercomputer/LLaMA-2-7B-32K-Instruct"
JUNK_SIZE = 28000
PROMPT_TYPE = "together_instruct"
MAX_NEW_TOKENS = 100
SUBSET_SELECTION_SEED = 0
SEEDS = [42, 123, 456]
DATA_PATH = os.path.join(PROJ_ROOT, "synthwiki", "data", "madlibs", "madlibs1.csv")
OUTPUT_BASE = os.path.join(PROJ_ROOT, "debiased_attnsort", "outputs")
RESULTS_BASE = os.path.join(PROJ_ROOT, "debiased_attnsort", "results")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_sorting", "random_reorder", "attnsort_k1", "attnsort_k5", "debiased_k1"], required=True)
    parser.add_argument("--seed", type=int, required=True, help="Random seed for distractor sampling/shuffling")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--sanity_check", action="store_true", help="Run only 10 examples for debugging")
    return parser.parse_args()


def load_data(num_examples):
    raw = pd.read_csv(DATA_PATH)
    rng = np.random.RandomState(SUBSET_SELECTION_SEED)
    indices = rng.choice(len(raw), size=min(num_examples, len(raw)), replace=False)
    indices.sort()
    subset = raw.iloc[indices].reset_index(drop=True)
    all_contexts = np.unique(raw["context"].values)
    return subset, all_contexts


def prepare_context(context, all_contexts, tokenizer, reshuffle=False):
    """Build distractor context and insert gold doc.

    Random state must be set by the caller before the loop (once per seed).
    If reshuffle=True, documents are reshuffled after insertion (random_reorder mode).
    """
    junk_contexts = [c for c in all_contexts if c != context]
    junk_docs = genJunkContext(junk_contexts, limit=JUNK_SIZE, tokenizer=tokenizer)
    random.shuffle(junk_docs)
    supp_docs, pos_to_insert = insertIntoJunk(junk_docs, context, "random")

    if reshuffle:
        gold_doc = supp_docs[pos_to_insert]
        random.shuffle(supp_docs)
        pos_to_insert = supp_docs.index(gold_doc)

    return supp_docs, pos_to_insert


def generate_answer(model, tokenizer, docs, question):
    prompt, doc_idx_range, prompt_len = generateAttnPrompt(
        docs=docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE
    )
    prompt = {k: v.to(model.device) for k, v in prompt.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_attentions=False,
            return_dict_in_generate=False,
        )

    answer_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return answer_text, prompt_len, len(docs)


def sort_docs_by_attention(docs, attention_scores, gold_pos, best_doc_last=True):
    score_order = np.argsort(-attention_scores)
    sorted_docs = [docs[i] for i in score_order]
    new_gold_pos = int(np.where(score_order == gold_pos)[0][0])
    if best_doc_last:
        sorted_docs = sorted_docs[::-1]
        new_gold_pos = len(sorted_docs) - 1 - new_gold_pos
    return sorted_docs, new_gold_pos, score_order


def run_attnsort_k1(args):
    from attention_utils import extract_first_token_attention

    load_dotenv(os.path.join(PROJ_ROOT, ".env"))
    import wandb

    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "calib-attnsort-onepass")
    run_name = f"attnsort_k1_seed{args.seed}_n{args.num_examples}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "mode": "attnsort_k1",
            "seed": args.seed,
            "num_examples": args.num_examples,
            "model": MODEL_ID,
            "junk_size": JUNK_SIZE,
            "prompt_type": PROMPT_TYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prefill_passes_per_query": 2,
        },
    )

    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples, {len(all_contexts)} unique contexts")

    out_dir = os.path.join(OUTPUT_BASE, "attnsort_k1_llama2_32k_instruct", f"seed_{args.seed}")
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
    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context = row["context"]
        question = row["question"]
        true_answer = row["answer"]

        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)

        seed_info = {
            "idx": idx,
            "question": question,
            "gold_doc_position": int(gold_pos),
            "total_docs": len(supp_docs),
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

        sorted_docs, sorted_gold_pos, sort_order = sort_docs_by_attention(
            supp_docs, attn_scores, gold_pos, best_doc_last=True
        )

        t1 = time.time()
        model_answer, gen_prompt_len, num_docs = generate_answer(
            model, tokenizer, sorted_docs, question
        )
        gen_time = time.time() - t1

        correct = checkCorrectness(model_answer, true_answer)
        correct_count += correct
        running_acc = correct_count / (idx + 1)

        result = {
            "idx": idx,
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "correct": correct,
            "seed": args.seed,
            "gold_doc_position": int(gold_pos),
            "sorted_gold_position": int(sorted_gold_pos),
            "total_docs": num_docs,
            "prompt_tokens": gen_prompt_len,
            "attention_scores": attn_scores.tolist(),
            "attn_extraction_time_s": round(attn_time, 2),
            "generation_time_s": round(gen_time, 2),
        }
        results.append(result)

        wandb.log({
            "example_idx": idx,
            "correct": correct,
            "gold_doc_position": int(gold_pos),
            "sorted_gold_position": int(sorted_gold_pos),
            "running_accuracy": running_acc,
            "generation_time_s": gen_time,
        })

        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->{sorted_gold_pos}/{num_docs} | attn={attn_time:.1f}s gen={gen_time:.1f}s")

    accuracy = correct_count / len(subset)
    print(f"\n=== FINAL: seed={args.seed} mode=attnsort_k1 accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    sorted_gold_positions = [r["sorted_gold_position"] for r in results]
    total_docs_list = [r["total_docs"] for r in results]
    mean_sorted_rank = np.mean(sorted_gold_positions)
    last_quartile_frac = np.mean([
        pos >= (total * 0.75) for pos, total in zip(sorted_gold_positions, total_docs_list)
    ])

    summary = {
        "model": MODEL_ID,
        "benchmark": "SynthWiki",
        "mode": "attnsort_k1",
        "seed": args.seed,
        "num_examples": len(subset),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "prefill_passes_per_query": 2,
        "junk_size": JUNK_SIZE,
        "mean_sorted_gold_position": float(mean_sorted_rank),
        "last_quartile_fraction": float(last_quartile_frac),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_attnsort_k5(args, num_iters=5):
    """Iterative attention sorting with k=5 iterations. Each iteration extracts
    first-token attention, sorts documents, generates an intermediate answer,
    then proceeds to the next iteration. Final answer is the last iteration's."""
    from attention_utils import extract_first_token_attention

    load_dotenv(os.path.join(PROJ_ROOT, ".env"))
    import wandb
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "calib-attnsort-onepass")
    run_name = f"attnsort_k5_seed{args.seed}_n{args.num_examples}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "mode": "attnsort_k5",
            "seed": args.seed,
            "num_examples": args.num_examples,
            "model": MODEL_ID,
            "junk_size": JUNK_SIZE,
            "prompt_type": PROMPT_TYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prefill_passes_per_query": num_iters + 1,
            "num_iters": num_iters,
        },
    )

    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples, {len(all_contexts)} unique contexts")

    out_dir = os.path.join(OUTPUT_BASE, "attnsort_k5_llama2_32k_instruct", f"seed_{args.seed}")
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
    intermediate_correct_counts = [0] * num_iters

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

        t_start = time.time()
        current_docs = list(supp_docs)
        current_gold_pos = gold_pos

        gold_positions_per_iter = []
        attn_scores_per_iter = []
        intermediate_answers = []
        intermediate_correct_list = []

        for iter_idx in range(num_iters):
            prompt, doc_idx_range, prompt_len = generateAttnPrompt(
                docs=current_docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE
            )
            prompt_device = {k: v.to(model.device) for k, v in prompt.items()}

            attn_scores, _ = extract_first_token_attention(
                model, tokenizer, prompt_device, doc_idx_range
            )
            attn_scores_per_iter.append(attn_scores.tolist())

            current_docs, current_gold_pos, _ = sort_docs_by_attention(
                current_docs, attn_scores, current_gold_pos, best_doc_last=True
            )
            gold_positions_per_iter.append(int(current_gold_pos))

            inter_answer, _, _ = generate_answer(model, tokenizer, current_docs, question)
            inter_correct = checkCorrectness(inter_answer, true_answer)
            intermediate_answers.append(inter_answer)
            intermediate_correct_list.append(inter_correct)
            intermediate_correct_counts[iter_idx] += inter_correct

            torch.cuda.empty_cache()

        final_answer = intermediate_answers[-1]
        final_correct = intermediate_correct_list[-1]
        correct_count += final_correct
        total_time = time.time() - t_start

        running_acc = correct_count / (idx + 1)

        result = {
            "idx": idx,
            "question": question,
            "true_answer": true_answer,
            "model_answer": final_answer,
            "correct": final_correct,
            "seed": args.seed,
            "gold_doc_position_original": int(gold_pos),
            "gold_position_after_each_iteration": gold_positions_per_iter,
            "total_docs": num_docs,
            "attention_scores_per_iteration": attn_scores_per_iter,
            "intermediate_answers": intermediate_answers,
            "intermediate_correct": intermediate_correct_list,
            "total_time_s": round(total_time, 2),
        }
        results.append(result)

        log_dict = {
            "example_idx": idx,
            "correct": final_correct,
            "running_accuracy": running_acc,
            "total_time_s": total_time,
        }
        for k_iter in range(num_iters):
            log_dict[f"gold_position_iter_{k_iter+1}"] = gold_positions_per_iter[k_iter]
            log_dict[f"intermediate_correct_iter_{k_iter+1}"] = intermediate_correct_list[k_iter]
        wandb.log(log_dict)

        iter_pos_str = ",".join(str(p) for p in gold_positions_per_iter)
        iter_corr_str = ",".join(str(int(c)) for c in intermediate_correct_list)
        print(f"[{idx+1}/{len(subset)}] correct={final_correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->[{iter_pos_str}]/{num_docs} | "
              f"iter_correct=[{iter_corr_str}] | time={total_time:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    accuracy = correct_count / len(subset)
    print(f"\n=== FINAL: seed={args.seed} mode=attnsort_k5 accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")

    for k_iter in range(num_iters):
        iter_acc = intermediate_correct_counts[k_iter] / len(subset)
        wandb.summary[f"intermediate_accuracy_iter_{k_iter+1}"] = iter_acc
        print(f"  Iteration {k_iter+1} intermediate accuracy: {iter_acc:.4f}")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    gold_positions_all = [r["gold_position_after_each_iteration"] for r in results]
    mean_gold_pos_per_iter = [float(np.mean([g[i] for g in gold_positions_all])) for i in range(num_iters)]
    intermediate_acc_per_iter = [intermediate_correct_counts[i] / len(subset) for i in range(num_iters)]

    summary = {
        "model": MODEL_ID,
        "benchmark": "SynthWiki",
        "mode": "attnsort_k5",
        "seed": args.seed,
        "num_examples": len(subset),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "prefill_passes_per_query": num_iters + 1,
        "num_iters": num_iters,
        "junk_size": JUNK_SIZE,
        "intermediate_accuracy_per_iter": intermediate_acc_per_iter,
        "mean_gold_position_per_iter": mean_gold_pos_per_iter,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_evaluation(args, model, tokenizer):
    load_dotenv(os.path.join(PROJ_ROOT, ".env"))

    import wandb

    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "calib-attnsort-onepass")
    run_name = f"{args.mode}_seed{args.seed}_n{args.num_examples}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "mode": args.mode,
            "seed": args.seed,
            "num_examples": args.num_examples,
            "model": MODEL_ID,
            "junk_size": JUNK_SIZE,
            "prompt_type": PROMPT_TYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )

    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples, {len(all_contexts)} unique contexts")

    out_dir = os.path.join(OUTPUT_BASE, f"{args.mode}_llama2_32k_instruct", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    data_seed_dir = os.path.join(OUTPUT_BASE, "data_seeds", f"seed_{args.seed}")
    os.makedirs(data_seed_dir, exist_ok=True)

    results = []
    correct_count = 0

    random.seed(args.seed)
    np.random.seed(args.seed)

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context = row["context"]
        question = row["question"]
        true_answer = row["answer"]

        reshuffle = args.mode == "random_reorder"
        supp_docs, gold_pos = prepare_context(
            context, all_contexts, tokenizer, reshuffle=reshuffle
        )

        seed_info = {
            "idx": idx,
            "question": question,
            "gold_doc_position": int(gold_pos),
            "total_docs": len(supp_docs),
        }
        with open(os.path.join(data_seed_dir, f"example_{idx}.json"), "w") as f:
            json.dump(seed_info, f)

        t0 = time.time()
        model_answer, prompt_len, num_docs = generate_answer(model, tokenizer, supp_docs, question)
        gen_time = time.time() - t0

        correct = checkCorrectness(model_answer, true_answer)
        correct_count += correct

        result = {
            "idx": idx,
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "correct": correct,
            "seed": args.seed,
            "gold_doc_position": int(gold_pos),
            "total_docs": num_docs,
            "prompt_tokens": prompt_len,
            "generation_time_s": round(gen_time, 2),
        }
        results.append(result)

        running_acc = correct_count / (idx + 1)
        wandb.log({
            "example_idx": idx,
            "correct": correct,
            "gold_doc_position": int(gold_pos),
            "running_accuracy": running_acc,
            "generation_time_s": gen_time,
        })

        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}/{num_docs} | time={gen_time:.1f}s")

    accuracy = correct_count / len(subset)
    print(f"\n=== FINAL: seed={args.seed} mode={args.mode} accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.finish()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model": MODEL_ID,
        "benchmark": "SynthWiki",
        "mode": args.mode,
        "seed": args.seed,
        "num_examples": len(subset),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "prefill_passes_per_query": 1,
        "junk_size": JUNK_SIZE,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()

    if args.mode == "attnsort_k1":
        summary = run_attnsort_k1(args)
        print(f"Done. Summary: {json.dumps(summary, indent=2)}")
        return

    if args.mode == "attnsort_k5":
        summary = run_attnsort_k5(args, num_iters=5)
        print(f"Done. Summary: {json.dumps(summary, indent=2)}")
        return

    if args.mode == "debiased_k1":
        from debiased_sorting import run_debiased_k1
        summary = run_debiased_k1(args)
        print(f"Done. Summary: {json.dumps(summary, indent=2)}")
        return

    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded successfully.")

    summary = run_evaluation(args, model, tokenizer)
    print(f"Done. Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
