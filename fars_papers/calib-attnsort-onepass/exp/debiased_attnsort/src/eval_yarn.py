# Evaluation pipeline for NousResearch/Yarn-Llama-2-7b-64k on SynthWiki.
# Supports modes: no_sorting, attnsort_k1, attnsort_k5, debiased_k1.
# Adapted from eval_pipeline.py -- uses "wizard" prompt type, torch.load
# monkey-patch for weights_only=False, and trust_remote_code=True.

import argparse
import functools
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthwiki_utils import genJunkContext, insertIntoJunk, checkCorrectness
from llama_utils import generateAttnPrompt
from bias_estimation import debias_scores

def extract_first_token_attention_yarn(model, tokenizer, input_ids, doc_idx_ranges):
    """Extract per-document attention scores from the first generated token.

    The YaRN custom model uses flash_attn internally which does not return
    proper attention weights during decode with KV cache. We monkey-patch
    each layer's self_attn.forward to compute attention weights manually
    (Q*K^T / norm_factor -> softmax) during a single decode step.

    Two-step approach:
      1. Prefill with output_attentions=False (builds KV cache via flash_attn).
      2. Decode one token with patched forward that computes manual attention.
    """
    with torch.no_grad():
        prefill_out = model(
            **input_ids,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = prefill_out.past_key_values
        next_token_logits = prefill_out.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

    captured_attn_weights = []
    orig_forwards = []

    def make_attn_capture_forward(orig_fn, attn_module):
        def patched_forward(hidden_states, attention_mask=None, position_ids=None,
                            past_key_value=None, output_attentions=False,
                            use_cache=False, is_padded_inputs=False):
            bsz, q_len, _ = hidden_states.size()
            q = attn_module.q_proj(hidden_states)
            k = attn_module.k_proj(hidden_states)

            q = q.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim)
            k = k.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim)

            has_past = past_key_value is not None
            past_len = past_key_value[1] if has_past else 0
            q, k = attn_module.rotary_emb(q, k, past_len)

            if has_past:
                past_kv_tensor = past_key_value[0]
                past_k = past_kv_tensor[:, :past_len, 0, :, :]
                if attn_module.num_key_value_groups > 1:
                    past_k = past_k.repeat_interleave(attn_module.num_key_value_groups, dim=2)
                full_k = torch.cat([past_k, k], dim=1)
            else:
                full_k = k
                if attn_module.num_key_value_groups > 1:
                    full_k = full_k.repeat_interleave(attn_module.num_key_value_groups, dim=2)

            scores = torch.einsum('bqhd,bkhd->bhqk', q, full_k) * (1.0 / attn_module.norm_factor)
            attn_w = torch.softmax(scores.float(), dim=-1)
            captured_attn_weights.append(attn_w.detach().half())

            return orig_fn(hidden_states, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_value=past_key_value,
                           output_attentions=output_attentions, use_cache=use_cache,
                           is_padded_inputs=is_padded_inputs)
        return patched_forward

    for layer in model.model.layers:
        orig_fn = layer.self_attn.forward
        orig_forwards.append(orig_fn)
        layer.self_attn.forward = make_attn_capture_forward(orig_fn, layer.self_attn)

    try:
        with torch.no_grad():
            model(
                input_ids=next_token_id,
                past_key_values=past_kv,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
    finally:
        for layer, orig_fn in zip(model.model.layers, orig_forwards):
            layer.self_attn.forward = orig_fn

    num_docs = len(doc_idx_ranges)
    attention_masses = np.zeros(num_docs, dtype=np.float64)

    for layer_attn in captured_attn_weights:
        attn = layer_attn[0, :, 0, :]
        for i, (start, end) in enumerate(doc_idx_ranges):
            attention_masses[i] += attn[:, start:end].sum().cpu().item()

    return attention_masses, doc_idx_ranges


MODEL_ID = "NousResearch/Yarn-Llama-2-7b-64k"
JUNK_SIZE = 28000
PROMPT_TYPE = "wizard"
MAX_NEW_TOKENS = 100
SUBSET_SELECTION_SEED = 0
SEEDS = [42, 123, 456]
DATA_PATH = os.path.join(PROJ_ROOT, "synthwiki", "data", "madlibs", "madlibs1.csv")
OUTPUT_BASE = os.path.join(PROJ_ROOT, "debiased_attnsort", "outputs")
RESULTS_BASE = os.path.join(PROJ_ROOT, "debiased_attnsort", "results")

ALPHA = 0.005
NUM_BINS = 40
AGGREGATION = "mean"
DEBIAS_MODE = "divisive"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_sorting", "attnsort_k1", "attnsort_k5", "debiased_k1"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--sanity_check", action="store_true")
    return parser.parse_args()


def load_data(num_examples):
    raw = pd.read_csv(DATA_PATH)
    rng = np.random.RandomState(SUBSET_SELECTION_SEED)
    indices = rng.choice(len(raw), size=min(num_examples, len(raw)), replace=False)
    indices.sort()
    subset = raw.iloc[indices].reset_index(drop=True)
    all_contexts = np.unique(raw["context"].values)
    return subset, all_contexts


def prepare_context(context, all_contexts, tokenizer):
    junk_contexts = [c for c in all_contexts if c != context]
    junk_docs = genJunkContext(junk_contexts, limit=JUNK_SIZE, tokenizer=tokenizer)
    random.shuffle(junk_docs)
    supp_docs, pos_to_insert = insertIntoJunk(junk_docs, context, "random")
    return supp_docs, pos_to_insert


def load_yarn_model():
    _orig = torch.load
    @functools.wraps(_orig)
    def _patched(*a, **kw):
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _patched
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    finally:
        torch.load = _orig
    model.eval()
    return model, tokenizer


def greedy_generate(model, input_ids, attention_mask, max_new_tokens, eos_token_id):
    generated_ids = []
    past_key_values = None
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1)
            generated_ids.append(next_token_id.item())
            if next_token_id.item() == eos_token_id:
                break
            past_key_values = outputs.past_key_values
            cur_input_ids = next_token_id.unsqueeze(0)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(1, 1, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype)], dim=1)
    return generated_ids


def generate_answer(model, tokenizer, docs, question):
    prompt, doc_idx_range, prompt_len = generateAttnPrompt(
        docs=docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE
    )
    prompt = {k: v.to(model.device) for k, v in prompt.items()}
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    generated_ids = greedy_generate(
        model, prompt["input_ids"], prompt["attention_mask"], MAX_NEW_TOKENS, eos_id)
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer_text, prompt_len, len(docs)


def sort_docs_by_attention(docs, attention_scores, gold_pos, best_doc_last=True):
    score_order = np.argsort(-attention_scores)
    sorted_docs = [docs[i] for i in score_order]
    new_gold_pos = int(np.where(score_order == gold_pos)[0][0])
    if best_doc_last:
        sorted_docs = sorted_docs[::-1]
        new_gold_pos = len(sorted_docs) - 1 - new_gold_pos
    return sorted_docs, new_gold_pos, score_order


def init_wandb(mode, seed, num_examples, extra_config=None):
    load_dotenv(os.path.join(PROJ_ROOT, ".env"))
    import wandb
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    wandb_project = os.environ.get("WANDB_PROJECT", "calib-attnsort-onepass")
    run_name = f"yarn_{mode}_seed{seed}_n{num_examples}"
    config = {
        "mode": mode,
        "seed": seed,
        "num_examples": num_examples,
        "model": MODEL_ID,
        "junk_size": JUNK_SIZE,
        "prompt_type": PROMPT_TYPE,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if extra_config:
        config.update(extra_config)
    wandb.init(project=wandb_project, name=run_name, mode=wandb_mode, config=config)
    return wandb


def run_no_sorting(args, model, tokenizer):
    wandb = init_wandb("no_sorting", args.seed, args.num_examples, {"prefill_passes_per_query": 1})
    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples")

    out_dir = os.path.join(OUTPUT_BASE, "no_sorting_yarn_llama2_64k", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    correct_count = 0

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context, question, true_answer = row["context"], row["question"], row["answer"]
        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)

        t0 = time.time()
        model_answer, prompt_len, num_docs = generate_answer(model, tokenizer, supp_docs, question)
        gen_time = time.time() - t0

        correct = checkCorrectness(model_answer, true_answer)
        correct_count += correct
        running_acc = correct_count / (idx + 1)

        results.append({
            "idx": idx, "question": question, "true_answer": true_answer,
            "model_answer": model_answer, "correct": correct, "seed": args.seed,
            "gold_doc_position": int(gold_pos), "total_docs": num_docs,
            "prompt_tokens": prompt_len, "generation_time_s": round(gen_time, 2),
        })
        wandb.log({"example_idx": idx, "correct": correct, "gold_doc_position": int(gold_pos),
                    "running_accuracy": running_acc, "generation_time_s": gen_time})
        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}/{num_docs} | time={gen_time:.1f}s")

    accuracy = correct_count / len(subset)
    print(f"\n=== FINAL: seed={args.seed} mode=no_sorting accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.finish()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary = {"model": MODEL_ID, "benchmark": "SynthWiki", "mode": "no_sorting",
               "seed": args.seed, "num_examples": len(subset), "accuracy": accuracy,
               "correct_count": correct_count, "prefill_passes_per_query": 1, "junk_size": JUNK_SIZE}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_attnsort_k1(args, model, tokenizer):
    wandb = init_wandb("attnsort_k1", args.seed, args.num_examples, {"prefill_passes_per_query": 2})
    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples")

    out_dir = os.path.join(OUTPUT_BASE, "attnsort_k1_yarn_llama2_64k", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    correct_count = 0

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context, question, true_answer = row["context"], row["question"], row["answer"]
        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)

        t0 = time.time()
        prompt, doc_idx_range, prompt_len = generateAttnPrompt(
            docs=supp_docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE)
        prompt_device = {k: v.to(model.device) for k, v in prompt.items()}
        attn_scores, _ = extract_first_token_attention_yarn(model, tokenizer, prompt_device, doc_idx_range)
        attn_time = time.time() - t0

        sorted_docs, sorted_gold_pos, _ = sort_docs_by_attention(supp_docs, attn_scores, gold_pos, best_doc_last=True)

        t1 = time.time()
        model_answer, gen_prompt_len, num_docs = generate_answer(model, tokenizer, sorted_docs, question)
        gen_time = time.time() - t1

        correct = checkCorrectness(model_answer, true_answer)
        correct_count += correct
        running_acc = correct_count / (idx + 1)

        results.append({
            "idx": idx, "question": question, "true_answer": true_answer,
            "model_answer": model_answer, "correct": correct, "seed": args.seed,
            "gold_doc_position": int(gold_pos), "sorted_gold_position": int(sorted_gold_pos),
            "total_docs": num_docs, "prompt_tokens": gen_prompt_len,
            "attention_scores": attn_scores.tolist(),
            "attn_extraction_time_s": round(attn_time, 2), "generation_time_s": round(gen_time, 2),
        })
        wandb.log({"example_idx": idx, "correct": correct, "gold_doc_position": int(gold_pos),
                    "sorted_gold_position": int(sorted_gold_pos), "running_accuracy": running_acc,
                    "generation_time_s": gen_time})
        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->{sorted_gold_pos}/{num_docs} | attn={attn_time:.1f}s gen={gen_time:.1f}s")

        torch.cuda.empty_cache()

    accuracy = correct_count / len(subset)
    print(f"\n=== FINAL: seed={args.seed} mode=attnsort_k1 accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.finish()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary = {"model": MODEL_ID, "benchmark": "SynthWiki", "mode": "attnsort_k1",
               "seed": args.seed, "num_examples": len(subset), "accuracy": accuracy,
               "correct_count": correct_count, "prefill_passes_per_query": 2, "junk_size": JUNK_SIZE}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_attnsort_k5(args, model, tokenizer, num_iters=5):
    wandb = init_wandb("attnsort_k5", args.seed, args.num_examples,
                        {"prefill_passes_per_query": num_iters + 1, "num_iters": num_iters})
    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples")

    out_dir = os.path.join(OUTPUT_BASE, "attnsort_k5_yarn_llama2_64k", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    correct_count = 0
    intermediate_correct_counts = [0] * num_iters

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context, question, true_answer = row["context"], row["question"], row["answer"]
        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)
        num_docs = len(supp_docs)

        t_start = time.time()
        current_docs = list(supp_docs)
        current_gold_pos = gold_pos
        gold_positions_per_iter = []
        attn_scores_per_iter = []
        intermediate_answers = []
        intermediate_correct_list = []

        for iter_idx in range(num_iters):
            prompt, doc_idx_range, prompt_len = generateAttnPrompt(
                docs=current_docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE)
            prompt_device = {k: v.to(model.device) for k, v in prompt.items()}
            attn_scores, _ = extract_first_token_attention_yarn(model, tokenizer, prompt_device, doc_idx_range)
            attn_scores_per_iter.append(attn_scores.tolist())

            current_docs, current_gold_pos, _ = sort_docs_by_attention(
                current_docs, attn_scores, current_gold_pos, best_doc_last=True)
            gold_positions_per_iter.append(int(current_gold_pos))

            inter_answer, _, _ = generate_answer(model, tokenizer, current_docs, question)
            inter_correct = checkCorrectness(inter_answer, true_answer)
            intermediate_answers.append(inter_answer)
            intermediate_correct_list.append(inter_correct)
            intermediate_correct_counts[iter_idx] += inter_correct
            torch.cuda.empty_cache()

        final_correct = intermediate_correct_list[-1]
        correct_count += final_correct
        total_time = time.time() - t_start
        running_acc = correct_count / (idx + 1)

        results.append({
            "idx": idx, "question": question, "true_answer": true_answer,
            "model_answer": intermediate_answers[-1], "correct": final_correct,
            "seed": args.seed, "gold_doc_position_original": int(gold_pos),
            "gold_position_after_each_iteration": gold_positions_per_iter,
            "total_docs": num_docs, "intermediate_correct": intermediate_correct_list,
            "total_time_s": round(total_time, 2),
        })

        log_dict = {"example_idx": idx, "correct": final_correct, "running_accuracy": running_acc,
                     "total_time_s": total_time}
        for k_iter in range(num_iters):
            log_dict[f"gold_position_iter_{k_iter+1}"] = gold_positions_per_iter[k_iter]
            log_dict[f"intermediate_correct_iter_{k_iter+1}"] = intermediate_correct_list[k_iter]
        wandb.log(log_dict)

        iter_pos_str = ",".join(str(p) for p in gold_positions_per_iter)
        print(f"[{idx+1}/{len(subset)}] correct={final_correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->[{iter_pos_str}]/{num_docs} | time={total_time:.1f}s")

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

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    intermediate_acc_per_iter = [intermediate_correct_counts[i] / len(subset) for i in range(num_iters)]
    summary = {"model": MODEL_ID, "benchmark": "SynthWiki", "mode": "attnsort_k5",
               "seed": args.seed, "num_examples": len(subset), "accuracy": accuracy,
               "correct_count": correct_count, "prefill_passes_per_query": num_iters + 1,
               "num_iters": num_iters, "junk_size": JUNK_SIZE,
               "intermediate_accuracy_per_iter": intermediate_acc_per_iter}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_debiased_k1(args, model, tokenizer):
    wandb = init_wandb("debiased_k1", args.seed, args.num_examples,
                        {"prefill_passes_per_query": 2, "alpha": ALPHA, "num_bins": NUM_BINS,
                         "aggregation": AGGREGATION, "debias_mode": DEBIAS_MODE,
                         "strategy": "full_sort_by_debiased"})
    n = 10 if args.sanity_check else args.num_examples
    subset, all_contexts = load_data(n)
    print(f"Loaded {len(subset)} examples")

    out_dir = os.path.join(OUTPUT_BASE, "debiased_k1_yarn_llama2_64k", f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    correct_count = 0
    raw_ranks = []
    debiased_ranks = []

    for idx in range(len(subset)):
        row = subset.iloc[idx]
        context, question, true_answer = row["context"], row["question"], row["answer"]
        supp_docs, gold_pos = prepare_context(context, all_contexts, tokenizer)
        num_docs = len(supp_docs)

        t0 = time.time()
        prompt, doc_idx_range, prompt_len = generateAttnPrompt(
            docs=supp_docs, question=question, tokenizer=tokenizer, prompt_type=PROMPT_TYPE)
        prompt_device = {k: v.to(model.device) for k, v in prompt.items()}
        attn_scores, _ = extract_first_token_attention_yarn(model, tokenizer, prompt_device, doc_idx_range)
        attn_time = time.time() - t0

        positions = np.arange(num_docs, dtype=np.float64)
        debiased_scores, bias_curve = debias_scores(
            attn_scores, positions, alpha=ALPHA, num_bins=NUM_BINS,
            aggregation=AGGREGATION, mode=DEBIAS_MODE)

        _, raw_gold_pos, _ = sort_docs_by_attention(
            supp_docs, attn_scores, gold_pos, best_doc_last=True)

        sorted_docs, debiased_gold_pos, _ = sort_docs_by_attention(
            supp_docs, debiased_scores, gold_pos, best_doc_last=True)

        did_swap = (int(np.argmax(attn_scores)) != int(np.argmax(debiased_scores)))

        t1 = time.time()
        model_answer, gen_prompt_len, _ = generate_answer(model, tokenizer, sorted_docs, question)
        gen_time = time.time() - t1

        correct = checkCorrectness(model_answer, true_answer)
        correct_count += correct
        running_acc = correct_count / (idx + 1)

        raw_ranks.append(int(raw_gold_pos))
        debiased_ranks.append(int(debiased_gold_pos))

        results.append({
            "idx": idx, "question": question, "true_answer": true_answer,
            "model_answer": model_answer, "correct": correct, "seed": args.seed,
            "gold_doc_position": int(gold_pos), "raw_rank": int(raw_gold_pos),
            "debiased_rank": int(debiased_gold_pos), "total_docs": num_docs,
            "prompt_tokens": gen_prompt_len, "raw_attention_scores": attn_scores.tolist(),
            "debiased_scores": debiased_scores.tolist(), "bias_curve": bias_curve.tolist(),
            "did_swap": did_swap, "attn_extraction_time_s": round(attn_time, 2),
            "generation_time_s": round(gen_time, 2),
        })
        wandb.log({"example_idx": idx, "correct": correct, "running_accuracy": running_acc,
                    "gold_doc_position": int(gold_pos), "raw_rank": int(raw_gold_pos),
                    "debiased_rank": int(debiased_gold_pos), "did_swap": int(did_swap),
                    "generation_time_s": gen_time})
        print(f"[{idx+1}/{len(subset)}] correct={correct} | running_acc={running_acc:.3f} | "
              f"gold_pos={gold_pos}->raw:{raw_gold_pos}/deb:{debiased_gold_pos}/{num_docs} | "
              f"swap={did_swap} | attn={attn_time:.1f}s gen={gen_time:.1f}s")

        torch.cuda.empty_cache()

    accuracy = correct_count / len(subset)
    swap_count = sum(1 for r in results if r["did_swap"])
    mean_raw_rank = float(np.mean(raw_ranks))
    mean_debiased_rank = float(np.mean(debiased_ranks))

    print(f"\n=== FINAL: seed={args.seed} mode=debiased_k1 accuracy={accuracy:.4f} ({correct_count}/{len(subset)}) ===")
    print(f"  Swaps: {swap_count}/{len(subset)}")

    wandb.summary["final_accuracy"] = accuracy
    wandb.summary["correct_count"] = correct_count
    wandb.summary["total_examples"] = len(subset)
    wandb.summary["swap_count"] = swap_count
    wandb.summary["mean_raw_rank"] = mean_raw_rank
    wandb.summary["mean_debiased_rank"] = mean_debiased_rank
    wandb.finish()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary = {"model": MODEL_ID, "benchmark": "SynthWiki", "mode": "debiased_k1",
               "seed": args.seed, "num_examples": len(subset), "accuracy": accuracy,
               "correct_count": correct_count, "prefill_passes_per_query": 2,
               "junk_size": JUNK_SIZE, "alpha": ALPHA, "num_bins": NUM_BINS,
               "aggregation": AGGREGATION, "debias_mode": DEBIAS_MODE,
               "strategy": "full_sort_by_debiased",
               "swap_count": swap_count, "mean_raw_rank": mean_raw_rank,
               "mean_debiased_rank": mean_debiased_rank}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    import traceback
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    try:
        args = parse_args()

        print(f"Loading YaRN model {MODEL_ID}...")
        model, tokenizer = load_yarn_model()
        print(f"Model loaded. Running mode={args.mode} seed={args.seed}")
        sys.stdout.flush()

        if args.mode == "no_sorting":
            summary = run_no_sorting(args, model, tokenizer)
        elif args.mode == "attnsort_k1":
            summary = run_attnsort_k1(args, model, tokenizer)
        elif args.mode == "attnsort_k5":
            summary = run_attnsort_k5(args, model, tokenizer)
        elif args.mode == "debiased_k1":
            summary = run_debiased_k1(args, model, tokenizer)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        print(f"Done. Summary: {json.dumps(summary, indent=2)}")

        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
