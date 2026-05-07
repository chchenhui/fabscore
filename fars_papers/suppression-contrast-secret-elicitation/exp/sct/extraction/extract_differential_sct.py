# Differential extraction: compares finetuned vs base model at control positions.
# Computes multiple scoring variants from a single pair of forward passes:
#   1. diff_SCT(v) = SCT_ft(v) - SCT_base(v)  [differential suppression]
#   2. mid_diff(v) = log p_mid_ft(v) - log p_mid_base(v)  [mid-layer uplift]
#   3. ft_sct(v) = log p_mid_ft(v) - log p_final_ft(v)  [standard SCT, full vocab]
#   4. ft_ll(v) = log p_mid_ft(v)  [standard logit lens, full vocab]
#   5. combined(v) = mid_diff(v) + lambda * ft_sct(v)  [hybrid]
# All at control positions (0,1 of assistant turn) using full vocabulary.

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
sys.path.insert(0, BENCHMARK_DIR)

from sampling.utils import load_model_and_tokenizer


def find_response_start(tokens, tokenizer):
    start_token_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    positions = [i for i, tid in enumerate(tokens) if tid == start_token_id]
    return positions[1] if len(positions) >= 2 else -1


class LayerHook:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.reps = {}
        self.hooks = []

    def _hook(self, idx):
        def fn(mod, inp, out):
            self.reps[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def __enter__(self):
        for idx in self.layers:
            h = self.model.model.layers[idx].register_forward_hook(self._hook(idx))
            self.hooks.append(h)
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.reps.clear()


def extract_ctrl_logprobs(model, tokens, mid_layer, final_layer, response_start, device):
    layers = [mid_layer, final_layer]
    input_ids = tokens.unsqueeze(0).to(device)
    ctrl_positions = [response_start + i for i in range(2) if response_start + i < len(tokens)]
    n = len(ctrl_positions)

    with LayerHook(model, layers) as hook:
        with torch.no_grad():
            model(input_ids)

        vocab = model.config.vocab_size
        mid_accum = torch.zeros(vocab, device=device)
        final_accum = torch.zeros(vocab, device=device)

        for pos in ctrl_positions:
            mid_normed = model.model.norm(hook.reps[mid_layer][:, pos:pos+1, :])
            fin_normed = model.model.norm(hook.reps[final_layer][:, pos:pos+1, :])
            mid_lp = F.log_softmax(model.lm_head(mid_normed).squeeze(0).squeeze(0).float(), dim=-1)
            fin_lp = F.log_softmax(model.lm_head(fin_normed).squeeze(0).squeeze(0).float(), dim=-1)
            mid_accum += mid_lp
            final_accum += fin_lp

    if n > 0:
        mid_accum /= n
        final_accum /= n

    return mid_accum.cpu(), final_accum.cpu()


def load_pairs(source_path, max_examples=None):
    print(f"Loading pairs from {source_path}...")
    with open(source_path, "r") as f:
        data = json.load(f)
    pairs = []
    for item in data["results"]:
        if "error" in item:
            continue
        pairs.append({
            "user_prompt": item["user_prompt"],
            "model_response": item["model_response"],
            "model_response_index": item.get("model_response_index", 0),
            "response_token_ids": item.get("response_token_ids", []),
        })
    del data
    gc.collect()
    if max_examples:
        pairs = pairs[:max_examples]
    print(f"Loaded {len(pairs)} pairs")
    return pairs


def extract_all(model, tokenizer, pairs, mid_layer, final_layer, device, label=""):
    mid_list = []
    final_list = []
    valid = []

    for pair in tqdm(pairs, desc=f"{label} extraction"):
        messages = [
            {"role": "user", "content": pair["user_prompt"]},
            {"role": "assistant", "content": pair["model_response"]},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False,
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
        rs = find_response_start(tokens, tokenizer)

        if rs == -1:
            mid_list.append(None)
            final_list.append(None)
            valid.append(False)
            continue

        try:
            mid_lp, final_lp = extract_ctrl_logprobs(model, tokens, mid_layer, final_layer, rs, device)
            mid_list.append(mid_lp)
            final_list.append(final_lp)
            valid.append(True)
        except Exception as e:
            print(f"  Error: {e}")
            mid_list.append(None)
            final_list.append(None)
            valid.append(False)

    return mid_list, final_list, valid


def score_and_rank(scores_tensor, output_set, top_k, tokenizer=None):
    mask = torch.ones(scores_tensor.shape[0], dtype=torch.bool)
    for tid in output_set:
        if tid < mask.shape[0]:
            mask[tid] = False
    masked = scores_tensor.clone()
    masked[~mask] = -float('inf')
    vals, ids = torch.topk(masked, top_k)
    ranked = []
    for k in range(min(top_k, len(vals))):
        tid = ids[k].item()
        entry = {"token_id": tid, "score": vals[k].item()}
        if tokenizer:
            entry["token"] = tokenizer.decode([tid])
        ranked.append(entry)
    return ranked


def compute_all_scores(
    ft_mid_list, ft_final_list, base_mid_list, base_final_list,
    valid_ft, valid_base, pairs, top_k, tokenizer, combo_lambda=1.0,
):
    results = []
    for idx in range(len(pairs)):
        pair = pairs[idx]
        if not valid_ft[idx] or not valid_base[idx]:
            results.append({
                "user_prompt": pair["user_prompt"],
                "model_response": pair["model_response"],
                "model_response_index": pair["model_response_index"],
                "error": "Extraction failed",
                "ranked_tokens": [],
            })
            continue

        ft_mid = ft_mid_list[idx]
        ft_final = ft_final_list[idx]
        base_mid = base_mid_list[idx]
        base_final = base_final_list[idx]
        output_set = set(pair["response_token_ids"])

        ft_sct = ft_mid - ft_final
        base_sct = base_mid - base_final
        diff_sct = ft_sct - base_sct
        mid_diff = ft_mid - base_mid
        combined = mid_diff + combo_lambda * ft_sct

        diff_ranked = score_and_rank(diff_sct, output_set, top_k, tokenizer)
        mid_diff_ranked = score_and_rank(mid_diff, output_set, top_k, tokenizer)
        ft_sct_ranked = score_and_rank(ft_sct, output_set, top_k, tokenizer)
        ft_ll_ranked = score_and_rank(ft_mid, output_set, top_k, tokenizer)
        combined_ranked = score_and_rank(combined, output_set, top_k, tokenizer)

        results.append({
            "user_prompt": pair["user_prompt"],
            "model_response": pair["model_response"],
            "model_response_index": pair["model_response_index"],
            "ranked_tokens": mid_diff_ranked[:20],
            "mid_diff_top": mid_diff_ranked,
            "diff_sct_top": diff_ranked,
            "ft_sct_top": ft_sct_ranked,
            "ft_ll_top": ft_ll_ranked,
            "combined_top": combined_ranked,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Differential SCT extraction")
    parser.add_argument("--ft_model_name", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--source_activations", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mid_layer", type=int, default=32)
    parser.add_argument("--final_layer", type=int, default=41)
    parser.add_argument("--top_k_out", type=int, default=500)
    parser.add_argument("--combo_lambda", type=float, default=1.0)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    pairs = load_pairs(args.source_activations, args.max_examples)

    print(f"Loading finetuned model: {args.ft_model_name}")
    ft_model, tokenizer = load_model_and_tokenizer(args.ft_model_name)
    device = next(ft_model.parameters()).device
    print(f"FT model on device: {device}")

    ft_mid, ft_final, valid_ft = extract_all(
        ft_model, tokenizer, pairs, args.mid_layer, args.final_layer, device, "FT"
    )
    print(f"FT done: {sum(valid_ft)}/{len(valid_ft)} valid")
    del ft_model
    gc.collect()
    torch.cuda.empty_cache()
    print("FT model unloaded")

    print(f"Loading base model: {args.base_model_name}")
    base_model, _ = load_model_and_tokenizer(args.base_model_name)

    base_mid, base_final, valid_base = extract_all(
        base_model, tokenizer, pairs, args.mid_layer, args.final_layer, device, "Base"
    )
    print(f"Base done: {sum(valid_base)}/{len(valid_base)} valid")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Base model unloaded")

    print("Computing scores...")
    results = compute_all_scores(
        ft_mid, ft_final, base_mid, base_final,
        valid_ft, valid_base, pairs, args.top_k_out, tokenizer, args.combo_lambda,
    )

    output = {
        "metadata": {
            "ft_model_name": args.ft_model_name,
            "base_model_name": args.base_model_name,
            "source_file": args.source_activations,
            "mid_layer": args.mid_layer,
            "final_layer": args.final_layer,
            "top_k_out": args.top_k_out,
            "combo_lambda": args.combo_lambda,
            "scoring_method": "differential_multi",
            "total_pairs": len(results),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f)

    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    errors = sum(1 for r in results if "error" in r)
    print(f"Saved to {args.output_path} ({size_mb:.1f} MB)")
    print(f"Done: {len(results) - errors}/{len(results)} successful, {errors} errors")

    if results and not results[0].get("error"):
        for method_key in ["mid_diff_top", "diff_sct_top", "ft_sct_top", "ft_ll_top", "combined_top"]:
            top5 = results[0].get(method_key, [])[:5]
            tokens_str = ", ".join(f"{t.get('token','').strip()!r}:{t['score']:.2f}" for t in top5)
            print(f"  {method_key:15s}: {tokens_str}")


if __name__ == "__main__":
    main()
