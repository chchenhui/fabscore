# Multi-layer full-vocabulary SCT scoring at control positions.
# Reuses existing prompt-response pairs from previous activation files.
# Computes exact SCT scores (no top-K truncation) at control positions
# for each mid-layer, enabling layer-sweep optimization.
# Output is compact: only pre-ranked token lists per layer per example.

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
sys.path.insert(0, BENCHMARK_DIR)

from sampling.utils import load_model_and_tokenizer


def find_second_start_of_turn_position(tokens, tokenizer):
    start_token_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    positions = []
    for i, tid in enumerate(tokens):
        if tid == start_token_id:
            positions.append(i)
    return positions[1] if len(positions) >= 2 else -1


class MultiLayerCaptureHook:
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.representations: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.representations[layer_idx] = hidden_states.detach()
        return hook_fn

    def __enter__(self):
        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx]
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.representations.clear()


def extract_and_score(
    model,
    tokens: torch.Tensor,
    mid_layers: List[int],
    final_layer: int,
    response_start: int,
    response_token_ids: List[int],
    top_k_out: int = 50,
) -> Dict[str, Any]:
    """Single forward pass: compute full-vocab SCT and logit-lens scores at control positions."""
    device = next(model.parameters()).device
    input_ids = tokens.unsqueeze(0).to(device)
    all_layers = sorted(set(mid_layers + [final_layer]))

    ctrl_abs = [response_start + i for i in range(2) if response_start + i < len(tokens)]
    output_set = set(response_token_ids)

    output_mask = torch.ones(model.config.vocab_size, dtype=torch.bool, device=device)
    for tid in output_set:
        if tid < output_mask.shape[0]:
            output_mask[tid] = False

    with MultiLayerCaptureHook(model, all_layers) as hook:
        with torch.no_grad():
            _ = model(input_ids)

        final_lps = []
        hidden_final = hook.representations[final_layer]
        normed_final = model.model.norm(hidden_final)
        logits_final = model.lm_head(normed_final)
        lp_final_all = F.log_softmax(logits_final.float(), dim=-1).squeeze(0)
        for abs_pos in ctrl_abs:
            final_lps.append(lp_final_all[abs_pos])

        per_layer_results = {}

        for mid_layer in mid_layers:
            hidden_mid = hook.representations[mid_layer]
            normed_mid = model.model.norm(hidden_mid)
            logits_mid = model.lm_head(normed_mid)
            lp_mid_all = F.log_softmax(logits_mid.float(), dim=-1).squeeze(0)

            mid_lps = [lp_mid_all[abs_pos] for abs_pos in ctrl_abs]
            num_ctrl = len(ctrl_abs)

            sct_accum = torch.zeros(model.config.vocab_size, device=device)
            ll_accum = torch.zeros(model.config.vocab_size, device=device)
            for i in range(num_ctrl):
                sct_accum += (mid_lps[i] - final_lps[i])
                ll_accum += mid_lps[i]

            if num_ctrl > 0:
                sct_mean = sct_accum / num_ctrl
                ll_mean = ll_accum / num_ctrl

            sct_masked = sct_mean.clone()
            sct_masked[~output_mask] = -float('inf')
            ll_masked = ll_mean.clone()
            ll_masked[~output_mask] = -float('inf')

            sct_vals, sct_ids = torch.topk(sct_masked, top_k_out)
            ll_vals, ll_ids = torch.topk(ll_masked, top_k_out)

            sct_tokens = []
            for k in range(top_k_out):
                tid = sct_ids[k].item()
                sct_tokens.append({
                    "token_id": tid,
                    "sct_score": sct_vals[k].item(),
                    "mid_lp": ll_mean[tid].item(),
                    "final_lp": (sum(final_lps[i][tid].item() for i in range(num_ctrl)) / num_ctrl),
                })

            ll_tokens = []
            for k in range(top_k_out):
                tid = ll_ids[k].item()
                ll_tokens.append({
                    "token_id": tid,
                    "score": ll_vals[k].item(),
                })

            per_layer_results[str(mid_layer)] = {
                "sct_ranked": sct_tokens,
                "ll_ranked": ll_tokens,
            }

    return {"per_layer": per_layer_results}


def main():
    parser = argparse.ArgumentParser(description="Multi-layer full-vocab SCT extraction")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--source_activations", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mid_layers", type=int, nargs="+", default=[15, 20, 24, 28, 32, 36])
    parser.add_argument("--final_layer", type=int, default=41)
    parser.add_argument("--top_k_out", type=int, default=50)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading source pairs from {args.source_activations}")
    with open(args.source_activations, "r") as f:
        source_data = json.load(f)

    pairs = []
    for item in source_data["results"]:
        if "error" in item:
            continue
        pairs.append({
            "user_prompt": item["user_prompt"],
            "model_response": item["model_response"],
            "model_response_index": item.get("model_response_index", 0),
        })

    if args.max_examples:
        pairs = pairs[:args.max_examples]
    print(f"Processing {len(pairs)} pairs")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    print(f"Mid layers: {args.mid_layers}, Final: {args.final_layer}, top_k_out: {args.top_k_out}")
    results = []

    for pair in tqdm(pairs, desc="Extracting"):
        messages = [
            {"role": "user", "content": pair["user_prompt"]},
            {"role": "assistant", "content": pair["model_response"]},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False,
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)

        response_start = find_second_start_of_turn_position(tokens, tokenizer)
        if response_start == -1:
            results.append({
                "user_prompt": pair["user_prompt"],
                "model_response": pair["model_response"],
                "model_response_index": pair.get("model_response_index", 0),
                "error": "Could not find response start position",
            })
            continue

        response_token_ids = tokens[response_start:].tolist()

        try:
            extraction = extract_and_score(
                model, tokens,
                mid_layers=args.mid_layers,
                final_layer=args.final_layer,
                response_start=response_start,
                response_token_ids=response_token_ids,
                top_k_out=args.top_k_out,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "user_prompt": pair["user_prompt"],
                "model_response": pair["model_response"],
                "model_response_index": pair.get("model_response_index", 0),
                "error": str(e),
            })
            continue

        results.append({
            "user_prompt": pair["user_prompt"],
            "model_response": pair["model_response"],
            "model_response_index": pair.get("model_response_index", 0),
            "response_token_ids": response_token_ids,
            **extraction,
        })

    output = {
        "metadata": {
            "model_name": args.model_name,
            "source_file": args.source_activations,
            "mid_layers": args.mid_layers,
            "final_layer": args.final_layer,
            "top_k_out": args.top_k_out,
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


if __name__ == "__main__":
    main()
