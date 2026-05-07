# Constrained logit lens scoring with mid-layer plausibility filter.
# Two modes:
#   1. "control_tokens" (default): average log-probs at control positions
#      (<start_of_turn> and model), matching the benchmark's approach.
#   2. "all_response": average across all response positions.
# Applies plausibility constraint (alpha=0.1), excludes output tokens.

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def score_logit_lens_constrained(
    mid_layer_logprobs: List[List[Dict[str, Any]]],
    response_token_ids: List[int],
    alpha: float = 0.1,
    top_k_out: int = 20,
    mode: str = "control_tokens",
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """Compute constrained logit lens token ranking for a single example.

    Args:
        mid_layer_logprobs: Per-position top-K log-probs from response_start_pos onwards.
            Position 0 = <start_of_turn>, Position 1 = model token.
        response_token_ids: Token IDs from response_start_pos onwards.
        alpha: Plausibility threshold fraction.
        top_k_out: Number of tokens to return.
        mode: "control_tokens" uses positions 0,1 only; "all_response" uses all positions.
    """
    if not mid_layer_logprobs:
        return []

    if mode == "control_tokens":
        positions_to_use = list(range(min(2, len(mid_layer_logprobs))))
    else:
        positions_to_use = list(range(len(mid_layer_logprobs)))

    if not positions_to_use:
        return []

    log_alpha = math.log(alpha)
    num_positions = len(positions_to_use)

    token_logprob_sums: Dict[int, float] = defaultdict(float)
    token_position_counts: Dict[int, int] = defaultdict(int)

    for pos_idx in positions_to_use:
        pos_data = mid_layer_logprobs[pos_idx]
        if not pos_data:
            continue
        max_logprob = pos_data[0]["log_prob"]
        threshold = log_alpha + max_logprob

        for entry in pos_data:
            if entry["log_prob"] >= threshold:
                tid = entry["token_id"]
                token_logprob_sums[tid] += entry["log_prob"]
                token_position_counts[tid] += 1

    output_token_set: Set[int] = set(response_token_ids)

    scored = []
    for tid, logprob_sum in token_logprob_sums.items():
        if tid in output_token_set:
            continue
        count = token_position_counts[tid]
        mean_logprob = logprob_sum / count
        token_str = ""
        if tokenizer is not None:
            token_str = tokenizer.decode([tid])
        scored.append({
            "token_id": tid,
            "token": token_str,
            "score": mean_logprob,
            "position_count": count,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k_out]


def score_all_examples(
    activations_data: Dict[str, Any],
    alpha: float = 0.1,
    top_k_out: int = 20,
    mode: str = "control_tokens",
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """Score all examples in an activations file."""
    results_list = activations_data.get("results", [])
    scored_results = []

    for item in results_list:
        if "error" in item:
            scored_results.append({
                "user_prompt": item.get("user_prompt", ""),
                "model_response": item.get("model_response", ""),
                "model_response_index": item.get("model_response_index", 0),
                "error": item["error"],
                "ranked_tokens": [],
            })
            continue

        mid_logprobs = item.get("mid_layer_logprobs", [])
        response_token_ids = item.get("response_token_ids", [])

        ranked = score_logit_lens_constrained(
            mid_layer_logprobs=mid_logprobs,
            response_token_ids=response_token_ids,
            alpha=alpha,
            top_k_out=top_k_out,
            mode=mode,
            tokenizer=tokenizer,
        )

        scored_results.append({
            "user_prompt": item.get("user_prompt", ""),
            "model_response": item.get("model_response", ""),
            "model_response_index": item.get("model_response_index", 0),
            "ranked_tokens": ranked,
        })

    return scored_results


def main():
    parser = argparse.ArgumentParser(description="Constrained logit lens scoring")
    parser.add_argument("--activations_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--top_k_out", type=int, default=20)
    parser.add_argument("--mode", type=str, default="control_tokens",
                        choices=["control_tokens", "all_response"])
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for tokenizer (to decode token strings)")
    args = parser.parse_args()

    print(f"Loading activations from {args.activations_file}")
    with open(args.activations_file, "r") as f:
        activations_data = json.load(f)

    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Scoring with alpha={args.alpha}, top_k_out={args.top_k_out}, mode={args.mode}")
    scored = score_all_examples(
        activations_data,
        alpha=args.alpha,
        top_k_out=args.top_k_out,
        mode=args.mode,
        tokenizer=tokenizer,
    )

    metadata = activations_data.get("metadata", {})
    output = {
        "metadata": {
            **metadata,
            "scoring_method": "constrained_logit_lens",
            "scoring_mode": args.mode,
            "alpha": args.alpha,
            "top_k_out": args.top_k_out,
        },
        "results": scored,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved scored results to {args.output_path}")

    errors = sum(1 for r in scored if "error" in r)
    print(f"Done: {len(scored) - errors}/{len(scored)} scored, {errors} errors")


if __name__ == "__main__":
    main()
