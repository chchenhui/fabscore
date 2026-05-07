# DoLa-direction (final-mid) scorer - negative control.
# score_DoLa(v) = E_t[log p_N(v|t) - log p_L(v|t)]
# Opposite of SCT: ranks tokens that become MORE probable in later layers.
# Uses plausibility constraint on mid-layer (alpha=0.1) and excludes output tokens.

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np


def score_dola_direction(
    mid_layer_logprobs: List[List[Dict[str, Any]]],
    final_layer_logprobs: List[List[Dict[str, Any]]],
    response_token_ids: List[int],
    alpha: float = 0.1,
    top_k_out: int = 20,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    if not mid_layer_logprobs or not final_layer_logprobs:
        return []

    log_alpha = math.log(alpha)
    num_positions = len(mid_layer_logprobs)

    token_score_sums: Dict[int, float] = defaultdict(float)
    token_position_counts: Dict[int, int] = defaultdict(int)

    for pos_idx in range(num_positions):
        mid_data = mid_layer_logprobs[pos_idx]
        final_data = final_layer_logprobs[pos_idx]
        if not mid_data or not final_data:
            continue

        max_mid_logprob = mid_data[0]["log_prob"]
        threshold = log_alpha + max_mid_logprob

        final_lookup = {entry["token_id"]: entry["log_prob"] for entry in final_data}
        final_floor = final_data[-1]["log_prob"] if final_data else -30.0

        for entry in mid_data:
            if entry["log_prob"] < threshold:
                break
            tid = entry["token_id"]
            mid_lp = entry["log_prob"]
            final_lp = final_lookup.get(tid, final_floor)
            token_score_sums[tid] += (final_lp - mid_lp)
            token_position_counts[tid] += 1

    output_token_set: Set[int] = set(response_token_ids)

    scored = []
    for tid, score_sum in token_score_sums.items():
        if tid in output_token_set:
            continue
        count = token_position_counts[tid]
        mean_score = score_sum / count
        token_str = ""
        if tokenizer is not None:
            token_str = tokenizer.decode([tid])
        scored.append({
            "token_id": tid,
            "token": token_str,
            "score": mean_score,
            "position_count": count,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k_out]


def score_all_examples(
    activations_data: Dict[str, Any],
    alpha: float = 0.1,
    top_k_out: int = 20,
    tokenizer=None,
) -> List[Dict[str, Any]]:
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
        final_logprobs = item.get("final_layer_logprobs", [])
        response_token_ids = item.get("response_token_ids", [])

        ranked = score_dola_direction(
            mid_layer_logprobs=mid_logprobs,
            final_layer_logprobs=final_logprobs,
            response_token_ids=response_token_ids,
            alpha=alpha,
            top_k_out=top_k_out,
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
    parser = argparse.ArgumentParser(description="DoLa-direction (final-mid) scorer")
    parser.add_argument("--activations_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--top_k_out", type=int, default=20)
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

    print(f"Scoring with alpha={args.alpha}, top_k_out={args.top_k_out}")
    scored = score_all_examples(
        activations_data,
        alpha=args.alpha,
        top_k_out=args.top_k_out,
        tokenizer=tokenizer,
    )

    metadata = activations_data.get("metadata", {})
    output = {
        "metadata": {
            **metadata,
            "scoring_method": "dola_direction",
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
