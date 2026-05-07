# Optimized SCT scorer: combines control-position SCT + weighted all-position SCT.
# Best variant from optimization sweep:
#   score(v) = CtrlSCT(v) + weight * AllSCT(v)
# where CtrlSCT uses positions 0,1 with alpha=0 (no plausibility filter),
# and AllSCT uses all positions with alpha=0.1 plausibility filter.

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


def score_optimized_sct(
    mid_layer_logprobs: List[List[Dict[str, Any]]],
    final_layer_logprobs: List[List[Dict[str, Any]]],
    response_token_ids: List[int],
    ctrl_alpha: float = 0.0,
    all_alpha: float = 0.1,
    weight: float = 0.1,
    num_ctrl_positions: int = 2,
    top_k_out: int = 20,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    if not mid_layer_logprobs or not final_layer_logprobs:
        return []

    output_set: Set[int] = set(response_token_ids)
    num_positions = len(mid_layer_logprobs)

    ctrl_sct: Dict[int, float] = defaultdict(float)
    ctrl_count: Dict[int, int] = defaultdict(int)

    num_ctrl = min(num_ctrl_positions, num_positions)
    for pos_idx in range(num_ctrl):
        mid_data = mid_layer_logprobs[pos_idx]
        final_data = final_layer_logprobs[pos_idx]
        if not mid_data or not final_data:
            continue

        max_mid = mid_data[0]["log_prob"]
        threshold = (math.log(ctrl_alpha) + max_mid) if ctrl_alpha > 0 else -float("inf")

        final_lookup = {e["token_id"]: e["log_prob"] for e in final_data}
        final_floor = final_data[-1]["log_prob"] if final_data else -30.0

        for e in mid_data:
            if e["log_prob"] < threshold:
                break
            tid = e["token_id"]
            if tid in output_set:
                continue
            flp = final_lookup.get(tid, final_floor)
            ctrl_sct[tid] += (e["log_prob"] - flp)
            ctrl_count[tid] += 1

    all_sct: Dict[int, float] = defaultdict(float)
    all_count: Dict[int, int] = defaultdict(int)

    if weight > 0:
        log_alpha = math.log(all_alpha) if all_alpha > 0 else None
        for pos_idx in range(num_positions):
            mid_data = mid_layer_logprobs[pos_idx]
            final_data = final_layer_logprobs[pos_idx]
            if not mid_data or not final_data:
                continue

            max_mid = mid_data[0]["log_prob"]
            threshold = (log_alpha + max_mid) if log_alpha is not None else -float("inf")

            final_lookup = {e["token_id"]: e["log_prob"] for e in final_data}
            final_floor = final_data[-1]["log_prob"] if final_data else -30.0

            for e in mid_data:
                if e["log_prob"] < threshold:
                    break
                tid = e["token_id"]
                if tid in output_set:
                    continue
                flp = final_lookup.get(tid, final_floor)
                all_sct[tid] += (e["log_prob"] - flp)
                all_count[tid] += 1

    combined: Dict[int, float] = {}
    all_tids = set(ctrl_sct.keys()) | set(all_sct.keys())
    for tid in all_tids:
        combined[tid] = ctrl_sct.get(tid, 0.0) + weight * all_sct.get(tid, 0.0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    results = []
    for tid, score in ranked[:top_k_out]:
        token_str = ""
        if tokenizer is not None:
            token_str = tokenizer.decode([tid])
        results.append({
            "token_id": tid,
            "token": token_str,
            "score": score,
            "position_count": ctrl_count.get(tid, 0) + all_count.get(tid, 0),
        })

    return results


def score_all_examples(
    activations_data: Dict[str, Any],
    ctrl_alpha: float = 0.0,
    all_alpha: float = 0.1,
    weight: float = 0.1,
    num_ctrl_positions: int = 2,
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

        ranked = score_optimized_sct(
            mid_layer_logprobs=mid_logprobs,
            final_layer_logprobs=final_logprobs,
            response_token_ids=response_token_ids,
            ctrl_alpha=ctrl_alpha,
            all_alpha=all_alpha,
            weight=weight,
            num_ctrl_positions=num_ctrl_positions,
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
    parser = argparse.ArgumentParser(description="Optimized SCT scorer")
    parser.add_argument("--activations_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ctrl_alpha", type=float, default=0.0)
    parser.add_argument("--all_alpha", type=float, default=0.1)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--num_ctrl_positions", type=int, default=2)
    parser.add_argument("--top_k_out", type=int, default=20)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading activations from {args.activations_file}")
    with open(args.activations_file, "r") as f:
        activations_data = json.load(f)

    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Scoring: ctrl_alpha={args.ctrl_alpha}, all_alpha={args.all_alpha}, weight={args.weight}")
    scored = score_all_examples(
        activations_data,
        ctrl_alpha=args.ctrl_alpha,
        all_alpha=args.all_alpha,
        weight=args.weight,
        num_ctrl_positions=args.num_ctrl_positions,
        top_k_out=args.top_k_out,
        tokenizer=tokenizer,
    )

    metadata = activations_data.get("metadata", {})
    output = {
        "metadata": {
            **metadata,
            "scoring_method": "optimized_sct",
            "ctrl_alpha": args.ctrl_alpha,
            "all_alpha": args.all_alpha,
            "weight": args.weight,
            "num_ctrl_positions": args.num_ctrl_positions,
            "top_k_out": args.top_k_out,
        },
        "results": scored,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved scored results to {args.output_path}")

    errors = sum(1 for r in scored if "error" in r)
    print(f"Done: {len(scored) - errors}/{len(scored)} scored, {errors} errors")

    if scored and scored[0].get("ranked_tokens"):
        print(f"\nTop-5 tokens for first example:")
        for t in scored[0]["ranked_tokens"][:5]:
            print(f"  {t['token']!r:15s} score={t['score']:.4f} positions={t['position_count']}")


if __name__ == "__main__":
    main()
