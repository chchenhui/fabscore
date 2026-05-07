# Improved SCT scorer v2: works with multi-layer extraction output.
# Supports layer sweep, control-token-only scoring, and combined scoring modes.
# Evaluates token recovery directly to enable rapid parameter search.

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def score_from_multilayer(
    results: List[Dict[str, Any]],
    secret_token_ids: List[int],
    mid_layer: int,
    mode: str = "sct",
    k_values: List[int] = [5, 20],
) -> Dict[str, Any]:
    """Evaluate a specific layer and scoring mode from multi-layer extraction output.

    Modes:
    - "sct": Use pre-computed full-vocab SCT scores at control positions
    - "ll": Use pre-computed logit lens scores at control positions
    """
    secret_set = set(secret_token_ids)
    layer_key = str(mid_layer)
    recovery = {k: 0 for k in k_values}
    total = 0
    ranked_per_example = []

    for item in results:
        if "error" in item:
            continue
        per_layer = item.get("per_layer", {})
        if layer_key not in per_layer:
            continue
        total += 1

        layer_data = per_layer[layer_key]
        if mode == "sct":
            ranked = layer_data.get("sct_ranked", [])
            score_key = "sct_score"
        else:
            ranked = layer_data.get("ll_ranked", [])
            score_key = "score"

        ranked_ids = [t["token_id"] for t in ranked]
        ranked_per_example.append(ranked)

        for k in k_values:
            top_k_ids = set(ranked_ids[:k])
            if top_k_ids & secret_set:
                recovery[k] += 1

    if total > 0:
        for k in k_values:
            recovery[k] /= total

    return {
        "mid_layer": mid_layer,
        "mode": mode,
        "total_examples": total,
        "token_recovery": {str(k): recovery[k] for k in k_values},
        "ranked_per_example": ranked_per_example,
    }


def score_from_original_activations(
    results: List[Dict[str, Any]],
    secret_token_ids: List[int],
    mode: str = "sct_control",
    alpha: float = 0.0,
    k_values: List[int] = [5, 20],
    top_k_out: int = 20,
) -> Dict[str, Any]:
    """Score using original activation files (backward compatible).

    Modes:
    - "sct_control": SCT at control positions only
    - "sct_all": SCT at all positions (original behavior)
    - "ll_control": Logit lens at control positions
    """
    secret_set = set(secret_token_ids)
    log_alpha = math.log(alpha) if alpha > 0 else None
    recovery = {k: 0 for k in k_values}
    total = 0
    ranked_per_example = []

    for item in results:
        if "error" in item:
            continue
        total += 1
        mid = item.get("mid_layer_logprobs", [])
        final = item.get("final_layer_logprobs", [])
        output_set = set(item.get("response_token_ids", []))

        if "control" in mode:
            positions = list(range(min(2, len(mid))))
        else:
            positions = list(range(len(mid)))

        num_pos = len(positions) if positions else 1

        scores = defaultdict(float)
        counts = defaultdict(int)

        for pos_idx in positions:
            if pos_idx >= len(mid) or pos_idx >= len(final):
                continue
            mid_data = mid[pos_idx]
            final_data = final[pos_idx]
            if not mid_data or not final_data:
                continue

            if log_alpha is not None:
                max_mid = mid_data[0]["log_prob"]
                threshold = log_alpha + max_mid
            else:
                threshold = -float('inf')

            if "sct" in mode:
                final_lookup = {e["token_id"]: e["log_prob"] for e in final_data}
                final_floor = final_data[-1]["log_prob"] if final_data else -30.0

                for e in mid_data:
                    if e["log_prob"] < threshold:
                        break
                    tid = e["token_id"]
                    if tid in output_set:
                        continue
                    flp = final_lookup.get(tid, final_floor)
                    scores[tid] += (e["log_prob"] - flp)
                    counts[tid] += 1
            else:
                for e in mid_data:
                    if e["log_prob"] < threshold:
                        break
                    tid = e["token_id"]
                    if tid in output_set:
                        continue
                    scores[tid] += e["log_prob"]
                    counts[tid] += 1

        if "sct" in mode:
            final_scores = dict(scores)
        else:
            final_scores = {tid: scores[tid] / max(counts[tid], 1) for tid in scores}

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_out]
        ranked_tokens = [{"token_id": tid, "score": sc} for tid, sc in ranked]
        ranked_per_example.append(ranked_tokens)

        ranked_ids = [tid for tid, _ in ranked]
        for k in k_values:
            top_k_ids = set(ranked_ids[:k])
            if top_k_ids & secret_set:
                recovery[k] += 1

    if total > 0:
        for k in k_values:
            recovery[k] /= total

    return {
        "mode": mode,
        "alpha": alpha,
        "total_examples": total,
        "token_recovery": {str(k): recovery[k] for k in k_values},
        "ranked_per_example": ranked_per_example,
    }


def main():
    parser = argparse.ArgumentParser(description="SCT scorer v2 with layer sweep")
    parser.add_argument("--multilayer_files", type=str, nargs="+",
                        help="Multi-layer extraction output files")
    parser.add_argument("--original_files", type=str, nargs="+",
                        help="Original activation files (for backward-compat scoring)")
    parser.add_argument("--secret_words", type=str, nargs="+", required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def get_secret_ids(word):
        ids = set()
        bare = tokenizer.encode(word, add_special_tokens=False)
        space = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(bare) == 1:
            ids.add(bare[0])
        if len(space) >= 1:
            ids.add(space[-1])
        if not ids:
            ids.add(bare[0])
        return list(ids)

    all_results = {}

    if args.multilayer_files:
        for fpath, secret_word in zip(args.multilayer_files, args.secret_words):
            secret_ids = get_secret_ids(secret_word)
            print(f"\n=== {secret_word} (multi-layer) ===")
            print(f"Secret token IDs: {secret_ids}")

            with open(fpath) as f:
                data = json.load(f)

            mid_layers = data["metadata"]["mid_layers"]
            results = data["results"]

            model_results = {}
            for mid_layer in mid_layers:
                for mode in ["sct", "ll"]:
                    res = score_from_multilayer(results, secret_ids, mid_layer, mode, args.k_values)
                    key = f"L{mid_layer}_{mode}"
                    model_results[key] = res
                    tr5 = res["token_recovery"].get("5", 0)
                    tr20 = res["token_recovery"].get("20", 0)
                    print(f"  {key:15s}: TR@5={tr5*100:5.1f}%  TR@20={tr20*100:5.1f}%")

            all_results[secret_word] = model_results

    if args.original_files:
        for fpath, secret_word in zip(args.original_files, args.secret_words):
            secret_ids = get_secret_ids(secret_word)
            print(f"\n=== {secret_word} (original activations) ===")

            with open(fpath) as f:
                data = json.load(f)

            results = data["results"]

            model_results = all_results.get(secret_word, {})
            modes = [
                ("sct_control", 0.0),
                ("sct_control", 0.1),
                ("sct_all", 0.1),
                ("ll_control", 0.0),
                ("ll_control", 0.1),
            ]

            for mode, alpha in modes:
                res = score_from_original_activations(
                    results, secret_ids, mode, alpha, args.k_values)
                key = f"orig_{mode}_a{alpha}"
                model_results[key] = {
                    k: v for k, v in res.items() if k != "ranked_per_example"
                }
                tr5 = res["token_recovery"].get("5", 0)
                tr20 = res["token_recovery"].get("20", 0)
                print(f"  {key:30s}: TR@5={tr5*100:5.1f}%  TR@20={tr20*100:5.1f}%")

            all_results[secret_word] = model_results

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        serializable = {}
        for word, model_res in all_results.items():
            serializable[word] = {}
            for key, val in model_res.items():
                clean = {k: v for k, v in val.items() if k != "ranked_per_example"}
                serializable[word][key] = clean
        with open(args.output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()
