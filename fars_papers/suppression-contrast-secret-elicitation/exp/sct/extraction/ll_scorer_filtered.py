# Logit lens scorer with cross-model generic token filter.
# Uses mid-layer log-prob at control positions (alpha=0), then removes tokens
# that appear frequently in the top-20 of ALL other models (cross-model filter).
# This removes game-generic tokens (Sorry, guess) that obscure the secret.

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


def compute_ctrl_ll_scores(
    mid_layer_logprobs: List[List[Dict[str, Any]]],
    response_token_ids: List[int],
    num_ctrl: int = 2,
) -> List[Tuple[int, float]]:
    if not mid_layer_logprobs:
        return []

    output_set = set(response_token_ids)
    scores: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)

    for pos_idx in range(min(num_ctrl, len(mid_layer_logprobs))):
        for e in mid_layer_logprobs[pos_idx]:
            tid = e["token_id"]
            if tid in output_set:
                continue
            scores[tid] += e["log_prob"]
            counts[tid] += 1

    mean_scores = {tid: scores[tid] / counts[tid] for tid in scores}
    return sorted(mean_scores.items(), key=lambda x: -x[1])


def build_generic_set_from_sct(
    activations_files: List[str],
    target_index: int,
    threshold_pct: float = 1.0,
    top_k_scan: int = 20,
    num_ctrl: int = 2,
) -> Set[int]:
    other_counts = []
    for i, path in enumerate(activations_files):
        if i == target_index:
            continue
        with open(path) as f:
            data = json.load(f)
        counter = Counter()
        n = 0
        for item in data["results"]:
            if "error" in item:
                continue
            n += 1
            resp_tokens = set(item["response_token_ids"])
            scores = {}
            for pos_idx in range(min(num_ctrl, len(item["mid_layer_logprobs"]))):
                mid_data = item["mid_layer_logprobs"][pos_idx]
                fin_data = item["final_layer_logprobs"][pos_idx]
                fin_lookup = {e["token_id"]: e["log_prob"] for e in fin_data}
                fin_floor = fin_data[-1]["log_prob"] if fin_data else -30.0
                for e in mid_data:
                    tid = e["token_id"]
                    if tid in resp_tokens:
                        continue
                    scores[tid] = scores.get(tid, 0) + (e["log_prob"] - fin_lookup.get(tid, fin_floor))
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            for tid, _ in ranked[:top_k_scan]:
                counter[tid] += 1
        other_counts.append((counter, n))
        del data

    all_tids: Set[int] = set()
    for c, _ in other_counts:
        all_tids.update(c.keys())
    generic = set()
    for tid in all_tids:
        rates = [c.get(tid, 0) / n * 100 for c, n in other_counts]
        if all(r >= threshold_pct for r in rates):
            generic.add(tid)
    return generic


def score_all_filtered(
    activations_data: Dict[str, Any],
    generic_set: Set[int],
    num_ctrl: int = 2,
    top_k_out: int = 20,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    scored_results = []
    for item in activations_data.get("results", []):
        if "error" in item:
            scored_results.append({
                "user_prompt": item.get("user_prompt", ""),
                "model_response": item.get("model_response", ""),
                "model_response_index": item.get("model_response_index", 0),
                "error": item["error"],
                "ranked_tokens": [],
            })
            continue

        ranked = compute_ctrl_ll_scores(
            item["mid_layer_logprobs"],
            item["response_token_ids"],
            num_ctrl,
        )
        filtered = [(tid, s) for tid, s in ranked if tid not in generic_set]
        tokens_out = []
        for tid, score in filtered[:top_k_out]:
            entry = {"token_id": tid, "score": score, "rank": len(tokens_out) + 1}
            if tokenizer:
                entry["token"] = tokenizer.decode([tid])
            tokens_out.append(entry)

        scored_results.append({
            "user_prompt": item.get("user_prompt", ""),
            "model_response": item.get("model_response", ""),
            "model_response_index": item.get("model_response_index", 0),
            "ranked_tokens": tokens_out,
        })
    return scored_results


def main():
    parser = argparse.ArgumentParser(description="LL scorer with cross-model generic filter")
    parser.add_argument("--activations_files", type=str, nargs="+", required=True)
    parser.add_argument("--target_index", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--threshold_pct", type=float, default=1.0)
    parser.add_argument("--top_k_scan", type=int, default=20)
    parser.add_argument("--top_k_out", type=int, default=20)
    parser.add_argument("--num_ctrl", type=int, default=2)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Building generic token set from {len(args.activations_files) - 1} other models...")
    generic_set = build_generic_set_from_sct(
        args.activations_files, args.target_index,
        args.threshold_pct, args.top_k_scan, args.num_ctrl,
    )
    print(f"Generic tokens: {len(generic_set)}")

    print(f"Loading target: {args.activations_files[args.target_index]}")
    with open(args.activations_files[args.target_index]) as f:
        target_data = json.load(f)

    print("Scoring with LL + cross-model filter...")
    scored = score_all_filtered(target_data, generic_set, args.num_ctrl, args.top_k_out, tokenizer)

    metadata = target_data.get("metadata", {})
    output = {
        "metadata": {
            **metadata,
            "scoring_method": "ctrl_ll_crossmodel_filtered",
            "threshold_pct": args.threshold_pct,
            "top_k_scan": args.top_k_scan,
            "top_k_out": args.top_k_out,
            "num_ctrl": args.num_ctrl,
            "generic_token_count": len(generic_set),
        },
        "results": scored,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f)
    print(f"Saved to {args.output_path}")
    errors = sum(1 for r in scored if "error" in r)
    print(f"Done: {len(scored) - errors}/{len(scored)} scored, {errors} errors")


if __name__ == "__main__":
    main()
