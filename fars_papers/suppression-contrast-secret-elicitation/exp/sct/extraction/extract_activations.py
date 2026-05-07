# Activation extraction: forward pass through Gemma-2 model organisms,
# capture hidden states at mid (layer 32) and final (layer 41) layers,
# project through layernorm + unembedding to get per-position log-probs.
# Also handles response generation via the benchmark's InferenceEngine.
# Outputs top-K token log-probs per position per layer as JSON.

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
from sampling.inference_engine import InferenceEngine


def find_second_start_of_turn_position(tokens, tokenizer):
    """Find position of the second <start_of_turn> token (beginning of assistant response)."""
    start_token_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    positions = []
    for i, tid in enumerate(tokens):
        if tid == start_token_id:
            positions.append(i)
    return positions[1] if len(positions) >= 2 else -1


class MultiLayerCaptureHook:
    """Captures hidden states at multiple transformer layers during forward pass."""

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


def extract_logprobs_at_layers(
    model,
    tokenizer,
    tokens: torch.Tensor,
    layers: List[int],
    top_k: int = 200,
) -> Dict[int, List[List[Dict[str, Any]]]]:
    """Run a single forward pass and extract top-K log-probs at specified layers.

    Returns dict: layer_idx -> list of positions, each is list of {token_id, log_prob}.
    """
    device = next(model.parameters()).device
    input_ids = tokens.unsqueeze(0).to(device)

    with MultiLayerCaptureHook(model, layers) as hook:
        with torch.no_grad():
            _ = model(input_ids)

        results = {}
        for layer_idx in layers:
            hidden = hook.representations[layer_idx]
            normed = model.model.norm(hidden)
            logits = model.lm_head(normed)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs = log_probs.squeeze(0)

            topk_vals, topk_ids = torch.topk(log_probs, top_k, dim=-1)

            layer_data = []
            for pos in range(log_probs.shape[0]):
                pos_data = []
                for k in range(top_k):
                    pos_data.append({
                        "token_id": topk_ids[pos, k].item(),
                        "log_prob": topk_vals[pos, k].item(),
                    })
                layer_data.append(pos_data)
            results[layer_idx] = layer_data

    return results


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    num_responses: int = 10,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    seed: int = 1,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """Generate responses for prompts using benchmark InferenceEngine."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    engine = InferenceEngine(model, tokenizer)

    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        fmt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        formatted_prompts.append(fmt)

    results_dict = engine.generate_batch(
        formatted_prompts=formatted_prompts,
        num_responses_per_prompt=num_responses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )

    pairs = []
    for prompt, fmt_prompt in zip(prompts, formatted_prompts):
        responses = results_dict.get(fmt_prompt, [])
        for resp_idx, response in enumerate(responses):
            pairs.append({
                "user_prompt": prompt,
                "model_response": response,
                "model_response_index": resp_idx,
            })
    return pairs


def extract_activations_for_pairs(
    model,
    tokenizer,
    pairs: List[Dict[str, Any]],
    mid_layer: int = 32,
    final_layer: int = 41,
    top_k: int = 200,
) -> List[Dict[str, Any]]:
    """Extract top-K log-probs at mid and final layers for each pair."""
    layers = [mid_layer, final_layer]
    results = []

    for pair in tqdm(pairs, desc="Extracting activations"):
        user_prompt = pair["user_prompt"]
        model_response = pair["model_response"]

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt")
        tokens = tokens.squeeze(0)

        response_start = find_second_start_of_turn_position(tokens, tokenizer)
        if response_start == -1:
            results.append({
                "user_prompt": user_prompt,
                "model_response": model_response,
                "model_response_index": pair.get("model_response_index", 0),
                "error": "Could not find response start position",
            })
            continue

        response_token_ids = tokens[response_start:].tolist()

        try:
            layer_logprobs = extract_logprobs_at_layers(
                model, tokenizer, tokens, layers, top_k
            )
        except Exception as e:
            results.append({
                "user_prompt": user_prompt,
                "model_response": model_response,
                "model_response_index": pair.get("model_response_index", 0),
                "error": str(e),
            })
            continue

        mid_data = layer_logprobs[mid_layer][response_start:]
        final_data = layer_logprobs[final_layer][response_start:]

        results.append({
            "user_prompt": user_prompt,
            "model_response": model_response,
            "model_response_index": pair.get("model_response_index", 0),
            "response_start_pos": response_start,
            "response_token_ids": response_token_ids,
            "mid_layer_logprobs": mid_data,
            "final_layer_logprobs": final_data,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract activations for Taboo model organisms")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mid_layer", type=int, default=32)
    parser.add_argument("--final_layer", type=int, default=41)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--num_responses", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts for debugging")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    with open(args.prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts")

    print(f"Generating {args.num_responses} responses per prompt...")
    pairs = generate_responses(
        model, tokenizer, prompts,
        num_responses=args.num_responses,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"Generated {len(pairs)} prompt-response pairs")

    print(f"Extracting activations at layers {args.mid_layer} and {args.final_layer}...")
    activations = extract_activations_for_pairs(
        model, tokenizer, pairs,
        mid_layer=args.mid_layer,
        final_layer=args.final_layer,
        top_k=args.top_k,
    )

    output = {
        "metadata": {
            "model_name": args.model_name,
            "prompts_file": args.prompts_file,
            "mid_layer": args.mid_layer,
            "final_layer": args.final_layer,
            "top_k": args.top_k,
            "num_responses": args.num_responses,
            "num_prompts": len(prompts),
            "total_pairs": len(activations),
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "results": activations,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f)
    print(f"Saved activations to {args.output_path}")

    errors = sum(1 for r in activations if "error" in r)
    print(f"Done: {len(activations) - errors}/{len(activations)} successful, {errors} errors")


if __name__ == "__main__":
    main()
