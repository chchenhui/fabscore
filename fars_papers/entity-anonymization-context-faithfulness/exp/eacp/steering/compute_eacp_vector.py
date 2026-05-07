"""Compute EACP-native ContextFocus steering vector. Extracts last-token residual
stream activations from positive (EACP with context) vs negative (EACP without
context) prompts. Reuses extraction logic from contextfocus.py."""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
VECTORS_DIR = os.path.join(SCRIPT_DIR, "vectors")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 4


def extract_activations_batched(model, tokenizer, texts, batch_size, num_layers, device):
    all_activations = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        actual_batch_size = len(batch_texts)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        layer_activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0].detach().cpu()
                else:
                    hidden = output.detach().cpu()
                layer_activations[layer_idx] = hidden
            return hook_fn

        hooks = []
        for l in range(num_layers):
            h = model.model.layers[l].register_forward_hook(make_hook(l))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        attention_mask = inputs["attention_mask"].cpu()
        seq_lengths = attention_mask.sum(dim=1) - 1

        batch_acts = []
        for l in range(num_layers):
            acts = layer_activations[l]
            last_token_acts = acts[torch.arange(actual_batch_size), seq_lengths]
            batch_acts.append(last_token_acts)

        batch_acts = torch.stack(batch_acts, dim=1)
        all_activations.append(batch_acts)

        if (batch_start // batch_size) % 20 == 0:
            print(f"  Processed {batch_start + len(batch_texts)}/{len(texts)} examples")

    return torch.cat(all_activations, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pairs_path", default=os.path.join(DATA_DIR, "eacp_steering_pairs.json"))
    parser.add_argument("--output_dir", default=VECTORS_DIR)
    args = parser.parse_args()

    with open(args.pairs_path) as f:
        pairs = json.load(f)

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Using {len(pairs)} examples (limited)")
    else:
        print(f"Using all {len(pairs)} examples")

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers, hidden_dim={model.config.hidden_size}")

    pos_texts = [p["positive_text"] for p in pairs]
    neg_texts = [p["negative_text"] for p in pairs]

    print(f"\nExtracting positive activations ({len(pos_texts)} examples)...")
    t0 = time.time()
    pos_acts = extract_activations_batched(
        model, tokenizer, pos_texts, args.batch_size, num_layers, device
    )
    print(f"Positive activations: {pos_acts.shape}, took {time.time()-t0:.1f}s")

    print(f"\nExtracting negative activations ({len(neg_texts)} examples)...")
    t0 = time.time()
    neg_acts = extract_activations_batched(
        model, tokenizer, neg_texts, args.batch_size, num_layers, device
    )
    print(f"Negative activations: {neg_acts.shape}, took {time.time()-t0:.1f}s")

    diff = pos_acts - neg_acts
    steering_vectors = diff.mean(dim=0)
    print(f"\nSteering vectors shape: {steering_vectors.shape}")

    print("\nPer-layer L2 norms:")
    for l in range(num_layers):
        norm = steering_vectors[l].float().norm().item()
        print(f"  Layer {l:2d}: L2 = {norm:.4f}")

    has_nan = torch.isnan(steering_vectors).any().item()
    has_inf = torch.isinf(steering_vectors).any().item()
    print(f"\nNaN check: {has_nan}, Inf check: {has_inf}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "llama31_8b_eacp_contextfocus.pt")
    torch.save(steering_vectors, out_path)
    print(f"\nSaved EACP steering vectors to {out_path}")


if __name__ == "__main__":
    main()
