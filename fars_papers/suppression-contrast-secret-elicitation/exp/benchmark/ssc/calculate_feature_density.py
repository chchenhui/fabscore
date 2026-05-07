#!/usr/bin/env python3
"""
Calculate feature density for SAE features.
Feature density = percentage of tokens on which a feature is activated (> 0).
"""
import argparse
import json

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sae_utils_ssc import ObservableLanguageModel, load_sae
from tqdm import tqdm

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Calculate SAE feature density")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000000,
        help="Maximum number of tokens to process (default: 1000000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="feature_density.json",
        help="Output JSON file (default: feature_density.json)"
    )
    args = parser.parse_args()

    # Configuration (same as explore_sae_features.py)
    MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    SAE_NAME = "Llama-3.3-70B-Instruct-SAE-l50"
    SAE_LAYER = "model.layers.50"
    EXPANSION_FACTOR = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("Loading language model...")
    model = ObservableLanguageModel(MODEL_NAME, device=DEVICE, dtype=dtype)

    print("Loading SAE...")
    sae = load_sae(
        SAE_NAME, d_model=model.d_model, expansion_factor=EXPANSION_FACTOR, device=DEVICE
    )

    print("Loading dataset...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Initialize feature activation counters
    num_features = sae.d_hidden
    feature_activation_count = torch.zeros(num_features, dtype=torch.long)
    total_tokens = 0

    print(f"Processing up to {args.max_tokens} tokens with batch size {args.batch_size}")

    batch_texts = []
    progress_bar = tqdm(total=args.max_tokens, desc="Processing tokens")

    for example in dataset:
        text = example["text"]
        batch_texts.append(text)

        if len(batch_texts) >= args.batch_size:
            tokens_processed = process_batch(
                batch_texts, model, sae, SAE_LAYER, feature_activation_count
            )
            total_tokens += tokens_processed
            progress_bar.update(tokens_processed)

            batch_texts = []

            if total_tokens >= args.max_tokens:
                break

    # Process remaining texts in batch
    if batch_texts and total_tokens < args.max_tokens:
        tokens_processed = process_batch(
            batch_texts, model, sae, SAE_LAYER, feature_activation_count
        )
        total_tokens += tokens_processed
        progress_bar.update(tokens_processed)

    progress_bar.close()

    # Calculate feature density
    print(f"Calculating feature density for {num_features} features over {total_tokens} tokens...")
    feature_density = {}

    for feature_idx in tqdm(range(num_features), desc="Computing densities"):
        activation_count = feature_activation_count[feature_idx].item()
        density = (activation_count / total_tokens) * 100 if total_tokens > 0 else 0.0
        feature_density[str(feature_idx)] = {
            "activation_count": activation_count,
            "total_tokens": total_tokens,
            "density_percent": density
        }

    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(feature_density, f, indent=2)

    print("Feature density calculation complete!")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Results saved to: {args.output}")


def process_batch(texts, model, sae, sae_layer, feature_activation_count):
    """Process a batch of texts and update feature activation counts."""
    tokens_processed = 0

    for text in texts:
        # Tokenize text
        input_tokens = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )["input_ids"]

        if input_tokens.shape[1] == 0:
            continue

        tokens_processed += input_tokens.shape[1]

        try:
            # Forward pass to get activations
            with torch.no_grad():
                _, _, feature_cache = model.forward(
                    input_tokens, cache_activations_at=[sae_layer]
                )

                activations = feature_cache[sae_layer]
                features = sae.encode(activations)

                # Count feature activations (features > 0)
                active_features = (features > 0).sum(dim=(0, 1))  # Sum over batch and sequence
                feature_activation_count += active_features.cpu()

        except Exception as e:
            print(f"Error processing text: {e}")
            continue

    return tokens_processed


if __name__ == "__main__":
    main()
