# Script to extract activations from Qwen2.5-7B-Instruct for all candidate
# layers (21-27) on balanced train/test sets and full Alpaca pool.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from key_search_bypass.data import build_balanced_dataset
from key_search_bypass.data.alpaca import load_alpaca_full
from key_search_bypass.monitor.extract import run_extraction_pipeline
from transformers import AutoTokenizer

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
OUTPUT_DIR = os.path.join(PROJ, "key_search_bypass", "outputs", "activations")
LAYERS = list(range(21, 28))
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Building balanced dataset...")
    train_data, test_data = build_balanced_dataset(tokenizer=tokenizer)
    print(f"  Train: {len(train_data['labels'])} samples")
    print(f"  Test: {len(test_data['labels'])} samples")

    print("Loading full Alpaca pool...")
    alpaca_full = load_alpaca_full(tokenizer=tokenizer)
    print(f"  Alpaca full: {len(alpaca_full['labels'])} samples")

    print(f"\nExtracting layers {LAYERS} to {OUTPUT_DIR}")
    run_extraction_pipeline(
        train_data=train_data,
        test_data=test_data,
        alpaca_full_data=alpaca_full,
        layers=LAYERS,
        output_dir=OUTPUT_DIR,
        batch_size=64,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
