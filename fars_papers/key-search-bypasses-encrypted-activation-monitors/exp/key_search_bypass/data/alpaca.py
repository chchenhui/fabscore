# Alpaca harmless prompt loader: loads tatsu-lab/alpaca via HF datasets,
# tokenizes instruction field for Qwen2.5-7B-Instruct. Provides full pool
# (52k for FPR calibration) and balanced subset (matching HarmBench size).

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 128


def _get_texts(dataset):
    return [ex["instruction"] for ex in dataset]


def load_alpaca_full(tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = _get_texts(ds)
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    labels = np.zeros(len(texts), dtype=np.int64)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": texts,
    }


def load_alpaca_subset(n, seed=42, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = _get_texts(ds)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(texts), size=n, replace=False)
    subset_texts = [texts[i] for i in indices]
    encoded = tokenizer(
        subset_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    labels = np.zeros(len(subset_texts), dtype=np.int64)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": subset_texts,
        "indices": indices,
    }
