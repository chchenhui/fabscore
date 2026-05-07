# HarmBench harmful prompt loader: loads CSV files from harmbench_raw/,
# excludes Copyright Violation category, tokenizes for Qwen2.5-7B-Instruct.
# Returns dicts with input_ids, attention_mask, labels.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent / "harmbench_raw"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 128


def load_harmbench_texts(csv_name="harmbench_behaviors_text_all.csv"):
    df = pd.read_csv(DATA_DIR / csv_name)
    df = df[df["FunctionalCategory"] != "copyright"]
    texts = df["Behavior"].tolist()
    return texts


def load_harmbench_tokenized(tokenizer=None, csv_name="harmbench_behaviors_text_all.csv"):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = load_harmbench_texts(csv_name)
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    labels = np.ones(len(texts), dtype=np.int64)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "texts": texts,
    }
