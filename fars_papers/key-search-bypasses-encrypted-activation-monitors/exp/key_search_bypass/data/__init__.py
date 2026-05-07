# Data package: provides build_balanced_dataset() for combined HarmBench+Alpaca
# with stratified 80/20 train/test split.

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .harmbench import load_harmbench_tokenized
from .alpaca import load_alpaca_subset, load_alpaca_full

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def build_balanced_dataset(tokenizer=None, test_size=0.2, seed=42):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    harm = load_harmbench_tokenized(tokenizer=tokenizer)
    n_harm = len(harm["labels"])
    safe = load_alpaca_subset(n=n_harm, seed=seed, tokenizer=tokenizer)

    input_ids = np.concatenate([harm["input_ids"], safe["input_ids"]], axis=0)
    attention_mask = np.concatenate([harm["attention_mask"], safe["attention_mask"]], axis=0)
    labels = np.concatenate([harm["labels"], safe["labels"]], axis=0)

    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    def _subset(indices):
        return {
            "input_ids": input_ids[indices],
            "attention_mask": attention_mask[indices],
            "labels": labels[indices],
        }

    return _subset(idx_train), _subset(idx_test)
