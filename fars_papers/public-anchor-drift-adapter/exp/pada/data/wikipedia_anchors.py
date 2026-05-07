"""Wikipedia public anchor corpus loader.

Loads English Wikipedia paragraphs from HuggingFace datasets (wikimedia/wikipedia
20231101.en) via streaming, splits articles into paragraphs, filters short ones,
and provides reproducible random sampling.
"""

import random

from datasets import load_dataset


def _iter_paragraphs(dataset):
    for article in dataset:
        text = article["text"]
        for para in text.split("\n"):
            para = para.strip()
            if len(para) >= 50:
                yield para


def load_wikipedia_paragraphs(cache_dir: str | None = None, max_paragraphs: int | None = None):
    pool_size = max_paragraphs or 50000
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        split="train", streaming=True,
    )
    paragraphs = []
    for para in _iter_paragraphs(ds):
        paragraphs.append(para)
        if len(paragraphs) >= pool_size:
            break
    return paragraphs


def sample_anchors(n: int, seed: int, paragraphs: list[str] | None = None, cache_dir: str | None = None):
    if paragraphs is None:
        paragraphs = load_wikipedia_paragraphs(cache_dir=cache_dir)
    rng = random.Random(seed)
    if n >= len(paragraphs):
        return paragraphs
    return rng.sample(paragraphs, n)
