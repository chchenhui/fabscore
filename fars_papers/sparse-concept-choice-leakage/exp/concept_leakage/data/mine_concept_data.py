"""Mine concept-specific sentence pairs (D_plus / D_minus) from PII-Masking-300K.
For each concept, scan source_text with lowercased regex, collect up to 10k matches,
split train/val/test at sentence level (70/15/15), then build D_minus by token deletion."""

import json
import os
import re
import random
from pathlib import Path
from datasets import load_dataset

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
MAX_PER_CONCEPT = 10_000
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN_DIR = BASE_DIR / "data" / "concept_tokens"
OUTPUT_DIR = BASE_DIR / "outputs" / "data"


def load_tokens(concept: str) -> list[str]:
    path = TOKEN_DIR / f"{concept}.txt"
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]


def build_regex(tokens: list[str]) -> re.Pattern:
    escaped = [re.escape(t) for t in sorted(tokens, key=len, reverse=True)]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def remove_tokens(text: str, regex: re.Pattern) -> str:
    cleaned = regex.sub("", text)
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


def split_sentences(sentences: list[str], ratios: tuple, rng: random.Random):
    n = len(sentences)
    shuffled = sentences.copy()
    rng.shuffle(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))
    return shuffled[:n_train], shuffled[n_train:n_val], shuffled[n_val:]


def mine_concept(dataset, concept: str):
    tokens = load_tokens(concept)
    regex = build_regex(tokens)

    matched = []
    for example in dataset:
        text = example.get("source_text", "")
        if not text:
            continue
        if regex.search(text.lower()):
            matched.append(text)
        if len(matched) >= MAX_PER_CONCEPT:
            break

    rng = random.Random(SEED)
    train_sents, val_sents, test_sents = split_sentences(matched, SPLIT_RATIOS, rng)

    out_dir = OUTPUT_DIR / concept
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sents in [("train", train_sents), ("val", val_sents), ("test", test_sents)]:
        records = []
        for s in sents:
            d_minus = remove_tokens(s, regex)
            records.append({"d_plus": s, "d_minus": d_minus})
        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  {concept}/{split_name}: {len(records)} pairs")

    return {
        "concept": concept,
        "total_matched": len(matched),
        "train": len(train_sents),
        "val": len(val_sents),
        "test": len(test_sents),
    }


def main():
    print("Loading PII-Masking-300K dataset...")
    ds = load_dataset("ai4privacy/pii-masking-300k")
    from datasets import concatenate_datasets
    splits = [ds[s] for s in ds.keys()]
    full_data = concatenate_datasets(splits)
    print(f"Dataset loaded: {len(full_data)} examples (all splits combined)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []

    for concept in CONCEPTS:
        print(f"\nMining concept: {concept}")
        s = mine_concept(full_data, concept)
        stats.append(s)

    print("\n=== Per-Concept Counts ===")
    for s in stats:
        print(f"  {s['concept']}: total={s['total_matched']}, "
              f"train={s['train']}, val={s['val']}, test={s['test']}")

    stats_path = OUTPUT_DIR / "mining_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
