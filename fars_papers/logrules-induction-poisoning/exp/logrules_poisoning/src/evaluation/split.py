"""Data splitting utility for LogRules poisoning experiment.
Splits each dataset into induction (K=10), canary (50), and test (~1940) sets
for each random seed. Ensures no overlap between splits.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]
K_INDUCTION = 10
K_CANARY = 50

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"


def load_dataset(dataset: str) -> List[Dict]:
    path = PROCESSED_DIR / f"{dataset}.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def create_split(dataset: str, seed: int) -> Dict[str, List[Dict]]:
    records = load_dataset(dataset)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    induction_idx = indices[:K_INDUCTION]
    canary_idx = indices[K_INDUCTION:K_INDUCTION + K_CANARY]
    test_idx = indices[K_INDUCTION + K_CANARY:]

    return {
        "induction": [records[i] for i in induction_idx],
        "canary": [records[i] for i in canary_idx],
        "test": [records[i] for i in test_idx],
    }


def save_split(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_all_splits():
    for dataset in DATASETS:
        for seed in SEEDS:
            split = create_split(dataset, seed)
            out_dir = SPLITS_DIR / dataset / f"seed_{seed}"
            for name, records in split.items():
                save_split(records, out_dir / f"{name}.jsonl")
            print(
                f"{dataset}/seed_{seed}: "
                f"induction={len(split['induction'])}, "
                f"canary={len(split['canary'])}, "
                f"test={len(split['test'])}"
            )


if __name__ == "__main__":
    generate_all_splits()
