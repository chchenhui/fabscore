"""Load and merge MAMO optimization benchmark data (EasyLP + ComplexLP)."""

import json
from pathlib import Path
from typing import Optional

MAMO_REPO_DIR = Path(__file__).resolve().parent / "mamo_repo"
EASY_LP_PATH = MAMO_REPO_DIR / "data" / "optimization" / "Easy_LP" / "mamo_easy_lp.jsonl"
COMPLEX_LP_PATH = MAMO_REPO_DIR / "data" / "optimization" / "Complex_LP" / "mamo_complex_lp.jsonl"


def _load_jsonl(path: Path, difficulty: str) -> list[dict]:
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            item["difficulty"] = difficulty
            instances.append(item)
    return instances


def load_mamo(
    difficulty: Optional[str] = None,
) -> list[dict]:
    """Load MAMO optimization instances.

    Args:
        difficulty: Filter by difficulty level ("EasyLP", "ComplexLP", or None for all).

    Returns:
        List of dicts with keys: id, Question, Answer, Category, Type, difficulty.
    """
    easy = _load_jsonl(EASY_LP_PATH, "EasyLP")
    complex_ = _load_jsonl(COMPLEX_LP_PATH, "ComplexLP")
    all_instances = easy + complex_

    if difficulty is not None:
        all_instances = [x for x in all_instances if x["difficulty"] == difficulty]

    return all_instances


if __name__ == "__main__":
    data = load_mamo()
    print(f"Total instances: {len(data)}")
    easy = [x for x in data if x["difficulty"] == "EasyLP"]
    complex_ = [x for x in data if x["difficulty"] == "ComplexLP"]
    print(f"  EasyLP: {len(easy)}, ComplexLP: {len(complex_)}")
    print(f"  Sample keys: {list(data[0].keys())}")
    print(f"  First instance id={data[0]['id']}, difficulty={data[0]['difficulty']}")
