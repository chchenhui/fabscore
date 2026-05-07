"""
Sample BFCL-v3 Non-Live single-turn subset for evaluation.
Follows Bitter Lesson sampling protocol: 50 examples per category with seed 42.
Uses all 7 Non-Live categories from BFCL's NON_LIVE_CATEGORY list.
"""
import json
import random
from pathlib import Path

BFCL_DATA_DIR = Path(__file__).resolve().parents[2] / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
POSSIBLE_ANSWER_DIR = BFCL_DATA_DIR / "possible_answer"
OUTPUT_PATH = Path(__file__).resolve().parent / "bfcl_nonlive_300.json"

NON_LIVE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

SAMPLES_PER_CATEGORY = 50
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    random.seed(SEED)
    all_samples = []

    for cat in NON_LIVE_CATEGORIES:
        data_path = BFCL_DATA_DIR / f"BFCL_v4_{cat}.json"
        examples = load_jsonl(data_path)
        print(f"{cat}: {len(examples)} available")

        if len(examples) <= SAMPLES_PER_CATEGORY:
            sampled = examples
        else:
            sampled = random.sample(examples, SAMPLES_PER_CATEGORY)

        gt_path = POSSIBLE_ANSWER_DIR / f"BFCL_v4_{cat}.json"
        gt_map = {}
        if gt_path.exists():
            gt_entries = load_jsonl(gt_path)
            gt_map = {e["id"]: e.get("ground_truth", []) for e in gt_entries}

        for ex in sampled:
            eid = ex["id"]
            question_turns = ex["question"]
            user_content = question_turns[0][0]["content"] if question_turns else ""
            functions = ex.get("function", [])
            ground_truth = gt_map.get(eid, [])

            all_samples.append({
                "id": eid,
                "category": cat,
                "question": user_content,
                "question_raw": question_turns,
                "functions": functions,
                "ground_truth": ground_truth,
            })

    print(f"\nTotal sampled: {len(all_samples)}")
    for cat in NON_LIVE_CATEGORIES:
        cat_samples = [s for s in all_samples if s["category"] == cat]
        n_empty_gt = sum(1 for s in cat_samples if s["ground_truth"] == [])
        print(f"  {cat}: {len(cat_samples)} samples, {n_empty_gt} with empty ground_truth")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
