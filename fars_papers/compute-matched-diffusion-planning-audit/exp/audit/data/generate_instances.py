"""Generate test and calibration datasets for Countdown and Mini Sudoku using Reasoning Gym.
Produces 4 JSONL files with fixed RNG seeds for reproducibility.
"""

import json
import os
import reasoning_gym

CONFIGS = [
    ("countdown", 500, 2024, "countdown_test.jsonl"),
    ("countdown", 50, 9999, "countdown_cal.jsonl"),
    ("mini_sudoku", 500, 2024, "sudoku_test.jsonl"),
    ("mini_sudoku", 50, 9999, "sudoku_cal.jsonl"),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))


def generate_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task_name, size, seed, filename in CONFIGS:
        ds = reasoning_gym.create_dataset(task_name, seed=seed, size=size)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            for idx, entry in enumerate(ds):
                record = {
                    "id": idx,
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "metadata": entry["metadata"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {size} instances to {path}")


if __name__ == "__main__":
    generate_all()
