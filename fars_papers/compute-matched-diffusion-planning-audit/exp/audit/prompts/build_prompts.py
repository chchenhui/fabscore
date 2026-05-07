"""Build 8-shot prompt templates for Countdown and Mini Sudoku.
Exemplars drawn from Reasoning Gym with seed=7777 to avoid overlap with test (seed=2024) and cal (seed=9999).
"""

import os
import reasoning_gym

OUTPUT_DIR = os.path.dirname(__file__)
EXEMPLAR_SEED = 7777
N_SHOTS = 8


def build_countdown_prompt():
    ds = reasoning_gym.create_dataset("countdown", seed=EXEMPLAR_SEED, size=N_SHOTS)
    lines = []
    for entry in ds:
        score = ds.score_answer(answer=entry["answer"], entry=entry)
        assert score == 1.0, f"Exemplar answer failed verification: {entry['answer']}"
        lines.append(entry["question"].strip())
        lines.append(f"Output: {entry['answer']}")
        lines.append("")
    template = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "countdown_8shot.txt")
    with open(path, "w") as f:
        f.write(template)
    print(f"Wrote countdown 8-shot template to {path}")
    return template


def build_sudoku_prompt():
    ds = reasoning_gym.create_dataset("mini_sudoku", seed=EXEMPLAR_SEED, size=N_SHOTS)
    lines = []
    for entry in ds:
        score = ds.score_answer(answer=entry["answer"], entry=entry)
        assert score == 1.0, f"Exemplar answer failed verification: {entry['answer']}"
        lines.append(entry["question"].strip())
        lines.append(f"Output: {entry['answer']}")
        lines.append("")
    template = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "sudoku_8shot.txt")
    with open(path, "w") as f:
        f.write(template)
    print(f"Wrote sudoku 8-shot template to {path}")
    return template


if __name__ == "__main__":
    build_countdown_prompt()
    build_sudoku_prompt()
    print("All prompt templates built successfully.")
