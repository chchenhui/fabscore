"""Compute bootstrap 95% CIs for the accuracy difference: Dream (Cond C) minus
Qwen best-of-k optimized (Cond B) on Countdown and Mini Sudoku.
Uses per-instance binary scores from JSONL files. B=10000 resamples of 500 instances.
Outputs results/tables/bootstrap_ci.csv.
"""

import csv
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

TASKS = {
    "countdown": {
        "dream_file": "dream_diffusion_countdown_scored.jsonl",
        "bok_files": [
            "qwen_bok_opt_countdown_seed42.jsonl",
            "qwen_bok_opt_countdown_seed123.jsonl",
            "qwen_bok_opt_countdown_seed456.jsonl",
        ],
    },
    "mini_sudoku": {
        "dream_file": "dream_diffusion_sudoku_scored.jsonl",
        "bok_files": [
            "qwen_bok_opt_sudoku_seed42.jsonl",
            "qwen_bok_opt_sudoku_seed123.jsonl",
            "qwen_bok_opt_sudoku_seed456.jsonl",
        ],
    },
}

B = 10000
N = 500
RNG_SEED = 2024


def load_dream_scores(filename):
    path = os.path.join(RAW_DIR, filename)
    scores = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            scores.append(float(entry["binary"]))
    assert len(scores) == N, f"Expected {N} instances, got {len(scores)}"
    return np.array(scores)


def load_bok_scores(filename):
    path = os.path.join(RAW_DIR, filename)
    scores = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            scores.append(1.0 if entry["solved"] else 0.0)
    assert len(scores) == N, f"Expected {N} instances, got {len(scores)}"
    return np.array(scores)


def bootstrap_delta(dream_scores, bok_scores_list, rng):
    bok_mean = np.mean(bok_scores_list, axis=0)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        dream_acc = dream_scores[idx].mean()
        bok_acc = bok_mean[idx].mean()
        deltas[b] = dream_acc - bok_acc
    return deltas


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for task, files in TASKS.items():
        dream = load_dream_scores(files["dream_file"])
        bok_list = [load_bok_scores(f) for f in files["bok_files"]]

        deltas = bootstrap_delta(dream, bok_list, rng)
        delta_mean = deltas.mean()
        ci_lower = np.percentile(deltas, 2.5)
        ci_upper = np.percentile(deltas, 97.5)

        print(f"{task}:")
        print(f"  Dream acc:     {dream.mean():.4f}")
        print(f"  BoK mean acc:  {np.mean([s.mean() for s in bok_list]):.4f}")
        print(f"  Delta mean:    {delta_mean:+.4f}")
        print(f"  95% CI:        [{ci_lower:+.4f}, {ci_upper:+.4f}]")
        ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
        print(f"  CI excludes 0: {ci_excludes_zero}")

        results.append({
            "task": task,
            "delta_mean": f"{delta_mean:.4f}",
            "ci_lower": f"{ci_lower:.4f}",
            "ci_upper": f"{ci_upper:.4f}",
        })

    out_path = os.path.join(TABLES_DIR, "bootstrap_ci.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "delta_mean", "ci_lower", "ci_upper"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
