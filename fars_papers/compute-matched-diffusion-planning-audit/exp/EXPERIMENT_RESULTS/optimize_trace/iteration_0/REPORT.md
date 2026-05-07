# Optimized Best-of-k Compute-Matched Evaluation (Iteration 0)

## Experiment Overview

Optimized the Qwen2.5-7B best-of-k sampling parameters to improve accuracy on both Countdown and Mini Sudoku. The optimization focused on increasing sample diversity through higher temperature and fixing a missing stop sequence for sudoku.

## Changes from Original

| Parameter | Original | Optimized |
|-----------|----------|-----------|
| Temperature | 0.8 | 1.2 |
| Stop sequence (sudoku) | None | `\n\n` |
| Stop sequence (countdown) | `\n\n` | `\n\n` (unchanged) |
| presence_penalty | 0.0 | 0.0 |
| top_p | 0.95 | 0.95 (unchanged) |
| k (countdown) | 35 | 35 (unchanged) |
| k (sudoku) | 39 | 39 (unchanged) |

## Hyperparameter Sweep (Calibration Set, 50 instances)

| Task | Temp | PP | Accuracy |
|------|------|-----|----------|
| Countdown | 0.8 | 0.0 | 36.0% |
| Countdown | 1.0 | 0.0 | 40.0% |
| Countdown | 1.2 | 0.0 | 44.0% |
| Countdown | 1.0 | 0.3 | 40.0% |
| Countdown | 1.2 | 0.3 | 46.0% |
| Sudoku | 0.8 | 0.0 | 60.0% |
| Sudoku | 1.0 | 0.0 | 64.0% |
| Sudoku | 1.2 | 0.0 | 68.0% |
| Sudoku | 1.0 | 0.3 | 64.0% |
| Sudoku | 1.2 | 0.3 | 66.0% |

Selected: temp=1.2, presence_penalty=0.0 (best overall improvement on both tasks).

## Full Results (500 instances, 3 seeds)

### Per-Seed Accuracy

| Task | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|------|---------|----------|----------|------|-----|
| Countdown (k=35) | 39.20% | 40.80% | 37.40% | **39.13%** | 1.39% |
| Sudoku (k=39) | 67.80% | 67.00% | 66.80% | **67.20%** | 0.43% |

### Comparison with Original

| Task | Original | Optimized | Delta |
|------|----------|-----------|-------|
| Countdown | 39.33% +/- 0.81% | 39.13% +/- 1.39% | -0.20pp |
| Mini Sudoku | 66.13% +/- 0.09% | 67.20% +/- 0.43% | **+1.07pp** |

### Sample Diversity

| Task | Original Unique | Optimized Unique |
|------|----------------|-----------------|
| Countdown | 31.3/35 | 33.4/35 |
| Sudoku | 23.5/39 | 26.9/39 |

## Updated Full Comparison

| Method | Task | Accuracy | k |
|--------|------|----------|---|
| Greedy (Cond. A) | Countdown | 6.00% | 1 |
| Best-of-k orig (Cond. B) | Countdown | 39.33% +/- 0.81% | 35 |
| **Best-of-k opt (Cond. B')** | Countdown | **39.13% +/- 1.39%** | 35 |
| Diffusion (Cond. C) | Countdown | 6.60% | 1 |
| Greedy (Cond. A) | Mini Sudoku | 16.80% | 1 |
| Best-of-k orig (Cond. B) | Mini Sudoku | 66.13% +/- 0.09% | 39 |
| **Best-of-k opt (Cond. B')** | Mini Sudoku | **67.20% +/- 0.43%** | 39 |
| Diffusion (Cond. C) | Mini Sudoku | 77.60% | 1 |

## Key Observations

1. Sudoku improved by +1.07pp (66.13% -> 67.20%), narrowing the gap with Dream from 11.47pp to 10.40pp.
2. Countdown was essentially unchanged (-0.20pp, within noise). Higher diversity doesn't help when the model can't compute correct arithmetic.
3. Sample diversity increased substantially on both tasks (sudoku: 23.5 -> 26.9 unique samples out of 39).
4. The improvement is modest but consistent: all 3 sudoku seeds improved over all 3 original seeds.
5. The core conclusion remains: diffusion advantage is task-dependent. AR with compute-matched BoK dominates on Countdown, while Dream retains a ~10pp edge on Sudoku.
