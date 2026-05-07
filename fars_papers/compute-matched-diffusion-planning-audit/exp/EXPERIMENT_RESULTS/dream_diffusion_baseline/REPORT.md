# Dream-v0-Base-7B Diffusion Baseline on Countdown and Mini Sudoku

## Experiment Overview

Evaluated Dream-v0-Base-7B (diffusion LM) with its recommended diffusion generation settings on procedurally generated Countdown and 4x4 Mini Sudoku instances from Reasoning Gym. This establishes Condition C -- the single-sample diffusion baseline. Also performed wall-clock timing calibration to compute the k value for the compute-matched best-of-k experiment.

## Setup

- **Model**: Dream-org/Dream-v0-Base-7B (7.62B params, diffusion LM)
- **Inference**: HF Transformers `diffusion_generate` with bfloat16
- **Generation params**: `alg='entropy'`, `alg_temp=0`, `temperature=0`, `top_p=1`, `max_new_tokens=64`, `diffusion_steps=64`
- **Prompting**: 8-shot few-shot prompts (same templates as Qwen baseline, exemplar seed=7777)
- **Hardware**: 1x A100-SXM4-80GB (via TrainService)

### Datasets
| Dataset | Size | Seed | Source |
|---------|------|------|--------|
| Countdown test | 500 | 2024 | reasoning_gym.create_dataset('countdown') |
| Countdown cal | 50 | 9999 | reasoning_gym.create_dataset('countdown') |
| Mini Sudoku test | 500 | 2024 | reasoning_gym.create_dataset('mini_sudoku') |
| Mini Sudoku cal | 50 | 9999 | reasoning_gym.create_dataset('mini_sudoku') |

### Evaluation
Binary accuracy using Reasoning Gym's `dataset.score_answer()` -- score=1.0 counted as correct, anything else as incorrect.

## Key Results

### Accuracy

| Task | Method | Accuracy | Correct | Total |
|------|--------|----------|---------|-------|
| Countdown | Dream diffusion | **6.60%** | 33 | 500 |
| Mini Sudoku | Dream diffusion | **77.60%** | 388 | 500 |

### Comparison with Qwen Greedy Baseline (Condition A)

| Task | Qwen Greedy | Dream Diffusion | Delta |
|------|-------------|-----------------|-------|
| Countdown | 6.0% | 6.6% | +0.6 |
| Mini Sudoku | 16.8% | 77.6% | +60.8 |

### Wall-Clock Timing Calibration

| Task | Dream median (s) | Qwen median (s) | k (median) | Dream p75 (s) | Qwen p75 (s) | k (p75) |
|------|-----------------|-----------------|------------|---------------|--------------|---------|
| Countdown | 47.68 | 1.36 | **35** | 47.71 | 1.37 | 34 |
| Mini Sudoku | 55.75 | 1.40 | **39** | 55.78 | 1.42 | 39 |

## Key Observations

1. Dream shows a massive advantage over Qwen on Mini Sudoku (77.6% vs 16.8%), a +60.8 percentage point gap. This is consistent with the Dream paper's claims about diffusion models excelling at planning tasks.
2. On Countdown, the advantage is minimal (6.6% vs 6.0%), suggesting both models struggle with exact arithmetic composition regardless of architecture.
3. Dream's diffusion generation is substantially slower than Qwen's autoregressive generation: ~35-39x slower per instance. This means the compute-matched best-of-k experiment will use k=35 for countdown and k=39 for sudoku.
4. Timing is extremely stable: Dream's per-instance time varies by less than 0.5s across 50 calibration instances per task.
5. The p75-based k values are nearly identical to median-based ones (34 vs 35 for countdown; 39 vs 39 for sudoku), suggesting timing robustness.
