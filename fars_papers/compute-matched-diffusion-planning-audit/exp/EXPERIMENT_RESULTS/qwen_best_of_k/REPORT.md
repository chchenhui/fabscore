# Qwen2.5-7B Best-of-k Compute-Matched Evaluation (Condition B, Optimized)

## Experiment Overview

Evaluated Qwen2.5-7B with best-of-k sampling where k is determined by wall-clock compute matching to Dream's diffusion generation time. This is the core experiment: testing whether AR models with compute-matched inference-time scaling can close the planning gap reported for diffusion models.

## Setup

- **Model**: Qwen/Qwen2.5-7B (7.62B params, autoregressive)
- **Inference**: vLLM with `n=k` batched sampling (bfloat16)
- **Generation params**: `temperature=1.2`, `top_p=0.95`, `max_tokens=64`, `stop=["\n\n"]`
- **Prompting**: 8-shot few-shot prompts (same templates as greedy baseline, exemplar seed=7777)
- **Seeds**: 42, 123, 456 (3 independent runs for variance)
- **Hardware**: 1x A100-SXM4-80GB (via TrainService)
- **Selection**: Best-of-k (instance solved if ANY of k samples passes verifier)
- **No early stopping**: all k samples always generated

### Optimization from Original
- Temperature increased from 0.8 to 1.2 (higher diversity, more coverage of solution space)
- Added `stop=["\n\n"]` for sudoku (was missing, causing wasted tokens on continuation text)
- Hyperparameter sweep on 50-instance calibration set confirmed temp=1.2 as optimal

### Compute Matching
| Task | Dream median (s) | Qwen median (s) | k |
|------|-----------------|-----------------|---|
| Countdown | 47.68 | 1.36 | 35 |
| Mini Sudoku | 55.75 | 1.40 | 39 |

## Key Results

### Best-of-k Accuracy (Condition B, Optimized)

| Task | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|------|---------|----------|----------|------|-----|
| Countdown (k=35) | 39.20% | 40.80% | 37.40% | **39.13%** | 1.39% |
| Mini Sudoku (k=39) | 67.80% | 67.00% | 66.80% | **67.20%** | 0.43% |

### Full Comparison (All Three Conditions)

| Method | Task | Accuracy | k |
|--------|------|----------|---|
| Greedy (Cond. A) | Countdown | 6.00% | 1 |
| **Best-of-k (Cond. B)** | Countdown | **39.13% +/- 1.39%** | 35 |
| Diffusion (Cond. C) | Countdown | 6.60% | 1 |
| Greedy (Cond. A) | Mini Sudoku | 16.80% | 1 |
| **Best-of-k (Cond. B)** | Mini Sudoku | **67.20% +/- 0.43%** | 39 |
| Diffusion (Cond. C) | Mini Sudoku | 77.60% | 1 |

## Key Observations

1. **Countdown**: Best-of-k (39.13%) massively outperforms both greedy (6.0%) and Dream diffusion (6.6%). The compute-matched AR model is +32.53pp above the diffusion model. This strongly refutes any diffusion advantage on Countdown.

2. **Mini Sudoku**: Best-of-k (67.20%) substantially closes the gap from greedy (16.8%) toward Dream (77.6%). Dream retains a 10.40pp advantage with compute matching. This suggests a genuine but moderate diffusion planning advantage on structured constraint satisfaction.

3. **Optimization impact**: Increasing temperature from 0.8 to 1.2 improved sudoku accuracy by +1.07pp (66.13% -> 67.20%) through increased sample diversity (23.5 -> 26.9 unique out of 39). Countdown was essentially unchanged.

4. **Inference was fast**: vLLM's n=k batched generation processed 500 instances x 35-39 samples efficiently.

5. **Summary**: The diffusion planning advantage is task-dependent. On Countdown (arithmetic composition), compute-matched AR *exceeds* diffusion by a wide margin. On Mini Sudoku (constraint satisfaction), diffusion retains a moderate edge (~10pp), but AR closes ~83% of the greedy-to-diffusion gap with compute matching.
