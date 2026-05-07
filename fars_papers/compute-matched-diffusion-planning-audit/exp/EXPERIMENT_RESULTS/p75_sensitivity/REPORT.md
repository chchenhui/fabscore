# p75 Wall-Clock Sensitivity Analysis

## Experiment Overview

This analysis assesses the robustness of the compute-matching conclusion to the choice of wall-clock time estimator used for computing k. The main experiment uses the **median** wall-clock time; this sensitivity check re-runs evaluation using k derived from the **75th percentile (p75)**, which yields a more conservative (lower) k for the AR model.

## Setup

- **Model**: Qwen2.5-7B (best-of-k sampling, temperature=1.2, top_p=0.95)
- **Tasks**: Countdown (500 instances), Mini Sudoku (500 instances)
- **Seeds**: 42, 123, 456 (3 seeds per condition)
- **k estimators**: median wall-clock time and p75 wall-clock time

### k Values

| Task | Dream p75 (s) | Qwen p75 (s) | k_median | k_p75 |
|------|--------------|--------------|----------|-------|
| Countdown | 47.71 | 1.37 | 35 | 34 |
| Mini Sudoku | 55.78 | 1.42 | 39 | 39 |

Note: For Mini Sudoku, k_p75 = k_median = 39, so existing results are reused directly.

## Key Results

| Task | k_median | Acc (median-k) | k_p75 | Acc (p75-k) | Dream Acc | Delta (median-k) | Delta (p75-k) |
|------|----------|----------------|-------|-------------|-----------|-------------------|---------------|
| Countdown | 35 | 39.1% +/- 1.4% | 34 | 38.9% +/- 1.4% | 6.6% | -32.5pp | -32.3pp |
| Mini Sudoku | 39 | 67.2% +/- 0.4% | 39 | 67.2% +/- 0.4% | 77.6% | +10.4pp | +10.4pp |

### Per-Seed Breakdown (Countdown p75, k=34)

| Seed | Accuracy |
|------|----------|
| 42 | 40.4% |
| 123 | 39.4% |
| 456 | 37.0% |

## Key Observations

1. **Countdown**: Switching from k_median=35 to k_p75=34 reduces k by only 1. Accuracy changes from 39.1% to 38.9% -- a negligible 0.2pp difference. The Dream-vs-Qwen gap remains large and negative (-32.3pp vs -32.5pp). Qwen best-of-k still massively outperforms Dream on this task.

2. **Mini Sudoku**: k_p75 = k_median = 39. Results are identical. Dream retains its +10.4pp advantage over compute-matched Qwen.

3. **Robustness**: Both estimators lead to the same qualitative conclusion on both tasks:
   - Countdown: Dream advantage absent (Qwen BoK wins by ~32pp)
   - Mini Sudoku: Dream advantage persists (~10pp)

4. **Conclusion**: The compute-matching finding is **robust** to the choice of wall-clock time estimator. The methodological choice between median and p75 does not alter the direction or meaningful magnitude of the Dream-vs-Qwen gap on either task.
