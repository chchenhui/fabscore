# Scaling Curve Analysis: Qwen2.5-7B Best-of-k Accuracy vs. k

## Experiment Overview

Analyzed how Qwen2.5-7B best-of-j accuracy scales with increasing number of samples (j) on Countdown and Mini Sudoku tasks. Used existing k-sample outputs from the Main Experiment (optimized, temperature=1.2, 3 seeds) to compute best-of-j for j = 1, 2, 4, 8, 16, 32, and k_max.

## Setup

- **Data**: Reused `qwen_bok_opt_{task}_seed{s}.jsonl` files (no new inference)
- **k_max**: 35 (Countdown), 39 (Mini Sudoku)
- **Seeds**: 42, 123, 456
- **Method**: For each j, take first j samples per instance, mark solved if any sample scores 1.0

## Key Results

### Countdown (Dream = 6.6%, k_median = 35)

| j | Mean Acc | Std |
|---|----------|-----|
| 1 | 2.3% | 1.7% |
| 2 | 5.1% | 2.1% |
| 4 | 9.1% | 1.4% |
| 8 | 17.3% | 0.7% |
| 16 | 27.3% | 1.2% |
| 32 | 37.9% | 1.4% |
| 35 | 39.1% | 1.4% |

- Qwen crosses Dream at j~4 (10% of compute budget)
- No plateau; steep growth throughout
- At k=35, Qwen is 32.5pp above Dream

### Mini Sudoku (Dream = 77.6%, k_median = 39)

| j | Mean Acc | Std |
|---|----------|-----|
| 1 | 7.3% | 1.6% |
| 2 | 13.0% | 0.8% |
| 4 | 23.0% | 1.9% |
| 8 | 34.5% | 2.7% |
| 16 | 46.9% | 2.1% |
| 32 | 62.5% | 0.3% |
| 39 | 67.2% | 0.4% |

- Qwen does NOT reach Dream by k=39 (10.4pp gap remains)
- Curve decelerates but does not plateau
- Log-linear extrapolation estimates k~87 to match Dream (2.2x compute budget)

## Key Observations

1. **Task-dependent scaling**: On Countdown, AR scaling is extremely efficient and Dream offers no advantage. On Sudoku, AR scaling is steady but slower relative to the task difficulty.

2. **No hard plateau on either task**: Both curves show continued improvement at k_max, indicating more samples would further increase accuracy. The diffusion advantage on Sudoku is a compute-efficiency gap, not a fundamental capability ceiling.

3. **Feasible extrapolation**: For Sudoku, matching Dream would require approximately k=87 samples (2.2x the compute-matched budget). This is technically feasible but represents a meaningful compute overhead, confirming that Dream has a genuine efficiency advantage on constraint-satisfaction tasks.

4. **Diminishing returns visible on Sudoku**: Marginal gains decrease from +15.6pp (j=16 to 32) to +4.7pp (j=32 to 39), consistent with standard best-of-k scaling behavior where each doubling yields smaller absolute gains.

## Artifacts

- Figures: `audit/results/figures/scaling_countdown.png`, `audit/results/figures/scaling_sudoku.png`
- Analysis notes: `audit/results/tables/scaling_analysis_notes.txt`
- Summary data: `audit/results/tables/scaling_summary.json`
- Script: `audit/analysis/plot_scaling.py`
