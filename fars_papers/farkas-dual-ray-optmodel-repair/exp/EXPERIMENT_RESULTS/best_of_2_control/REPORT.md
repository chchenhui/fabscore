# Best-of-2 Inference Scaling Control Experiment

## Experiment Overview

Compute-matched inference-time scaling baseline: generate two stochastic samples (temperature=0.7, top_p=0.95) for each MAMO instance and select the first solver-correct one. This controls for the possibility that a second inference call alone (without solver diagnostic feedback) could improve pass@1, ensuring any gains from repair methods are attributable to feedback content rather than simply having a second chance.

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Sampling**: temperature=0.7, top_p=0.95, max_tokens=4096, n=2
- **Seeds**: [42, 123, 456] (3 runs for mean +/- std)
- **Benchmark**: MAMO-Optimization (863 instances: 652 EasyLP + 211 ComplexLP)
- **Selection**: For each instance, pick the first of 2 samples that passes solver verification (correct objective within tolerance). If neither passes, instance is a failure.
- **GPU**: 1x GPU via TrainService, vLLM offline inference
- **Runtime**: ~63 minutes total (3 seeds sequential)

## Key Results

| Metric | Attempt-0 (greedy) | Best-of-2 (mean +/- std) | Delta |
|--------|-------------------|--------------------------|-------|
| Overall pass@1 | 58.05% (501/863) | 65.12% +/- 0.66% | +7.07pp |
| EasyLP pass@1 | 71.17% (464/652) | 79.50% +/- 0.51% | +8.33pp |
| ComplexLP pass@1 | 17.54% (37/211) | 20.70% +/- 1.56% | +3.16pp |

### Per-Seed Breakdown

| Seed | Overall | EasyLP | ComplexLP |
|------|---------|--------|-----------|
| 42 | 64.54% (557/863) | 78.83% (514/652) | 20.38% (43/211) |
| 123 | 66.05% (570/863) | 80.06% (522/652) | 22.75% (48/211) |
| 456 | 64.77% (559/863) | 79.60% (519/652) | 18.96% (40/211) |

## Key Observations

1. **Best-of-2 provides a meaningful gain over greedy**: +7.07pp overall, confirming that stochastic sampling with oracle selection improves pass@1 substantially.
2. **EasyLP benefits more than ComplexLP**: +8.33pp vs +3.16pp. ComplexLP problems are harder and even with two tries, the model rarely produces correct solutions.
3. **Low variance across seeds**: std < 1.6% on all splits, indicating stable results.
4. **This sets the bar for repair methods**: Any repair method (IIS-TopK, DualRay-TopK, etc.) must outperform 65.12% overall pass@1 to demonstrate that solver feedback adds value beyond simply having a second generation attempt.
5. **Infeasible rates remain non-trivial**: Even with best-of-2, 28-37 instances per seed remain infeasible (vs 31 in greedy), confirming the infeasibility diagnosis and repair pipeline remains relevant.
