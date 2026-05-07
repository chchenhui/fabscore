# Effectiveness Evaluation Report

## Verdict: bad

## Summary

The DualRayRank method -- ranking IIS constraints by Farkas dual ray multipliers to provide better repair feedback to LLMs -- does not outperform the IIS-TopK baseline under controlled experimental conditions. The primary hypothesis that dual-ray ranking helps in the truncation regime (where IIS size > K) is refuted: both methods achieve 0/16 repair success on truncated instances. All repair methods are dominated by simple Best-of-2 inference scaling by approximately 6 percentage points. The approach needs to be reconsidered.

## Experiment Feasibility Check

All experiments ran successfully and produced results:

- **Attempt-0 (no repair)**: 863 instances evaluated, 501 passed (58.05%), 31 infeasible identified.
- **Best-of-2**: 3 seeds completed, mean Pass@1 = 65.12% (std = 0.66%).
- **IIS-TopK (A)**: 31 infeasible instances processed, 1/31 repaired.
- **DualRay-TopK (B)**: 31 infeasible instances processed, 1/31 repaired.
- **DualRay+Weights (C)**: 31 infeasible instances processed, 1/31 repaired.
- **Optimized C+ (7B)**: K=10, best-of-6, 3/31 repaired.
- **Optimized C+ (72B)**: K=10, N=16, 2 rounds, 7/31 repaired.
- **Optimized C+ (32B)**: K=10, N=16, 2 rounds, 6/31 repaired.

No infrastructure or environment failures. All conditions produced comparable results.

## Results Analysis

### Controlled Comparison (Primary): Same Model, Same K, Greedy

| Condition | Model | K | Decoding | Repair (31) | Truncation (16) | Pass@1 |
|-----------|-------|---|----------|-------------|-----------------|--------|
| IIS-TopK (A) | 7B | 5 | greedy | 1 (3.23%) | 0 (0.0%) | 58.17% |
| DualRay-TopK (B) | 7B | 5 | greedy | 1 (3.23%) | 0 (0.0%) | 58.17% |
| DualRay+Wt (C) | 7B | 5 | greedy | 1 (3.23%) | 0 (0.0%) | 58.17% |

All three conditions produce identical results. The single repaired instance (EasyLP_425, IIS size=2, not truncated) is the same across all methods.

### Extended Comparison: Optimized Settings

| Condition | Model | K | N | Rounds | Repair (31) | Truncation (16) | Pass@1 |
|-----------|-------|---|---|--------|-------------|-----------------|--------|
| IIS K=10 bo6 | 7B | 10 | 6 | 1 | 2 (6.45%) | 0 (0.0%) | 58.29% |
| DualRay+Wt K=10 bo6 | 7B | 10 | 6 | 1 | 3 (9.68%) | 0 (0.0%) | 58.40% |
| DualRay+Wt K=10 N=16 R=2 | 72B | 10 | 16 | 2 | 7 (22.58%) | 1 (6.25%) | 58.86% |
| DualRay+Wt K=10 N=16 R=2 | 32B | 10 | 16 | 2 | 6 (19.35%) | 0 (0.0%) | 58.75% |

### Inference Scaling Baseline

| Method | Model | Samples | Pass@1 Overall |
|--------|-------|---------|---------------|
| Attempt-0 | 7B | 1 | 58.05% |
| Best-of-2 | 7B | 2 | **65.12%** |
| Best repair (C+ 72B) | 72B | 34 | 58.86% |

Best-of-2 outperforms all repair methods by ~6pp while using a weaker model and fewer samples.

### Per-Difficulty Breakdown

| Condition | EasyLP (652) | ComplexLP (211) |
|-----------|-------------|----------------|
| Attempt-0 | 71.17% | 17.54% |
| Best-of-2 | 79.50% | 20.70% |
| IIS-TopK (A) 7B | 71.32% (+0.15pp) | 17.54% (+0.00pp) |
| DualRay-TopK (B) 7B | 71.32% (+0.15pp) | 17.54% (+0.00pp) |
| C+ 72B optimized | 71.78% (+0.61pp) | 18.96% (+1.42pp) |

### Truncation Regime Analysis

- 16 out of 31 infeasible instances have IIS size > K=5 (truncation regime)
- All 16 are ComplexLP instances
- IIS sizes in truncation regime: min=6, max=33, mean=10.9
- Under controlled conditions: 0/16 repaired by any method
- Under best optimized conditions (72B): 1/16 repaired (ComplexLP_86, IIS size=6)
- The truncation regime count (16) fails the 50-instance practical relevance gate

### Successfully Repaired Instances (Best Configuration: 72B)

| Instance | IIS Size | Truncated? | Round | Notes |
|----------|----------|-----------|-------|-------|
| EasyLP_246 | 2 | No | 1 | 17/17 pass rate with 72B; 0/6 with 7B |
| EasyLP_425 | 2 | No | 1 | Repaired by all methods |
| EasyLP_620 | 4 | No | 1 | DualRay-unique repair (0/6 for IIS) |
| EasyLP_645 | 1 | No | 1 | 6/17 pass rate |
| ComplexLP_62 | 4 | No | 1 | First ComplexLP repair |
| ComplexLP_111 | 1 | No | 2 | Iterative repair success |
| ComplexLP_86 | 6 | **Yes** | 2 | Only truncation regime repair |

6 of 7 repaired instances have small IIS (<=4), indicating the repair signal is useful primarily when IIS already fits within K.

## Statistical Significance

Given the small sample sizes (31 infeasible, 16 truncation regime), formal statistical tests have limited power:

- **B vs A (controlled)**: 1/31 vs 1/31. No difference to test.
- **C+ 7B vs A+ 7B (K=10, bo6)**: 3/31 vs 2/31. Fisher's exact test p-value >> 0.05. Not significant.
- **C+ 72B vs A (repair rate)**: 7/31 vs 1/31. This comparison is confounded with model size (72B vs 7B), K (10 vs 5), N (16 vs 1), and rounds (2 vs 1), so it does not isolate the feedback method.
- **Best repair vs Best-of-2**: 58.86% vs 65.12% on 863 instances. The gap (-6.26pp) is large and practically significant regardless of statistical test, especially since Best-of-2 uses a weaker model.

The fundamental constraint is that only 31 instances are infeasible (3.59% of benchmark), limiting the statistical power of any repair comparison.

## Verdict Justification

### Evidence for "bad" verdict:

1. **Primary hypothesis refuted (B vs A on truncation regime)**: Under controlled conditions, DualRay-TopK produces identical results to IIS-TopK. Both achieve 0/16 on the truncation regime and repair the same single instance. There is no evidence that dual-ray ranking provides better constraint selection than alphabetical IIS truncation.

2. **Practical relevance gate failed**: Only 16 instances in the truncation regime, far below the 50-instance threshold. Even perfect repair of all truncated instances would yield at most +1.85pp end-to-end improvement.

3. **Repair dominated by inference scaling**: Best-of-2 achieves 65.12% using the same 7B model with 2 samples, while the best repair configuration (72B, 34 samples, solver overhead) achieves only 58.86%. Solver feedback repair is not competitive with simple re-sampling on this benchmark.

4. **Optimization confounds**: The improvements from optimization (7/31 with 72B) are primarily attributable to model capacity (72B >> 7B), not the feedback signal. This is evidenced by EasyLP_246 going from 0/6 pass (7B) to 17/17 pass (72B) regardless of feedback method.

5. **Weights add nothing under controlled conditions**: C = B = A = 1/31 under K=5 greedy. The Farkas multiplier weights do not help the 7B model repair infeasible instances.

### What the method does achieve (mitigating factors):

- Under optimized conditions, the full pipeline (DualRay + Weights + 72B + iterative) achieves 7/31 (22.58%) repair rate, which is a meaningful improvement over 1/31.
- Three ComplexLP instances were repaired for the first time, demonstrating that the repair concept has some value with sufficient model capacity.
- EasyLP_620 appears to be a genuine DualRay advantage (3/6 pass vs 0/6 for IIS under 7B K=10 bo6).

However, these positive signals are insufficient to change the verdict because: (a) they are confounded with other factors, (b) the controlled comparison shows zero difference, and (c) the entire repair paradigm is dominated by inference scaling.
