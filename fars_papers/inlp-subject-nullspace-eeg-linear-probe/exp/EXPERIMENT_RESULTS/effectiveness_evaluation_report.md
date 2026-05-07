# Effectiveness Evaluation Report

## Verdict: bad

## Summary

The proposed INLP-based subject-identity removal from frozen CBraMod embeddings does not improve cross-subject EEG motor imagery classification. After optimization (inner-CV selection of INLP iterations and regularization), the method achieves 56.29% mean LOSO accuracy -- a negligible +0.02 pp improvement over the EA baseline (56.27%) that is statistically indistinguishable from zero. The pre-defined success criteria (+2 pp over EA, +1 pp over PCA-k) are not met. The hypothesis is refuted.

## Experiment Feasibility Check

All experiments ran successfully without infrastructure or environment issues:

- **EA Baseline** (Task 1): Completed. 9-fold LOSO, 3 seeds, deterministic results.
- **PCA-k Control** (Task 2): Completed. 6 values of k tested (1,2,3,5,7,10), 3 seeds each.
- **Main INLP** (Task 3): Completed. Original fixed 10-iteration INLP with 3 seeds.
- **Optimized INLP** (Task 4): Completed. Inner-CV selection over 28 hyperparameter configs per fold, 3 seeds.

All results are available and consistent across seeds. No missing data, OOM errors, or other failures.

## Results Analysis

### Full Comparison Table

| Method | Mean Acc (%) | Std (seeds, %) | Std (folds, %) | Delta vs EA (pp) |
|--------|:-----------:|:-------------:|:--------------:|:-----------------:|
| **EA + Linear Head (baseline)** | **56.27** | 0.00 | 8.09 | -- |
| EA + PCA-1 + Linear Head | 55.47 | 0.02 | 7.37 | -0.80 |
| EA + PCA-2 + Linear Head | 55.13 | 0.01 | 6.96 | -1.14 |
| EA + PCA-3 + Linear Head | 55.26 | 0.04 | 6.92 | -1.01 |
| EA + PCA-5 + Linear Head | 54.38 | 0.04 | 6.34 | -1.89 |
| EA + PCA-7 + Linear Head | 52.05 | 0.10 | 5.46 | -4.22 |
| EA + PCA-10 + Linear Head | 51.47 | 0.04 | 5.42 | -4.80 |
| EA + INLP-10 (original) + Linear | 55.18 | 0.28 | 7.19 | -1.09 |
| **EA + INLP-CV (optimized) + Linear** | **56.29** | **0.13** | **7.98** | **+0.02** |
| Vanilla linear probing (reported) | 41.45 | 0.50 | -- | -14.82 |
| Full fine-tuning (reported) | 53.03 | 0.22 | -- | -3.24 |

### Per-Subject Breakdown (Optimized INLP vs EA Baseline)

| Subject | EA Baseline (%) | INLP-CV (%) | Delta (pp) | Direction |
|:-------:|:--------------:|:-----------:|:----------:|:---------:|
| 1 | 60.42 | 61.81 | +1.39 | improved |
| 2 | 40.10 | 40.91 | +0.81 | improved |
| 3 | 62.50 | 62.44 | -0.06 | unchanged |
| 4 | 50.00 | 50.62 | +0.62 | improved |
| 5 | 51.39 | 51.27 | -0.12 | unchanged |
| 6 | 53.99 | 52.89 | -1.10 | degraded |
| 7 | 63.89 | 64.76 | +0.87 | improved |
| 8 | 64.58 | 64.06 | -0.52 | degraded |
| 9 | 59.55 | 57.87 | -1.68 | degraded |

- 4 subjects improved (S1, S2, S4, S7): +0.62 to +1.39 pp
- 2 subjects unchanged (S3, S5): within +/-0.12 pp
- 3 subjects degraded (S6, S8, S9): -0.52 to -1.68 pp
- No consistent pattern: improvements and degradations are distributed across both easy and hard subjects

## Decision Rule Evaluation

### Criterion (a): Proceed -- INLP >= EA + 2 pp AND INLP >= PCA-k + 1 pp

- INLP - EA = 56.29% - 56.27% = **+0.02 pp** (required: >= +2.00 pp) -- **FAILS**
- INLP - PCA-1 = 56.29% - 55.47% = **+0.82 pp** (required: >= +1.00 pp) -- **FAILS**
- **Result: NOT triggered.** Neither threshold is met.

### Criterion (b): Pivot -- INLP ~ PCA-k AND both > EA

- INLP - PCA-1 = +0.82 pp (approximately equal, < 1 pp difference) -- condition met
- PCA-1 (55.47%) < EA (56.27%) -- "both > EA" is **FALSE**
- **Result: NOT triggered.** PCA-k does not exceed the EA baseline.

### Criterion (c): Refute -- INLP <= max(EA, PCA-k)

- max(EA, PCA-1) = max(56.27%, 55.47%) = 56.27% (EA)
- INLP = 56.29% ~ 56.27% (difference = +0.02 pp, within seed variance of 0.13%)
- **Result: TRIGGERED.** INLP is statistically indistinguishable from the EA baseline. Subject-identity removal does not help.

### Over-Removal Diagnostic

The optimization trace provides strong evidence of the over-removal problem:
- Inner CV selects 1-3 INLP iterations in **74% of folds** (vs original 10)
- C=0.01 is selected in **93% of cases** (vs default C=1.0)
- The original INLP (10 iterations) removed rank 80 from 17600-dim embeddings, reducing subject-ID accuracy from 99.86% to 23.88% -- but this aggressive removal hurt task accuracy by -1.09 pp
- The optimized INLP achieves its best accuracy by applying minimal intervention, confirming that subject-identity directions are entangled with task-relevant signal

## Statistical Significance

### Seed Variance Analysis

The INLP per-seed accuracies are: 56.38%, 56.35%, 56.15% (mean=56.29%, std=0.13%).
The EA baseline is deterministic at 56.27% (lbfgs solver, std=0.00%).

A one-sample t-test of INLP seed means against the EA baseline value:
- t = (56.29 - 56.27) / (0.13 / sqrt(3)) = 0.02 / 0.075 = 0.27
- df = 2, p-value (two-tailed) >> 0.05
- The difference is **not statistically significant**.

### Fold-Level Variance

With 9 LOSO folds and fold-level std of ~8%, per-subject deltas of +/-1-2 pp are well within expected noise. No individual subject shows a statistically meaningful change.

### Comparison with PCA-k

The INLP-PCA gap of +0.82 pp is modest and likely reflects the different hyperparameter selection mechanism (inner CV for INLP vs fixed for PCA) rather than targeted identity removal. Both PCA-k and original INLP degrade accuracy relative to the baseline, supporting the interpretation that removing any directions from these embeddings is generally harmful.

## Contextualization Against Fine-Tuning Upper Bound

The reported baselines from Liu et al. 2026 Table XV are:
- Vanilla linear probing: 41.45%
- Full fine-tuning: 53.03%
- Gap: 11.58 pp

The EA + flatten baseline (56.27%) already **exceeds** the reported fine-tuning upper bound by +3.24 pp. This means:
1. EA preprocessing is the dominant factor, not INLP
2. There is no remaining "linear-probing-to-fine-tuning gap" for INLP to close -- EA already closed it and then some
3. The original problem framing (improving frozen transfer to close the gap with fine-tuning) is moot because EA already achieves this

## Verdict Justification

The verdict is **bad** based on the following evidence:

1. **Both success criteria fail**: INLP - EA = +0.02 pp (need +2 pp); INLP - PCA-1 = +0.82 pp (need +1 pp). The method does not meet its own pre-defined success thresholds.

2. **Statistically negligible improvement**: The +0.02 pp difference is within seed variance (std=0.13%) and far below any reasonable significance threshold.

3. **Optimization reveals the method self-nullifies**: When given the freedom to tune, inner CV selects minimal INLP intervention (1-3 iterations, C=0.01), effectively making the projection a near-identity operation. The best INLP is one that barely modifies the embeddings.

4. **Original INLP actively hurts**: The unoptimized 10-iteration INLP degrades accuracy by -1.09 pp, and all PCA-k controls also degrade accuracy (up to -4.80 pp for k=10). Removing directions from these embeddings is harmful in general.

5. **No consistent per-subject benefit**: 4 subjects improve, 2 are unchanged, 3 degrade. No systematic pattern suggests targeted identity removal helps any identifiable subgroup.

6. **Baseline already exceeds upper bound**: The EA baseline (56.27%) already surpasses the reported fine-tuning accuracy (53.03%), eliminating the motivation for INLP intervention.

**Conclusion**: INLP-based subject-identity removal is not an effective post-hoc intervention for improving frozen EEG-FM transfer on BNCI2014001 with Euclidean Alignment. The subject-identity information in frozen CBraMod embeddings is too entangled with task-discriminative signal to be removed beneficially. The strong EA preprocessing already achieves state-of-the-art frozen-encoder performance, and INLP provides no additive benefit.
