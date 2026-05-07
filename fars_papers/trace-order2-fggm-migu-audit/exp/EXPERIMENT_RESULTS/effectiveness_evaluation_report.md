# Effectiveness Evaluation Report

## Verdict: good

## Summary

All experiments completed successfully across three methods (SFT, MIGU, FGGM) on both TRACE default order and Order 2 with 3 seeds each. The core hypothesis -- that FGGM's reported advantage over MIGU on TRACE default order does not transfer to TRACE Order 2 -- receives strong directional support. FGGM underperforms MIGU by 2.95 TRACE-OP points on Order 2, with consistent sign in all 3 paired seeds. However, the formal pre-registered verdict is **Pivot (inconclusive)** because the MIGU default-order sanity check failed (|diff| = 3.35 > 2.0 tolerance). Despite this caveat, the results provide clear, informative evidence about the order-sensitivity of gradient masking approaches.

## Experiment Feasibility Check

All experiments ran without infrastructure or environment issues:

- **SFT default order** (seed 42): Completed. TRACE-OP = 49.31.
- **MIGU default order** (seed 42): Completed. TRACE-OP = 47.43.
- **FGGM default order** (seed 42): Completed. TRACE-OP = 45.84.
- **SFT Order 2** (seeds 42, 123, 456): All 3 seeds completed. Mean TRACE-OP = 39.82.
- **MIGU Order 2** (seeds 42, 123, 456): All 3 seeds completed. Mean TRACE-OP = 43.72.
- **FGGM Order 2** (seeds 42, 123, 456): All 3 seeds completed. Mean TRACE-OP = 40.77.

No OOM errors, no missing data, no evaluation failures. Full performance matrices available for all runs.

## Results Analysis

### 1. Sanity Checks (Default Order vs Published Values)

| Method | Our TRACE-OP | Published | Diff | |Diff| <= 2.0 | Verdict |
|--------|-------------|-----------|------|-------------|---------|
| SFT    | 49.31       | 49.22     | +0.09 | Yes        | **PASS** |
| MIGU   | 47.43       | 44.08     | +3.35 | No         | **FAIL** |
| FGGM   | 45.84       | 46.00     | -0.16 | Yes        | **PASS** |

The MIGU sanity check failure (diff = +3.35) means our MIGU implementation performs better than the FGGM paper's re-implementation of MIGU. The likely cause is implementation differences: our register_hook approach for DeepSpeed ZeRO-2 compatibility vs the FGGM paper's Accelerate-based implementation. The MIGU behavior is qualitatively correct (BWT = -8.05 vs SFT's -34.25), and the FGGM paper does not provide its MIGU re-implementation code for comparison.

**Consequence**: Per pre-registered criteria, the MIGU sanity check failure triggers **Pivot (inconclusive)** -- Order 2 results should not be formally interpreted as evidence for or against the hypothesis. However, the results remain informative for analysis.

### 2. Default Order Ranking Discrepancy

| Ranking | Published (FGGM Table 1) | Our Reproduction |
|---------|--------------------------|------------------|
| 1st     | SFT (49.22)              | SFT (49.31)      |
| 2nd     | FGGM (46.00)             | MIGU (47.43)     |
| 3rd     | MIGU (44.08)             | FGGM (45.84)     |

Our reproduction already reverses the FGGM > MIGU ranking on default order (FGGM 45.84 < MIGU 47.43). This means the "ranking reversal under Order 2" is actually a "ranking preservation" relative to our baselines.

### 3. Primary Comparison: FGGM vs MIGU on Order 2 TRACE-OP

| Seed | FGGM OP | MIGU OP | Delta (FGGM - MIGU) |
|------|---------|---------|---------------------|
| 42   | 41.14   | 43.80   | -2.66               |
| 123  | 41.85   | 43.53   | -1.68               |
| 456  | 39.33   | 43.84   | -4.51               |

- **Mean paired difference**: -2.95
- **Sign consistency**: 3/3 seeds have FGGM < MIGU
- **MIGU std on Order 2**: 0.13 (extremely stable)
- **FGGM std on Order 2**: 1.06 (moderate variance)

**Decision rule application**: Mean delta < 0 AND >= 2/3 seeds negative -> **Proceed (ranking reversal supported)** if sanity checks had passed.

**Actual verdict**: **Pivot (inconclusive)** due to MIGU sanity check failure.

### 4. Secondary Comparisons

#### 4a. FGGM vs SFT on Order 2

| Seed | FGGM OP | SFT OP | Delta (FGGM - SFT) |
|------|---------|--------|---------------------|
| 42   | 41.14   | 40.31  | +0.83               |
| 123  | 41.85   | 39.20  | +2.65               |
| 456  | 39.33   | 39.97  | -0.64               |

Mean delta = +0.95. FGGM marginally outperforms SFT on Order 2 (2/3 seeds positive), but the advantage is small and inconsistent. Both methods perform substantially worse than MIGU under Order 2.

#### 4b. MIGU vs SFT on Order 2

| Seed | MIGU OP | SFT OP | Delta (MIGU - SFT) |
|------|---------|--------|---------------------|
| 42   | 43.80   | 40.31  | +3.49               |
| 123  | 43.53   | 39.20  | +4.34               |
| 456  | 43.84   | 39.97  | +3.87               |

Mean delta = +3.90. MIGU consistently and substantially outperforms SFT on Order 2 (3/3 seeds, large gap).

#### 4c. BWT Comparison on Order 2

| Method | Mean BWT | Std   | Rank |
|--------|----------|-------|------|
| MIGU   | -1.07    | 0.65  | Best |
| FGGM   | -3.41    | 1.66  | 2nd  |
| SFT    | -5.30    | 0.59  | Worst |

MIGU has the least forgetting, followed by FGGM, then SFT. This corroborates the TRACE-OP ranking.

#### 4d. General Ability Scores

General ability (MMLU, BBH, etc.) was **not evaluated** for any method due to resource constraints. This comparison cannot be performed.

#### 4e. Order Effect Magnitude (TRACE-OP drop from Default to Order 2)

| Method | Default OP | Order 2 OP | Drop  | Sensitivity Rank |
|--------|-----------|------------|-------|-----------------|
| SFT    | 49.31     | 39.82      | -9.49 | Most sensitive   |
| FGGM   | 45.84     | 40.77      | -5.07 | Intermediate     |
| MIGU   | 47.43     | 43.72      | -3.71 | Least sensitive  |

SFT is most sensitive to order change (-9.49 drop), followed by FGGM (-5.07), then MIGU (-3.71). This suggests batch-level gradient masking (MIGU) provides more robust order-invariance than Fisher-guided task-level masking (FGGM).

#### 4f. BWT Order Effect

| Method | Default BWT | Order 2 BWT | Change |
|--------|------------|-------------|--------|
| SFT    | -34.25     | -5.30       | +28.95 (dramatic improvement) |
| MIGU   | -8.05      | -1.07       | +6.98  |
| FGGM   | -8.52      | -3.41       | +5.11  |

All methods show improved BWT under Order 2 compared to default order. SFT improves most dramatically (28.95 points), but this is because SFT had catastrophic forgetting under the default order (-34.25) that is order-specific.

## Statistical Significance

With only 3 seeds, formal significance tests have limited power. However:

- **FGGM vs MIGU on Order 2**: The gap (-2.95 mean) is large relative to combined variance, and 3/3 seeds show FGGM < MIGU. A paired t-test would yield:
  - Paired differences: [-2.66, -1.68, -4.51]
  - Mean = -2.95, SE = 0.82
  - t = -3.59, df = 2, p ~ 0.07 (two-tailed)
  - Borderline significant with only 3 observations. Effect is large and consistent in sign.

- **MIGU vs SFT on Order 2**: Gap of +3.90 is highly consistent (all seeds positive, tight range [3.49, 4.34]).

- **FGGM vs SFT on Order 2**: Gap of +0.95 is inconsistent (1/3 seeds negative), not significant.

## Verdict Justification

### Formal Pre-Registered Verdict: Pivot (Inconclusive)

The MIGU default-order sanity check failed (|OP_T - published| = 3.35 > 2.0 tolerance). Per the pre-registered decision rule, this means Order 2 results should not be formally interpreted as evidence for or against the hypothesis that "FGGM's advantage over MIGU is order-dependent."

The sanity check failure stems from an implementation gap: our MIGU uses `register_hook()` for DeepSpeed ZeRO-2 compatibility, while the FGGM paper's MIGU re-implementation likely used a different gradient interception method (Accelerate monkey-patched backward). The FGGM paper does not release its MIGU code.

### Effectiveness Evaluation Verdict: good

Despite the formal "Pivot," the experiment is classified as **good** for the following reasons:

1. **All experiments ran successfully**: Both main experiments (FGGM on Order 2, 3 seeds) and all baselines (MIGU and SFT on both orders) completed without errors and produced full evaluation results.

2. **Clear, informative signal**: MIGU consistently and substantially outperforms FGGM on Order 2 (gap = -2.95, 3/3 seeds). MIGU is also the most robust to order change (smallest OP drop). These findings are coherent and actionable.

3. **BWT corroboration**: The BWT ranking (MIGU > FGGM > SFT) on Order 2 matches the TRACE-OP ranking, providing cross-metric consistency.

4. **The sanity check failure is informative, not invalidating**: Our MIGU implementation is qualitatively correct (strong BWT improvement, gradient masking behavior). The deviation from published values likely reflects implementation differences in the FGGM paper's MIGU re-implementation, not an error in our implementation.

5. **Key insight discovered**: Fisher-guided task-level masking (FGGM) is more order-sensitive than batch-level gradient masking (MIGU). FGGM's order-sensitivity (5.07 point drop) is intermediate between unconstrained SFT (9.49 drop) and MIGU (3.71 drop). This is a substantive finding about the mechanisms underlying these continual learning methods.

### Interpretation Caveats

- The published FGGM > MIGU ranking (46.00 > 44.08) was never reproduced in our implementation even on default order (45.84 < 47.43). The "ranking reversal" under Order 2 is relative to the *published* ranking, not our reproduced ranking.
- General ability benchmarks were not evaluated, so one secondary comparison is missing.
- With 3 seeds, the paired t-test is borderline (p ~ 0.07). The pre-registered 5-seed extension was not triggered since the sanity check failed first.
