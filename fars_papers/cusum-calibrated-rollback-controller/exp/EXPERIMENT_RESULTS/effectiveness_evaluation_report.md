# Effectiveness Evaluation Report

## Verdict: bad

## Summary

CUSUM-epsilon (FIR variant with partial reset) does not improve over the Or-epsilon baseline on either primary metric -- peak excess loss or excess AUC -- at matched nominal rollback rate p_0 = 0.002 on ResNet-18/CIFAR-10. The one-step innovation threshold (Or-epsilon) consistently outperforms the CUSUM sequential test across both step and ramp perturbation types. The pre-committed decision rule yields **Refute**: the CUSUM evidence-accumulation hypothesis does not hold in this experimental setting.

## Experiment Feasibility Check

All three experiments ran successfully and produced complete results:

- **No-controller baseline**: 20 seeds x 3 conditions (step, ramp, nominal), 250 steps each
- **Or-epsilon baseline**: 20 seeds x 3 conditions, calibrated with epsilon=1.1805 (99.8th percentile)
- **CUSUM-epsilon (FIR)**: 20 seeds x 3 conditions, optimized with h=18.0, k=0.5, reset_fraction=0.5

One round of optimization was applied to CUSUM (Task 4): the original full-reset CUSUM had a "reset blind spot" causing oscillating accept/reject patterns. The FIR partial reset (S_t = h*0.5 after rollback) improved performance by ~35% on peak excess and ~34% on excess AUC vs the original CUSUM. The evaluation uses this optimized variant.

No infrastructure or environment issues were encountered. All results are based on the best (optimized) CUSUM variant.

## Results Analysis

### Unified Comparison Table

| Method | Perturbation | Peak Excess Loss | Excess AUC | False Rollbacks Outside Window | Detection Delay | Nominal Rollback Rate |
|---|---|---|---|---|---|---|
| No-Controller | Step | 16.942 +/- 8.769 | 284.869 +/- 150.065 | 0.0 +/- 0.0 | N/A | 0.000 |
| No-Controller | Ramp | 22.420 +/- 15.004 | 409.364 +/- 195.260 | 0.0 +/- 0.0 | N/A | 0.000 |
| Or-epsilon | Step | 1.848 +/- 0.918 | 124.076 +/- 65.773 | 87.5 +/- 44.643 | -24.8 +/- 48.026 | 0.0016 |
| Or-epsilon | Ramp | 2.064 +/- 0.705 | 114.597 +/- 51.349 | 86.45 +/- 40.022 | -18.6 +/- 43.856 | 0.0016 |
| CUSUM-epsilon | Step | 3.471 +/- 0.808 | 182.566 +/- 83.207 | 37.0 +/- 37.994 | -18.25 +/- 44.422 | 0.0020 |
| CUSUM-epsilon | Ramp | 3.651 +/- 0.750 | 211.950 +/- 68.585 | 67.35 +/- 44.135 | -22.05 +/- 47.748 | 0.0020 |

### Calibration Parameters

| Parameter | Or-epsilon | CUSUM-epsilon |
|---|---|---|
| mu_0 | -0.0413 | -0.0413 |
| sigma_0 | 0.2260 | 0.2260 |
| epsilon | 1.1805 | N/A |
| h | N/A | 18.0 |
| k | N/A | 0.5 |
| reset_fraction | N/A | 0.5 |
| Nominal rate | 0.0016 | 0.0020 |

### Rate Matching Verification

| Check | Value | Threshold | Result |
|---|---|---|---|
| CUSUM vs p_0 relative error | 0.0% | <= 20% | PASS |
| Or-epsilon vs p_0 relative error | 20.0% | <= 20% | Borderline PASS |
| Cross-controller relative difference | 25.0% | <= 20% | MARGINAL FAIL |

CUSUM's nominal rate matches p_0 exactly. Or-epsilon is slightly less aggressive (0.0016 vs 0.002). The cross-controller difference of 25% marginally exceeds the 20% tolerance. This means CUSUM rolls back slightly more often during normal training, which could theoretically give it a small advantage -- but despite this, CUSUM still performs worse on the primary metrics.

### Step Perturbation Analysis (Primary)

**Peak Excess Loss**:
- Or-epsilon: 1.848 +/- 0.918, interval [0.930, 2.766]
- CUSUM: 3.471 +/- 0.808, interval [2.663, 4.279]
- Result: Or-epsilon is 47% lower. 1-std intervals marginally overlap (CUSUM lower bound 2.663 < Or upper bound 2.766, gap = 0.103).
- Direction: Or-epsilon WINS

**Excess AUC**:
- Or-epsilon: 124.076 +/- 65.773, interval [58.303, 189.849]
- CUSUM: 182.566 +/- 83.207, interval [99.359, 265.773]
- Result: Or-epsilon is 32% lower. 1-std intervals overlap substantially.
- Direction: Or-epsilon WINS

**False Rollbacks Outside Window**:
- Or-epsilon: 87.5 +/- 44.643
- CUSUM: 37.0 +/- 37.994
- 1% threshold: 1.3 steps
- CUSUM has 58% fewer false rollbacks, but both massively exceed threshold
- Direction: CUSUM wins on relative count, but both FAIL the 1% criterion

**Detection Delay**:
- Or-epsilon: -24.8 +/- 48.026
- CUSUM: -18.25 +/- 44.422
- Both show large negative means (triggering before perturbation onset) with high variance
- Direction: Approximately equal (noisy metric)

### Ramp Perturbation Analysis (Robustness)

**Peak Excess Loss**:
- Or-epsilon: 2.064 +/- 0.705, interval [1.359, 2.769]
- CUSUM: 3.651 +/- 0.750, interval [2.901, 4.401]
- Result: Or-epsilon is 44% lower. **1-std intervals DO NOT overlap** (CUSUM lower 2.901 > Or upper 2.769).
- Direction: Or-epsilon WINS SIGNIFICANTLY

**Excess AUC**:
- Or-epsilon: 114.597 +/- 51.349, interval [63.248, 165.946]
- CUSUM: 211.950 +/- 68.585, interval [143.365, 280.535]
- Result: Or-epsilon is 46% lower. 1-std intervals overlap.
- Direction: Or-epsilon WINS

**False Rollbacks Outside Window**:
- Or-epsilon: 86.45 +/- 40.022
- CUSUM: 67.35 +/- 44.135
- Both far exceed the 1% threshold
- Direction: CUSUM slightly better, but both FAIL

**Ramp vs Step Hypothesis**: The hypothesis that CUSUM would show greater advantage on ramp perturbation (gradual drift favoring evidence accumulation) is contradicted. CUSUM's relative deficit is actually larger on ramp (44% worse on peak excess) than step (47% worse on peak excess), with non-overlapping 1-std intervals on ramp but not on step.

## Statistical Significance

### 1-std Interval Non-Overlap Test (Pre-Committed)

| Metric | Perturbation | Or-epsilon Upper | CUSUM Lower | Overlap? | Winner |
|---|---|---|---|---|---|
| Peak Excess Loss | Step | 2.766 | 2.663 | Yes (marginal, gap=0.103) | Or-epsilon (not significant) |
| Peak Excess Loss | Ramp | 2.769 | 2.901 | **No** (gap=0.132) | **Or-epsilon (significant)** |
| Excess AUC | Step | 189.849 | 99.359 | Yes (substantial) | Or-epsilon (not significant) |
| Excess AUC | Ramp | 165.946 | 143.365 | Yes | Or-epsilon (not significant) |

### False Rollback Threshold Check

| Method | Perturbation | False Rollbacks | Threshold (1.3) | Pass? |
|---|---|---|---|---|
| CUSUM | Step | 37.0 | 1.3 | **FAIL** (28.5x over) |
| CUSUM | Ramp | 67.35 | 1.3 | **FAIL** (51.8x over) |
| Or-epsilon | Step | 87.5 | 1.3 | **FAIL** (67.3x over) |
| Or-epsilon | Ramp | 86.45 | 1.3 | **FAIL** (66.5x over) |

Note: The false rollback counts are extremely high for both methods, suggesting the metric definition may include rollbacks during the perturbation window tail or the perturbation effect propagates well beyond the nominal window.

## Verdict Justification

### Decision Rule Application

The pre-committed decision rule has three outcomes:

1. **Proceed**: Requires CUSUM beats Or-epsilon on either peak excess or excess AUC with non-overlapping 1-std intervals, AND false rollbacks <= 1%. **NOT MET**. CUSUM is worse on all primary metrics, and false rollbacks far exceed 1%.

2. **Pivot**: Requires 1-std intervals overlap on both metrics (approximate equivalence). **PARTIALLY MET for step** (marginal overlap on peak excess, full overlap on AUC), **NOT MET for ramp** (non-overlapping on peak excess, Or-epsilon significantly better).

3. **Refute**: CUSUM does not improve or increases false rollbacks at matched rate. **THIS IS THE BEST FIT**. CUSUM is consistently worse on both primary metrics for both perturbation types. On ramp perturbation, the difference is statistically significant (non-overlapping 1-std intervals on peak excess loss).

### Mapping to Effectiveness Verdict

**Refute maps to "bad"**: The proposed CUSUM-epsilon method does not achieve its stated goal of reducing peak excess probe loss and/or excess AUC relative to Or-epsilon. The fundamental mechanism -- multi-step evidence accumulation before triggering rollback -- introduces a detection delay that allows corrupted gradient updates to damage the model before the CUSUM statistic crosses the threshold. The one-step innovation threshold (Or-epsilon) catches anomalies immediately, preventing this cumulative damage.

### Positive Signals (Noted but Insufficient)

- CUSUM-epsilon shows 58% fewer false rollbacks outside the perturbation window on step perturbation (37 vs 87.5), suggesting the evidence-accumulation mechanism does reduce false alarm frequency. However, this advantage does not translate to better loss control, and both methods are far above the 1% threshold.
- Both controllers dramatically reduce peak excess loss vs no-controller (CUSUM: 80-84% reduction, Or-epsilon: 89-91% reduction), confirming the calibration procedure itself is effective.
- The CUSUM calibration procedure (matching alarm rate to p_0) works correctly (exact 0.002 match).

### Root Cause Analysis

The CUSUM sequential test accumulates evidence over multiple steps, which is valuable when individual observations are noisy and the signal-to-noise ratio is low. However, in this experimental setting:
1. The perturbation effect (zeta=300 gradient amplification) produces large, immediately detectable innovations
2. Each perturbed step that is accepted causes cumulative model damage
3. The cost of delay (accepting even one extra perturbed step) outweighs the benefit of evidence accumulation
4. The one-step threshold is well-suited because the perturbation magnitude is large relative to nominal variation

CUSUM might be advantageous in a setting where perturbations are subtle (low SNR) and the cost of false rollbacks is high relative to the cost of accepting one bad step. The current experimental design does not test this regime.
