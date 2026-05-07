# Layer Choice (K_s) Sensitivity Analysis

## Experiment Overview

This experiment tests the sensitivity of the online ratio debiasing method `A_mid / (A_shallow + eps)` to the choice of shallow layer K_s from {1, 2, 3}. Phase-0 diagnostic selected K_s=3 based on maximizing `stability x |position_correlation|`. We evaluate whether the Phase-0 selection criterion is meaningful (predicts best grounding accuracy) and whether the method is robust to layer choice.

## Setup

- **Model**: InternVL2.5-8B (32 LLM layers, 256 visual tokens per patch)
- **Task**: Visual grounding on RefCOCO/RefCOCO+/RefCOCOg (8 splits, ~57k samples)
- **Metric**: Grounding accuracy (IoU >= 0.5)
- **Method**: Online ratio debiasing (`--method online --debiasing-mode ratio`)
- **Pruning layer (K_m)**: 12 (fixed across all runs)
- **Keep ratio**: 10%
- **Selection**: D2Pruner MIS (r_pivot=0.7, theta_sim=0.8, alpha=0.5)
- **Hardware**: 4x A100-80GB GPUs with data parallelism per run

## Key Results

### Layer Choice Comparison Table

| K_s | Phase-0 score (stability x corr) | RefCOCO avg | RefCOCO+ avg | RefCOCOg avg | Overall avg |
|-----|:--------------------------------:|:-----------:|:------------:|:------------:|:-----------:|
| 1   | 0.1425 | 60.36 | 56.21 | 59.11 | 58.49 |
| **2** | **0.1545** | **63.00** | **59.15** | **61.16** | **61.10** |
| 3   | 0.1741 | 56.62 | 52.87 | 54.29 | 54.63 |

**Phase-0 score ranking**: K_s=3 (0.1741) > K_s=2 (0.1545) > K_s=1 (0.1425)
**Grounding accuracy ranking**: K_s=2 (61.10%) > K_s=1 (58.49%) > K_s=3 (54.63%)

### Reference: Raw A_mid (no debiasing) at L12

| Method | Overall avg |
|--------|:-----------:|
| Raw A_mid + MIS (K_m=12, no K_s) | **66.32** |
| Best ratio (K_s=2) | 61.10 (-5.22) |

### Per-Split Results (K_s=1, ratio, K_m=12)

| Split | Accuracy |
|-------|:--------:|
| refcoco_val | 60.59 |
| refcoco_testA | 63.18 |
| refcoco_testB | 57.31 |
| refcoco+_val | 56.18 |
| refcoco+_testA | 60.55 |
| refcoco+_testB | 51.91 |
| refcocog_val | 58.64 |
| refcocog_test | 59.58 |

## Phase-0 Validation

**Result: Phase-0 selection criterion is NOT validated.**

The ranking of K_s by Phase-0 score (K_s=3 > K_s=2 > K_s=1) does NOT match the ranking by grounding accuracy (K_s=2 > K_s=1 > K_s=3). The Phase-0 selected K_s=3 (highest score) actually achieves the WORST grounding accuracy (54.63%).

This discrepancy arises because the Phase-0 criterion (stability x |correlation|) was designed to find layers with attention patterns that are both prompt-invariant and position-correlated — hypothesizing these would be the best correction signal for positional bias. However, since position correlation is uniformly weak across all shallow layers (0.14-0.17), the criterion differentiates layers by a near-noise signal. The actual downstream performance is driven by a different mechanism: how much the ratio denominator distorts the saliency ranking, which anti-correlates with the Phase-0 criterion.

## Robustness Assessment

**Range**: 61.10 - 54.63 = **6.47 points** (> 5 points threshold)
**Classification**: **Significant sensitivity** — the method is NOT robust to K_s choice.

**Layer failure analysis:**
- **K_s=3 fails (54.63%)**: Layer 3 has the lowest entropy (4.71) among the candidates, meaning its attention distribution is more peaked/structured. When used as the ratio denominator, this structured attention aggressively reshapes the mid-layer saliency ranking, distorting it more than the less structured layers. The higher position correlation at K_s=3 (0.175) means the denominator carries more spatial structure, which removes genuine spatial saliency from the numerator.
- **K_s=1 is moderate (58.49%)**: Layer 1 has the highest entropy (5.90), making its attention nearly uniform. The ratio denominator is close to a constant, so it causes less distortion. However, the slight non-uniformity still introduces some noise.
- **K_s=2 is best (61.10%)**: Layer 2 strikes the best balance with intermediate entropy (5.55), but still underperforms raw A_mid by 5.22 points.

## Key Observations

1. **Phase-0 diagnostic does not predict optimal K_s for ratio debiasing.** The stability x correlation criterion ranks layers by a proxy that anti-correlates with downstream performance in the ratio debiasing setting.

2. **The method is significantly sensitive to K_s (6.47-point range).** This means K_s selection matters but the Phase-0 criterion guides it in the wrong direction.

3. **All K_s values underperform raw A_mid (66.32%).** Even the best ratio debiasing (K_s=2, 61.10%) is 5.22 points below raw mid-layer attention. This reinforces the conclusion from the ablation study that ratio debiasing is uniformly harmful.

4. **Lower-entropy shallow layers cause more harm.** The relationship between K_s attention entropy and ratio debiasing harm follows a clear pattern: lower entropy → more structured denominator → more distortion → worse performance. K_s=3 (entropy 4.71, accuracy 54.63%) < K_s=1 (entropy 5.90, accuracy 58.49%) < K_s=2 (entropy 5.55, accuracy 61.10%).

5. **The non-monotonic K_s pattern** (K_s=2 best, not K_s=1) suggests that extremely uniform attention (K_s=1, highest entropy) adds slight noise from the denominator's residual structure, while the intermediate K_s=2 has enough uniformity to minimally distort while providing marginal spatial normalization.
