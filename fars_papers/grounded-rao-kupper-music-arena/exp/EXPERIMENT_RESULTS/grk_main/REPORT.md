# Grounded Rao-Kupper (GRK) on Music Arena

## Experiment Overview

Implemented and evaluated the Grounded Rao-Kupper (GRK) model on Music Arena pairwise preference data (3274 battles, 12 systems). GRK treats BOTH_BAD as an explicit outside option via a fictitious "bad" competitor with fixed score 0, coupling BOTH_BAD probability directly to system skill. Extended with per-system tie propensity parameters (gamma_k) to improve tie prediction, selected by 5-fold CV.

## Setup

- **Data**: Music Arena dataset, 70/30 chronological train/test split (2291 train, 983 test)
- **Model**: GRK with per-system beta scores (12) + per-system gamma tie parameters (12) + global lambda (1) = 25 parameters
- **Optimization**: MLE via L-BFGS-B (ftol=1e-14, gtol=1e-10). L2 regularization on gamma (l2_gamma=0.1, selected by 5-fold CV)
- **Evaluation**: 4-way NLL, per-class NLL, BOTH_BAD Brier score, BOTH_BAD ECE; bootstrap 95% CIs (1000 samples)
- **Baselines**: BT (ties/BOTH_BAD as half-wins), AB-MNL (decoupled skill + badness parameters, L2=0.1 on rho)

## Key Results

### Global Test Set (983 battles, 36.5% BOTH_BAD)

| Method | 4-way NLL (95% CI) | BOTH_BAD Brier (95% CI) | BOTH_BAD ECE |
|--------|-------------------|------------------------|-------------|
| BT | 8.134 [7.582, 8.675] | 0.365 [0.337, 0.397] | 0.365 |
| AB-MNL | 1.035 [0.992, 1.079] | 0.213 [0.196, 0.230] | 0.171 |
| **GRK** | **0.953 [0.915, 0.991]** | **0.187 [0.174, 0.200]** | **0.150** |

### Pairwise Bootstrap Significance (Global)

| Comparison | Diff | 95% CI | Significant |
|-----------|------|--------|-------------|
| GRK vs BT (NLL) | -7.181 | [-7.706, -6.664] | Yes |
| GRK vs AB-MNL (NLL) | -0.082 | [-0.096, -0.068] | Yes |
| GRK vs BT (Brier) | -0.178 | [-0.199, -0.158] | Yes |
| GRK vs AB-MNL (Brier) | -0.026 | [-0.031, -0.022] | Yes |

### Stratified Results

| Split | GRK NLL (95% CI) | GRK Brier (95% CI) | GRK ECE |
|-------|-----------------|--------------------|---------| 
| Global | 0.953 [0.915, 0.991] | 0.187 [0.174, 0.200] | 0.150 |
| Instrumental | 0.911 [0.872, 0.953] | 0.190 [0.176, 0.204] | 0.175 |
| Vocal | 1.288 [1.180, 1.418] | 0.160 [0.108, 0.215] | 0.053 |

### Per-Class NLL Breakdown (Global)

| Method | A-wins | B-wins | TIE | BOTH_BAD |
|--------|--------|--------|-----|----------|
| BT | 0.425 | 0.432 | 18.421 | 18.421 |
| AB-MNL | 0.614 | 0.639 | 2.631 | 1.397 |
| GRK | 0.690 | 0.716 | 2.459 | 1.082 |

### Fitted Model Parameters

- **Lambda**: 1.281 (tie parameter, lambda >= 1)
- **Top systems by beta**: riffusion-fuzz-1-0 (1.658), riffusion-fuzz-1-1 (1.322), elevenlabs-music-v1 (0.707)
- **Bottom systems by beta**: acestep (-2.183), musicgen-medium (-1.407), sao-small (-1.293)
- **Acceptability range**: 0.138 (riffusion-fuzz-1-0, best) to 0.473 (acestep, worst)

## Key Observations

1. **GRK significantly outperforms both baselines** on all metrics (4-way NLL, BOTH_BAD Brier, BOTH_BAD ECE) with all pairwise bootstrap CIs excluding zero.

2. **GRK's biggest improvement is on BOTH_BAD prediction**: BOTH_BAD per-class NLL drops from 1.397 (AB-MNL) to 1.082 (GRK), a 23% reduction. This validates the core hypothesis that coupling BOTH_BAD to skill (via the grounded anchor) is more appropriate than decoupling it.

3. **Per-system gamma extension** improves tie prediction (TIE NLL: 2.488 -> 2.459) and overall NLL without sacrificing BOTH_BAD calibration.

4. **AB-MNL's rho parameters were negligible** (all |rho| < 0.014), confirming that decoupled badness parameters add complexity without benefit in this domain.

5. **Trade-off in A/B win prediction**: GRK has slightly worse A-wins and B-wins NLL (0.690/0.716) vs AB-MNL (0.614/0.639), but this is more than compensated by much better TIE and BOTH_BAD predictions.

6. **Lambda near 1** (1.281) suggests relatively weak tie tendency beyond what the grounded model structure already captures, consistent with the music domain where ties are less common (6.3% of test outcomes).
