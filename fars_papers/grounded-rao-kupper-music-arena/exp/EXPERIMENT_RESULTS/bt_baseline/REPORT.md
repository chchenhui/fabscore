# Bradley-Terry Baseline on Music Arena

## Experiment Overview

Standard Bradley-Terry (BT) model fitted on Music Arena battle data as the primary baseline for arena leaderboard evaluation. BT models only the probability of one system beating another, treating ties as half-wins and collapsing BOTH_BAD into ties. This baseline establishes the reference point for evaluating whether 4-way outcome models (AB-MNL, GRK) improve predictive performance.

Reference: Bradley & Terry (1952), "Rank analysis of incomplete block designs: I. The method of paired comparisons"

## Setup

- **Dataset**: `music-arena/music-arena-dataset` from HuggingFace (configs: 2025_07-08 through 2026_01)
- **Total battles**: 3,274 across 12 systems
- **Split**: Chronological 70/30 -- train: 2,291 battles, test: 983 battles
- **BOTH_BAD rate**: 24.3% overall, 19.1% train, 36.5% test
- **Underpower gate**: Passed (test BOTH_BAD count=359, rate=36.5%)
- **Model**: BT with L-BFGS optimization, identifiability constraint sum(beta)=0
- **Outcome handling**: A->win_a, B->win_b, TIE/BOTH_BAD->0.5 each side
- **4-way prediction**: BT assigns epsilon=1e-8 to TIE and BOTH_BAD (cannot model these)

## Key Results

### Global Test Set (n=983)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 8.1341 | [7.5818, 8.6749] |
| Per-class NLL (A) | 0.4247 | -- |
| Per-class NLL (B) | 0.4316 | -- |
| Per-class NLL (TIE) | 18.4207 | -- |
| Per-class NLL (BOTH_BAD) | 18.4207 | -- |
| BOTH_BAD Brier | 0.3652 | [0.3367, 0.3967] |
| BOTH_BAD ECE | 0.3652 | -- |

### Instrumental Test Subset (n=875)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 8.3288 | [7.6963, 8.9107] |
| BOTH_BAD Brier | 0.3863 | [0.3520, 0.4160] |

### Vocal Test Subset (n=108)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 6.5562 | [5.0499, 8.2558] |
| BOTH_BAD Brier | 0.1944 | [0.1204, 0.2688] |

### BT Leaderboard Scores (sorted)

| System | Beta Score |
|--------|-----------|
| riffusion-fuzz-1-0 | 1.1643 |
| riffusion-fuzz-1-1 | 0.8250 |
| preview-jerboa | 0.5626 |
| elevenlabs-music-v1 | 0.5560 |
| preview-ocelot | 0.5515 |
| sonauto-v2-2 | 0.3238 |
| magenta-rt-large | 0.0514 |
| musicgen-medium | -0.5408 |
| sao | -0.6258 |
| musicgen-small | -0.6600 |
| sao-small | -0.9871 |
| acestep | -1.2210 |

## Key Observations

1. **BT has strong A/B prediction**: Per-class NLL for A (0.42) and B (0.43) are reasonable, confirming BT captures relative skill differences well.
2. **BT cannot predict TIE or BOTH_BAD**: Per-class NLL for both is 18.42 (essentially log(1/epsilon)), confirming BT assigns near-zero mass to these outcomes. This is the fundamental limitation the 4-way models aim to address.
3. **High BOTH_BAD rate in test set**: 36.5% BOTH_BAD in test vs 19.1% in train suggests temporal shift -- later battles involve more BOTH_BAD outcomes (possibly from weaker system pairs being added).
4. **BOTH_BAD Brier = base rate**: BT's Brier score (0.365) approximately equals the BOTH_BAD base rate squared, as expected since BT predicts ~0 for BOTH_BAD probability.
5. **Vocal subset has lower BOTH_BAD rate**: 19.4% vs 38.6% for instrumental, suggesting vocal tracks are generally more acceptable.
6. **Underpower gate passed**: 359 BOTH_BAD events in test set provides adequate statistical power for calibration evaluation.
