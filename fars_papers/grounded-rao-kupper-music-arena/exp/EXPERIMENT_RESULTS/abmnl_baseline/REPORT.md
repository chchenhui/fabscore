# AB-MNL Baseline on Music Arena

## Experiment Overview

AB-MNL (Absolute-Badness Multinomial Logit) baseline fitted on Music Arena battle data. AB-MNL is a 4-way outcome model that decouples relative skill from absolute badness by giving each system two parameters: a skill parameter beta_k (governing win/loss) and a badness parameter rho_k (governing BOTH_BAD propensity). This serves as a matched-complexity comparison to the proposed GRK model.

The AB-MNL model defines 4 logits for a pair (i, j):
- u_A = beta_i, u_B = beta_j, u_tie = tau + 0.5*(beta_i + beta_j), u_bad = kappa + 0.5*(rho_i + rho_j)
- P(y) = softmax(u_A, u_B, u_tie, u_bad)

Reference: Novel baseline defined in the research proposal; softmax formulation follows standard multinomial logit conventions.

## Setup

- **Dataset**: `music-arena/music-arena-dataset` from HuggingFace (configs: 2025_07-08 through 2026_01)
- **Total battles**: 3,274 across 12 systems
- **Split**: Chronological 70/30 -- train: 2,291 battles, test: 983 battles
- **BOTH_BAD rate**: 24.3% overall, 19.1% train, 36.5% test
- **Underpower gate**: Passed (test BOTH_BAD count=359, rate=36.5%)
- **Model**: AB-MNL with L-BFGS-B optimization, identifiability: center beta and rho
- **L2 on rho**: Tuned by 5-fold CV on training data; candidates: [0, 0.001, 0.01, 0.1, 1.0, 10.0]
- **Selected L2**: 0.1 (CV mean NLL: 1.0096; all candidates within 0.003 of each other)
- **Parameters**: 2*12 + 2 = 26 (12 beta, 12 rho, tau, kappa)

## Key Results

### Global Test Set (n=983)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 1.0345 | [0.9915, 1.0785] |
| Per-class NLL (A) | 0.6140 | -- |
| Per-class NLL (B) | 0.6393 | -- |
| Per-class NLL (TIE) | 2.6314 | -- |
| Per-class NLL (BOTH_BAD) | 1.3966 | -- |
| BOTH_BAD Brier | 0.2132 | [0.1962, 0.2300] |
| BOTH_BAD ECE | 0.1711 | -- |

### Instrumental Test Subset (n=875)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 0.9948 | [0.9499, 1.0406] |
| Per-class NLL (A) | 0.5838 | -- |
| Per-class NLL (B) | 0.6184 | -- |
| Per-class NLL (TIE) | 2.6519 | -- |
| Per-class NLL (BOTH_BAD) | 1.3338 | -- |
| BOTH_BAD Brier | 0.2189 | [0.2012, 0.2364] |
| BOTH_BAD ECE | 0.1882 | -- |

### Vocal Test Subset (n=108)

| Metric | Value | 95% CI |
|--------|-------|--------|
| 4-way NLL | 1.3564 | [1.2105, 1.5150] |
| Per-class NLL (A) | 0.7893 | -- |
| Per-class NLL (B) | 0.8077 | -- |
| Per-class NLL (TIE) | 2.5674 | -- |
| Per-class NLL (BOTH_BAD) | 2.4075 | -- |
| BOTH_BAD Brier | 0.1672 | [0.1076, 0.2296] |
| BOTH_BAD ECE | 0.1036 | -- |

### AB-MNL Skill Leaderboard (beta, sorted)

| System | Beta Score |
|--------|-----------|
| riffusion-fuzz-1-0 | 1.5591 |
| riffusion-fuzz-1-1 | 1.1589 |
| preview-ocelot | 0.9000 |
| preview-jerboa | 0.8824 |
| elevenlabs-music-v1 | 0.7715 |
| sonauto-v2-2 | 0.5155 |
| magenta-rt-large | 0.2104 |
| musicgen-small | -0.8151 |
| sao | -0.8929 |
| musicgen-medium | -1.0004 |
| sao-small | -1.5581 |
| acestep | -1.7311 |

### AB-MNL Badness Scores (rho, higher = more bad)

| System | Rho Score |
|--------|-----------|
| riffusion-fuzz-1-1 | 0.0106 |
| sao | 0.0092 |
| musicgen-medium | 0.0085 |
| elevenlabs-music-v1 | 0.0033 |
| riffusion-fuzz-1-0 | 0.0033 |
| sonauto-v2-2 | 0.0012 |
| preview-jerboa | -0.0003 |
| magenta-rt-large | -0.0025 |
| preview-ocelot | -0.0031 |
| sao-small | -0.0044 |
| musicgen-small | -0.0126 |
| acestep | -0.0132 |

### Global Parameters

| Parameter | Value |
|-----------|-------|
| tau (tie intercept) | -1.6621 |
| kappa (badness intercept) | -0.4954 |

### CV Results Summary

| L2 on rho | CV Mean NLL | CV Std NLL |
|-----------|-------------|------------|
| 0 | 1.0127 | 0.0367 |
| 0.001 | 1.0102 | 0.0369 |
| 0.01 | 1.0096 | 0.0385 |
| 0.1 | **1.0096** | 0.0390 |
| 1.0 | 1.0096 | 0.0390 |
| 10.0 | 1.0096 | 0.0390 |

## Key Observations

1. **AB-MNL dramatically improves over BT**: Global 4-way NLL drops from 8.13 (BT) to 1.03 (AB-MNL), an ~8x improvement. This is primarily because AB-MNL can actually model TIE and BOTH_BAD outcomes instead of assigning epsilon probability.

2. **BOTH_BAD Brier improves significantly**: 0.213 (AB-MNL) vs 0.365 (BT), a 42% reduction. AB-MNL can assign meaningful probability to BOTH_BAD events.

3. **Per-class NLL shows A/B prediction degrades slightly**: A-class NLL goes from 0.42 (BT) to 0.61 (AB-MNL), B from 0.43 to 0.64. This is the cost of distributing probability mass to TIE and BOTH_BAD -- less mass is available for win predictions.

4. **Rho values are negligibly small**: All rho values are in [-0.013, 0.011], essentially zero. The regularization has minimal effect (CV NLL is nearly identical across all L2 candidates). This suggests the model does not find per-system badness parameters useful -- BOTH_BAD propensity is governed almost entirely by the global kappa intercept rather than system-specific rho.

5. **Negative kappa = -0.50 and negative tau = -1.66**: BOTH_BAD has a much higher baseline mass than TIE. The tie intercept is strongly negative, suppressing TIE predictions. This aligns with the data: TIE is rare while BOTH_BAD is common (36.5% in test).

6. **Vocal subset has higher NLL but lower Brier**: AB-MNL struggles more on vocal predictions (NLL 1.36 vs 0.99 for instrumental), possibly due to fewer vocal training examples and different outcome distribution.

7. **Skill rankings largely preserved from BT**: The beta ordering is similar to BT, with riffusion-fuzz-1-0 at top and acestep at bottom. AB-MNL produces larger score spreads than BT (1.56 vs 1.16 for top system) because it accounts for outcome classes that BT collapses.
