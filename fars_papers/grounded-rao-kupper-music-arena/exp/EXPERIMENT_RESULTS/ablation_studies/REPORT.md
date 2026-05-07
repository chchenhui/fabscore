# Ablation Studies: GRK Grounding Mechanism and Regularization Analysis

## Experiment Overview

Two ablation studies probing the mechanism behind GRK's performance improvement:

1. **GRK Without Grounding**: Tests whether the structural coupling of BOTH_BAD probability to model quality (the "grounding" mechanism) is necessary for GRK's benefit. Replaces the model-quality-dependent P(BOTH_BAD) = 1/(1+phi_i+phi_j) with a constant P(BOTH_BAD) = sigmoid(c).

2. **AB-MNL Regularization Sensitivity**: Tests whether strong L2 regularization on AB-MNL's rho parameters pushes it toward GRK-like behavior, which would suggest GRK's improvement is primarily a regularization/parsimony effect rather than a structural insight.

## Setup

- Dataset: Music Arena, 3274 battles, 12 systems
- Train: 2291 battles (70%), Test: 983 battles (30%)
- Full GRK variant: gamma extension, use_gamma=True, l2_beta=0.0, l2_gamma=0.1 (selected by CV)
- Metrics: 4-way NLL, per-class NLL, BOTH_BAD Brier score
- Bootstrap: 1000 resamples, 95% CIs

## Key Results

### Ablation 1: GRK Without Grounding

| Model             | 4-way NLL | 95% CI            | BOTH_BAD Brier | BOTH_BAD NLL |
|-------------------|-----------|-------------------|----------------|--------------|
| BT                | 8.134     | [7.582, 8.675]    | 0.3652         | 18.421       |
| AB-MNL (L2=0.1)  | 1.035     | [0.992, 1.078]    | 0.2132         | 1.397        |
| GRK-No-Grounding  | 1.156     | [1.115, 1.199]    | 0.2621         | 1.655        |
| GRK (full, gamma) | 0.953     | [0.915, 0.990]    | 0.1867         | 1.082        |

**Pairwise bootstrap (GRK-No-Grounding minus GRK):**
- NLL diff: +0.204, 95% CI [+0.178, +0.229] -- significant, excludes zero
- Brier diff: +0.075, 95% CI [+0.065, +0.086] -- significant, excludes zero

**Interpretation**: Removing grounding significantly degrades GRK performance. GRK-No-Grounding is worse than AB-MNL on every metric. The structural coupling of BOTH_BAD to model quality is essential -- a constant BOTH_BAD rate cannot capture the matchup-dependent quality signal.

### Ablation 2: AB-MNL Regularization Sensitivity

| L2 on rho | 4-way NLL | BOTH_BAD Brier | Pearson corr(rho, -beta) | Spearman corr(rho, -beta) |
|-----------|-----------|----------------|--------------------------|---------------------------|
| 0         | 1.026     | 0.2097         | -0.342                   | -0.531                    |
| 0.001     | 1.030     | 0.2117         | -0.410                   | -0.427                    |
| 0.01      | 1.034     | 0.2130         | -0.424                   | -0.420                    |
| 0.1       | 1.035     | 0.2132         | -0.416                   | -0.399                    |
| 0.5       | 1.035     | 0.2132         | -0.414                   | -0.399                    |
| 1.0       | 1.035     | 0.2132         | -0.413                   | -0.399                    |
| 5.0       | 1.035     | 0.2132         | -0.413                   | -0.399                    |
| 10.0      | 1.035     | 0.2132         | -0.413                   | -0.399                    |
| 50.0      | 1.035     | 0.2132         | -0.415                   | -0.399                    |
| 100.0     | 1.035     | 0.2132         | -0.414                   | -0.399                    |

GRK reference NLL: **0.953**

**Interpretation**: AB-MNL's test NLL varies minimally across the full L2 range (1.026--1.035), never approaching GRK's 0.953. The correlation between rho and -beta is consistently negative (~-0.41 Pearson), meaning high-badness systems are NOT the same as low-skill systems. This is the opposite of what GRK structurally enforces (where badness is inversely related to skill). Strong regularization does NOT push AB-MNL toward GRK-like behavior.

## Key Observations

1. **Grounding is essential**: The structural coupling of BOTH_BAD probability to model quality is the primary driver of GRK's improvement, not the Rao-Kupper framework itself. Without grounding, GRK-No-Grounding performs worse than AB-MNL.

2. **Not a regularization artifact**: AB-MNL with arbitrarily strong regularization on rho never approaches GRK's performance (NLL gap ~0.08 at best). The improvement is genuinely structural.

3. **Decoupled badness is suboptimal**: AB-MNL's negative correlation between rho and -beta suggests that allowing free badness parameters leads the model to capture noise rather than the true skill-quality relationship that GRK enforces.

4. **GRK's parsimony matters**: GRK uses fewer parameters (no separate rho_k) yet achieves better fit, confirming that the inductive bias (BOTH_BAD coupled to skill) is correct for this domain.

## Artifacts

- `grk_music_arena/results/grk_no_ground_global.json`: Full GRK-no-grounding results
- `grk_music_arena/results/abmnl_reg_sensitivity.json`: Full regularization sweep results
- `grk_music_arena/figures/grk_ablation.pdf`: Bar chart comparing all models
- `grk_music_arena/figures/abmnl_reg_curve.pdf`: Regularization sensitivity curve
