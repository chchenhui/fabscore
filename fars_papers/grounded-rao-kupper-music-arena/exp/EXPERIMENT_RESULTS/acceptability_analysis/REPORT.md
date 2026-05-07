# Acceptability Analysis: GRK Per-Model BOTH_BAD Estimates

## Experiment Overview

This analysis validates GRK's per-model acceptability signal and tests whether AB-MNL's decoupled badness parameters support or contradict GRK's coupling assumption. Four sub-analyses were performed:

1. **Empirical BOTH_BAD rates** per system with bootstrap CIs
2. **GRK-implied acceptability vs empirical** BOTH_BAD correlation
3. **AB-MNL rho vs -beta** coupling test
4. **Leaderboard comparison table** across all models

## Setup

- Dataset: Music Arena (3274 battles, 12 systems)
- Models used: BT baseline, GRK (gamma extension, L2_gamma=0.1), AB-MNL (CV-selected L2_rho=0.1)
- All model parameters loaded from previously fitted results
- Bootstrap: 1000 resamples, seed=42

## Key Results

### 1. Empirical BOTH_BAD Rates

Systems vary widely in BOTH_BAD involvement:
- **Lowest**: preview-ocelot (7.5%), riffusion-fuzz-1-0 (9.2%), acestep (9.2%)
- **Highest**: musicgen-medium (49.2%), sao (37.9%), sao-small (37.7%)
- Note: acestep has a low empirical BOTH_BAD rate (9.2%) despite being the lowest-ranked system by all models. This is a key outlier.

### 2. GRK Acceptability vs Empirical BOTH_BAD

- **Pearson r = 0.597** (p = 0.041, significant at 5%)
- **Spearman r = 0.566** (p = 0.055, marginally significant)

GRK's structural model produces acceptability estimates that correlate meaningfully with empirical BOTH_BAD rates. The moderate (not perfect) correlation is expected: GRK couples BOTH_BAD to skill, so it cannot capture systems like acestep where low skill does not imply high empirical BOTH_BAD.

### 3. AB-MNL rho vs -beta (Coupling Test)

- **Pearson r = -0.416** (p = 0.178, not significant)
- **Spearman r = -0.399** (p = 0.199, not significant)
- **OLS slope = -0.003**, R^2 = 0.17

The correlation is weak and **negative** (opposite to what GRK's coupling would predict). Under GRK's assumption, rho should be proportional to -beta (i.e., worse skill implies more badness). The weak negative correlation means AB-MNL's decoupled parameters do NOT recover GRK's coupling pattern.

However, the near-zero rho values (all |rho| < 0.015) indicate that AB-MNL finds very little system-specific badness variation beyond what skill already explains. This is consistent with heavy L2 regularization on rho effectively collapsing toward GRK's coupled model.

### 4. Leaderboard Comparison

**Kendall's tau = 0.818** (p = 4.4e-05): BT and GRK rankings are highly correlated but not identical.

Notable rank changes (BT rank -> GRK rank):
- **musicgen-medium**: BT rank ~7 -> GRK rank 11 (penalized for 49.2% BOTH_BAD rate)
- **acestep**: BT rank 12 -> GRK rank 12 (lowest in both, but GRK score much more negative)
- **magenta-rt-large**: BT rank 7 -> GRK rank 7 (31.5% BOTH_BAD keeps it mid-tier)

Top-tier systems (riffusion-fuzz-1-0/1-1) maintain their ranking across all methods.

| Rank | System | BT Score | GRK Score | AB-MNL Skill | AB-MNL Badness | Emp. BOTH_BAD (%) | GRK BOTH_BAD (%) |
|------|--------|----------|-----------|--------------|----------------|-------------------|------------------|
| 1 | riffusion-fuzz-1-0 | 1.164 | 1.658 | 1.559 | 0.0033 | 9.2 | 13.8 |
| 2 | riffusion-fuzz-1-1 | 0.825 | 1.322 | 1.159 | 0.0106 | 11.6 | 17.4 |
| 3 | elevenlabs-music-v1 | 0.556 | 0.707 | 0.771 | 0.0033 | 27.0 | 24.8 |
| 4 | preview-ocelot | 0.551 | 0.665 | 0.900 | -0.0031 | 7.5 | 25.4 |
| 5 | preview-jerboa | 0.563 | 0.623 | 0.882 | -0.0003 | 9.4 | 25.9 |
| 6 | sonauto-v2-2 | 0.324 | 0.596 | 0.515 | 0.0012 | 19.3 | 26.2 |
| 7 | magenta-rt-large | 0.051 | 0.459 | 0.210 | -0.0025 | 31.5 | 27.9 |
| 8 | musicgen-small | -0.660 | -0.537 | -0.815 | -0.0126 | 36.3 | 38.7 |
| 9 | sao | -0.626 | -0.609 | -0.893 | 0.0092 | 37.9 | 39.3 |
| 10 | sao-small | -0.987 | -1.293 | -1.558 | -0.0044 | 37.7 | 44.0 |
| 11 | musicgen-medium | -0.541 | -1.407 | -1.000 | 0.0085 | 49.2 | 44.5 |
| 12 | acestep | -1.221 | -2.183 | -1.731 | -0.0132 | 9.2 | 47.3 |

Kendall's tau (BT vs GRK rankings): 0.8182 (p=4.4129e-05)

## Key Observations

1. **GRK's acceptability signal is meaningful**: The moderate positive correlation (r=0.60) with empirical BOTH_BAD rates validates that GRK's structural assumption produces sensible per-model acceptability estimates without needing separate badness parameters.

2. **AB-MNL's rho does not recover GRK's coupling**: The weak, non-significant rho-vs-(-beta) correlation suggests that in the Music Arena data, badness is not simply proportional to negative skill. However, the tiny magnitude of all rho values suggests the decoupled parameters add little explanatory power beyond skill.

3. **acestep is the key outlier**: It has the lowest skill across all models but a very low empirical BOTH_BAD rate (9.2%). GRK overestimates its BOTH_BAD probability (47.3%) because it couples badness to skill. This is the main failure mode of GRK's coupling assumption.

4. **Rank stability**: Despite the coupling assumption, GRK and BT rankings are highly correlated (tau=0.82), indicating the ranking is robust. The main changes involve systems with extreme BOTH_BAD rates.

## Artifacts

- `grk_music_arena/results/empirical_bothbad_rates.json`
- `grk_music_arena/results/rho_beta_correlation.json`
- `grk_music_arena/results/leaderboard_comparison.json`
- `grk_music_arena/results/leaderboard_comparison.md`
- `grk_music_arena/figures/grk_acceptability_vs_empirical.pdf`
- `grk_music_arena/figures/rho_vs_neg_beta.pdf`
