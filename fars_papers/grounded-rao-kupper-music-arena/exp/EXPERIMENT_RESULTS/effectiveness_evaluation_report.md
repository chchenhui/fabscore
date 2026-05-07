# Effectiveness Evaluation Report

## Verdict: good

## Summary

The Grounded Rao-Kupper (GRK) model is effective as a leaderboard model for Music Arena. GRK significantly outperforms both the Bradley-Terry (BT) baseline and the AB-MNL (decoupled badness) baseline on the two primary evaluation metrics -- 4-way NLL and BOTH_BAD Brier score -- across the global test set and both stratified subsets (instrumental-only, vocal-only). The pre-specified decision rule for "Proceed" is satisfied: GRK beats AB-MNL on 4-way NLL with bootstrap 95% CI fully excluding zero, and BOTH_BAD Brier is also improved.

## Experiment Feasibility Check

All three models ran successfully to completion:

- **Bradley-Terry baseline**: Fitted on 2291 training battles, evaluated on 983 test battles across global/instrumental/vocal splits. Results in `EXPERIMENT_RESULTS/bt_baseline/RESULTS.json`.
- **AB-MNL baseline**: Fitted with 5-fold CV for L2 regularization tuning (best L2=0.1). Results in `EXPERIMENT_RESULTS/abmnl_baseline/RESULTS.json`.
- **GRK (proposed, optimized)**: Optimized with per-system gamma tie propensity extension, probability renormalization fix, and improved optimizer convergence. Results in `EXPERIMENT_RESULTS/grk_main/RESULTS.json` and `EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`.

Dataset: music-arena/music-arena-dataset, 3274 battles, 12 systems, 70/30 chronological split. Underpower gate passed (359 BOTH_BAD events in test set, 36.5% rate). No infrastructure or environment issues encountered.

## Results Analysis

### Primary Metrics: 4-Way NLL

| Setting       | BT     | AB-MNL | GRK    | GRK vs AB-MNL |
|---------------|--------|--------|--------|----------------|
| Global        | 8.134  | 1.035  | 0.953  | -0.082 (7.9%)  |
| Instrumental  | 8.329  | 0.995  | 0.911  | -0.084 (8.4%)  |
| Vocal         | 6.556  | 1.356  | 1.288  | -0.068 (5.0%)  |

GRK achieves the lowest 4-way NLL in all three settings. The improvement over AB-MNL ranges from 5.0% (vocal) to 8.4% (instrumental).

### Primary Metrics: BOTH_BAD Brier Score

| Setting       | BT     | AB-MNL | GRK    | GRK vs AB-MNL |
|---------------|--------|--------|--------|----------------|
| Global        | 0.365  | 0.213  | 0.187  | -0.027 (12.4%) |
| Instrumental  | 0.386  | 0.219  | 0.190  | -0.029 (13.2%) |
| Vocal         | 0.194  | 0.167  | 0.160  | -0.007 (4.4%)  |

GRK achieves the lowest BOTH_BAD Brier score in all three settings. The improvement is most pronounced on the instrumental subset (13.2% relative).

### Secondary Metric: BOTH_BAD ECE

| Setting       | BT     | AB-MNL | GRK    |
|---------------|--------|--------|--------|
| Global        | 0.365  | 0.171  | 0.150  |
| Instrumental  | 0.386  | 0.188  | 0.175  |
| Vocal         | 0.194  | 0.104  | 0.053  |

GRK is best-calibrated in all settings. The vocal-only ECE of 0.053 is notably low.

### Per-Class NLL Decomposition (Global Test)

| Outcome   | BT     | AB-MNL | GRK    |
|-----------|--------|--------|--------|
| A (win)   | 0.425  | 0.614  | 0.690  |
| B (win)   | 0.432  | 0.639  | 0.716  |
| TIE       | 18.421 | 2.631  | 2.459  |
| BOTH_BAD  | 18.421 | 1.397  | 1.083  |

Key observations:
- **BT** has low A/B NLL (it only models win probability) but catastrophically fails on TIE and BOTH_BAD (NLL ~18.4, near -log(epsilon)).
- **GRK's gains over AB-MNL are concentrated in BOTH_BAD** (1.08 vs 1.40, a 23% improvement) and TIE (2.46 vs 2.63, 6.5% improvement). This confirms GRK's grounding mechanism is specifically improving the modeling of acceptability outcomes.
- **GRK trades slightly higher A/B NLL** (0.69/0.72 vs 0.61/0.64) for substantially better BOTH_BAD modeling. This is expected: GRK couples BOTH_BAD to skill scores, which slightly constrains the win/loss probabilities but greatly improves BOTH_BAD calibration.

### Bootstrap 95% Confidence Intervals

| Metric               | Setting | Point Estimate | 95% CI             |
|----------------------|---------|----------------|---------------------|
| GRK 4-way NLL        | Global  | 0.953          | [0.915, 0.991]     |
| AB-MNL 4-way NLL     | Global  | 1.035          | [0.992, 1.079]     |
| BT 4-way NLL         | Global  | 8.134          | [7.582, 8.675]     |
| GRK Brier BOTH_BAD   | Global  | 0.187          | [0.174, 0.200]     |
| AB-MNL Brier BOTH_BAD| Global  | 0.213          | [0.196, 0.230]     |
| BT Brier BOTH_BAD    | Global  | 0.365          | [0.337, 0.397]     |

### Model Complexity

| Model  | Parameters |
|--------|-----------|
| BT     | 12 (beta only) |
| AB-MNL | 26 (12 beta + 12 rho + tau + kappa) |
| GRK    | 25 (12 beta + 12 gamma + 1 lambda) |

GRK and AB-MNL have comparable parameter counts (25 vs 26), so GRK's improvement is not attributable to higher model complexity.

## Statistical Significance

Pairwise bootstrap significance tests on global test set (1000 bootstrap resamples):

| Comparison          | Metric     | Difference | 95% CI             | Significant? |
|---------------------|-----------|------------|---------------------|--------------|
| GRK vs BT           | 4-way NLL | -7.182     | [-7.707, -6.664]   | Yes          |
| GRK vs AB-MNL       | 4-way NLL | -0.082     | [-0.096, -0.068]   | Yes          |
| GRK vs BT           | Brier     | -0.179     | [-0.199, -0.158]   | Yes          |
| GRK vs AB-MNL       | Brier     | -0.027     | [-0.031, -0.022]   | Yes          |

All four pairwise comparisons are statistically significant at the 95% level, with confidence intervals fully excluding zero. The narrowest CI (GRK vs AB-MNL Brier: [-0.031, -0.022]) still excludes zero by a comfortable margin.

## Decision Rule Application

The pre-specified decision rule:

> **Proceed** if GRK beats AB-MNL on test-set 4-way NLL by a margin whose bootstrap 95% CI excludes 0, AND BOTH_BAD Brier score is also improved or unchanged.

Evaluation:
1. GRK 4-way NLL (0.953) < AB-MNL 4-way NLL (1.035). Difference = -0.082. Bootstrap 95% CI = [-0.096, -0.068]. **CI excludes zero. Condition 1 satisfied.**
2. GRK BOTH_BAD Brier (0.187) < AB-MNL BOTH_BAD Brier (0.213). **Brier improved. Condition 2 satisfied.**

**Decision: PROCEED.**

The "Pivot" and "Refute" conditions do not apply:
- Pivot requires AB-MNL to win on NLL. It does not; GRK wins.
- Refute requires GRK to show no improvement over AB-MNL on NLL and no improvement over BT on Brier. GRK beats both baselines on both metrics.

## Verdict Justification

**Verdict: good** -- GRK is effective and the approach should proceed to further analysis.

Evidence supporting this verdict:

1. **Consistent improvement across all settings**: GRK outperforms both baselines on both primary metrics in all three evaluation settings (global, instrumental, vocal). There is no setting where GRK underperforms.

2. **Statistical significance**: All pairwise comparisons are statistically significant with bootstrap 95% CIs excluding zero. The effect is not noise.

3. **Improvement where it matters most**: Per-class NLL decomposition shows GRK's gains are concentrated in BOTH_BAD modeling (23% improvement over AB-MNL), which is the specific outcome that GRK's grounding mechanism is designed to improve. This confirms the method is working as theorized.

4. **Matched complexity**: GRK uses 25 parameters vs AB-MNL's 26, ruling out overfitting as an explanation for the improvement.

5. **AB-MNL rho parameters are negligible**: The AB-MNL badness parameters rho are all near zero (max |rho| = 0.013), with badness governed by a single global intercept kappa. This is consistent with GRK's design hypothesis that BOTH_BAD can be grounded in skill scores rather than requiring a separate badness dimension.

6. **Actionable acceptability estimates**: GRK produces well-calibrated BOTH_BAD probabilities (ECE as low as 0.053 on vocal subset) that directly indicate whether both systems in a pair are below the quality threshold. This is a meaningful capability beyond what BT rankings provide.
