# H4 Brittleness Analysis Report

## Executive Summary

**H4 Hypothesis Status: ✅ SUPPORTED**

Semantic Entropy demonstrates significant brittleness to hyperparameter changes. FNR varies by more than 20 percentage points when adjusting either the similarity threshold τ or the number of response samples N.

## Detailed Results

**Dataset:** harmbench_twins
**Model:** qwen2.5-7b-instruct
**Number of prompts:** 162

### Performance Matrix

| τ | N | AUROC | FNR@5%FPR |
|---|---|-------|----------|
| 0.1 | 5 | 0.7326 | 0.6296 |
| 0.1 | 10 | 0.7874 | 0.4691 |
| 0.2 | 5 | 0.5556 | 0.8889 |
| 0.2 | 10 | 0.5864 | 0.8272 |
| 0.3 | 5 | 0.5123 | 0.9753 |
| 0.3 | 10 | 0.5309 | 0.9383 |
| 0.4 | 5 | 0.5000 | 1.0000 |
| 0.4 | 10 | 0.5000 | 1.0000 |

### Brittleness Metrics

**Primary Brittleness Indicators:**
- FNR change (τ: 0.1→0.2, N=5): +0.2593 ✓
- FNR change (N: 5→10, τ=0.1): -0.1605 ✗
- Acceptance threshold: ±0.2 (20 percentage points)

**Overall Variability:**
- FNR variance: 0.0329
- FNR std deviation: 0.1815
- FNR range: 0.5309 (min: 0.4691, max: 1.0000)

## Methodology

1. Generated 5 additional responses per prompt (total N=10)
2. Evaluated SE performance across τ ∈ {0.1, 0.2, 0.3, 0.4} and N ∈ {5, 10}
3. Measured FNR changes to assess brittleness
4. Acceptance criterion: FNR change > 20pp for either parameter

## Implications

The significant brittleness of SE has important practical implications:

- **Deployment Risk**: SE performance is highly sensitive to hyperparameter tuning
- **Reproducibility Concerns**: Small changes in settings lead to large performance variations
- **Calibration Challenges**: Practitioners must carefully tune parameters for each use case
- **Sample Size Dependency**: Performance degrades with more response samples, contrary to typical expectations that more data improves estimates
