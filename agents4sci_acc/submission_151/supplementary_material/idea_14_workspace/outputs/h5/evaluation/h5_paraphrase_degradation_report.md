# H5 Paraphrase Robustness Evaluation Report

**Date**: 2025-08-31 07:55:23

**Hypothesis**: Paraphrasing JailbreakBench prompts will disproportionately degrade SE performance compared to simpler baselines.

**Success Criterion**: SE must degrade >15pp more than baseline methods (particularly for Qwen-2.5-7B).

## Executive Summary

**Result**: **FAIL**

❌ The H5 hypothesis is **not confirmed**. Semantic Entropy does not show significantly worse degradation compared to baseline methods.

## Detailed Results

### Llama-4-Scout

#### H1 Signal Quality Assessment

Assessment of which tau values have sufficient signal in H1 (original prompts) for meaningful comparison:

| Tau | AUROC | Harmful Mean | Benign Mean | Separation | Valid? | Reason |
|-----|-------|--------------|-------------|------------|--------|---------|
| 0.1 | 0.685 | 0.682 | 0.266 | 0.416 | ✅ Yes | Good signal |
| 0.2 | 0.672 | 0.349 | 0.048 | 0.301 | ✅ Yes | Good signal |
| 0.3 | 0.625 | 0.205 | 0.012 | 0.193 | ✅ Yes | Good signal |
| 0.4 | 0.583 | 0.129 | 0.000 | 0.129 | ✅ Yes | Good signal |

**Valid tau values**: [0.1, 0.2, 0.3, 0.4]

#### Full H5 Results (All Tau Values)

Complete degradation results for all methods and tau values:

| Method | FNR@5%FPR (H1→H5) | ΔFNR | AUROC (H1→H5) | ΔAUROC |
|--------|-------------------|------|---------------|--------|
| SE (τ=0.1) | 1.000 → 0.946 | -0.054 | 0.685 → 0.687 | -0.002 |
| SE (τ=0.2) | 0.850 → 0.804 | -0.046 | 0.672 → 0.623 | +0.048 |
| SE (τ=0.3) | 0.733 → 0.804 | +0.070 | 0.625 → 0.598 | +0.027 |
| SE (τ=0.4) | 0.833 → 0.857 | +0.024 | 0.583 → 0.571 | +0.012 |
| Avg Pairwise Bertscore | 0.600 → 0.536 | -0.064 | 0.767 → 0.714 | +0.053 |
| Embedding Variance | 0.667 → 0.732 | +0.065 | 0.654 → 0.662 | -0.009 |
| Levenshtein Variance | 0.883 → 0.875 | -0.008 | 0.289 → 0.254 | +0.035 |

#### H5 Hypothesis Test Results (Filtered)

**Acceptance Criterion**: SE must show ≥15% FNR degradation on paraphrases

**Testing only tau values with good H1 signal for meaningful comparison:**

| SE Config | FNR Degradation | H1 AUROC | Valid Signal? | Passes Test? |
|-----------|-----------------|----------|---------------|-------------|
| τ=0.1 | -0.054 | 0.685 | ✅ Yes | ❌ No |
| τ=0.2 | -0.046 | 0.672 | ✅ Yes | ❌ No |
| τ=0.3 | 0.070 | 0.625 | ✅ Yes | ❌ No |
| τ=0.4 | 0.024 | 0.583 | ✅ Yes | ❌ No |

#### Baseline Degradations (Context)

| Baseline Method | FNR Degradation | AUROC Degradation |
|-----------------|-----------------|------------------|
| Avg Pairwise Bertscore | -0.064 | 0.053 |
| Embedding Variance | 0.065 | -0.009 |
| Levenshtein Variance | -0.008 | 0.035 |

**Final Model Result**: **FAIL** (SE does not show ≥15pp degradation on valid tau values)

### Qwen-2.5-7B (PRIMARY MODEL)

#### H1 Signal Quality Assessment

Assessment of which tau values have sufficient signal in H1 (original prompts) for meaningful comparison:

| Tau | AUROC | Harmful Mean | Benign Mean | Separation | Valid? | Reason |
|-----|-------|--------------|-------------|------------|--------|---------|
| 0.1 | 0.690 | 0.989 | 0.539 | 0.450 | ✅ Yes | Good signal |
| 0.2 | 0.529 | 0.151 | 0.148 | 0.003 | ❌ No | Low AUROC (0.529 < 0.55); Poor separation (0.003 < 0.1) |
| 0.3 | 0.483 | 0.016 | 0.049 | 0.032 | ❌ No | Low AUROC (0.483 < 0.55); Low variance (est. 0.024 < 0.05); Poor separation (0.032 < 0.1) |
| 0.4 | 0.500 | 0.000 | 0.000 | 0.000 | ❌ No | Low AUROC (0.500 < 0.55); Low variance (est. 0.000 < 0.05); Poor separation (0.000 < 0.1) |

**Valid tau values**: [0.1]
**Excluded tau values**: [0.2, 0.3, 0.4] (poor H1 signal)

#### Full H5 Results (All Tau Values)

Complete degradation results for all methods and tau values:

| Method | FNR@5%FPR (H1→H5) | ΔFNR | AUROC (H1→H5) | ΔAUROC |
|--------|-------------------|------|---------------|--------|
| SE (τ=0.1) | 1.000 → 1.000 | +0.000 | 0.690 → 0.695 | -0.004 |
| SE (τ=0.2) | 0.983 → 1.000 | +0.017 | 0.529 → 0.535 | -0.006 |
| SE (τ=0.3) | 0.983 → 0.964 | -0.019 | 0.483 → 0.501 | -0.017 |
| SE (τ=0.4) | 1.000 → 1.000 | +0.000 | 0.500 → 0.500 | +0.000 |
| Avg Pairwise Bertscore | 0.867 → 0.804 | -0.063 | 0.615 → 0.606 | +0.009 |
| Embedding Variance | 0.967 → 0.946 | -0.020 | 0.721 → 0.702 | +0.018 |
| Levenshtein Variance | 0.767 → 0.857 | +0.090 | 0.601 → 0.497 | +0.105 |

#### H5 Hypothesis Test Results (Filtered)

**Acceptance Criterion**: SE must show ≥15% FNR degradation on paraphrases

**Testing only tau values with good H1 signal for meaningful comparison:**

| SE Config | FNR Degradation | H1 AUROC | Valid Signal? | Passes Test? |
|-----------|-----------------|----------|---------------|-------------|
| τ=0.1 | 0.000 | 0.690 | ✅ Yes | ❌ No |

**Excluded tau values (poor H1 signal):**

| SE Config | Reason for Exclusion |
|-----------|-----------------------|
| τ=0.2 | Low AUROC (0.529 < 0.55); Poor separation (0.003 < 0.1) |
| τ=0.3 | Low AUROC (0.483 < 0.55); Low variance (est. 0.024 < 0.05); Poor separation (0.032 < 0.1) |
| τ=0.4 | Low AUROC (0.500 < 0.55); Low variance (est. 0.000 < 0.05); Poor separation (0.000 < 0.1) |

#### Baseline Degradations (Context)

| Baseline Method | FNR Degradation | AUROC Degradation |
|-----------------|-----------------|------------------|
| Avg Pairwise Bertscore | -0.063 | 0.009 |
| Embedding Variance | -0.020 | 0.018 |
| Levenshtein Variance | 0.090 | 0.105 |

**Final Model Result**: **FAIL** (SE does not show ≥15pp degradation on valid tau values)

## Conclusion

Based on the primary model (Qwen-2.5-7B-Instruct), the H5 hypothesis is **not confirmed**.

