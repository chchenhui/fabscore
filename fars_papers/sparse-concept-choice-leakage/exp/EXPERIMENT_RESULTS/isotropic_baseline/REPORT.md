# Isotropic Noise Baseline Experiment (Condition B)

## Experiment Overview

Evaluated concept-identification accuracy under isotropic (concept-agnostic) noise as the control baseline for concept-choice leakage. Under isotropic noise (Sigma=I, trace=d=768), all dimensions receive equal noise variance regardless of concept, so the attacker's template-matching accuracy should be near chance (1/K = 0.20 for K=5 concepts).

## Setup

- **Embedding model**: sentence-transformers/gtr-t5-base (768-d)
- **Dataset**: ai4privacy/pii-masking-300k (225,405 documents total)
- **Concepts (K=5)**: weekdays, months, countries, gender, cities
- **Noise mechanism**: Isotropic generalized Laplace, epsilon=10, Sigma=I
- **Privacy budget**: epsilon=10
- **Seeds**: [42, 123, 456]
- **Fingerprint groups**: G=50 per concept (except weekdays G=1 due to small data), M varies by concept

### Per-Concept Data Counts (test split)

| Concept | Test Docs | Groups (G) | Group Size (M) |
|---------|-----------|------------|-----------------|
| weekdays | 21 | 1 | 21 |
| months | 1500 | 50 | 30 |
| countries | 1500 | 50 | 30 |
| gender | 783 | 50 | 15 |
| cities | 482 | 50 | 9 |

## Key Results

### Concept-Identification Attack (Template Matching)

| Metric | Value |
|--------|-------|
| Accuracy (mean +/- std) | 0.1891 +/- 0.0146 |
| Macro-F1 (mean +/- std) | 0.1679 +/- 0.0122 |
| Chance level (1/K) | 0.2000 |

All three seeds produce accuracy near or below chance, confirming that isotropic noise provides no concept-distinguishing signal.

### STS12 Utility Evaluation

| Metric | Value |
|--------|-------|
| Clean Pearson | 0.7423 |
| Noisy Pearson (mean +/- std) | -0.0148 +/- 0.0043 |

Isotropic noise at epsilon=10 in 768 dimensions effectively destroys embedding utility (Pearson near zero).

### Noise Verification

- Per-dimension variance: mean=7.688, std=0.049
- Coefficient of variation: 0.0064 (highly uniform)
- Expected per-dim variance (d/eps^2): 7.680

## Key Observations

1. **Concept-ID accuracy is at chance level (~0.19)**, confirming the isotropic baseline provides the lower bound for concept-choice leakage. This is expected because all K=5 templates are identical uniform vectors under isotropic noise.

2. **STS12 utility is completely destroyed** under isotropic noise with epsilon=10. Pearson correlation drops from 0.74 (clean) to approximately 0 (noisy). This sets a utility floor for comparison with anisotropic mechanisms.

3. **The isotropic noise sampler is correctly implemented**: per-dimension variance is highly uniform (CV=0.006), matching the theoretical expected value of d/eps^2 = 7.68.

4. **Weekdays concept has very limited data** (only 21 test docs / 138 total), which restricts group-level analysis for this concept.
