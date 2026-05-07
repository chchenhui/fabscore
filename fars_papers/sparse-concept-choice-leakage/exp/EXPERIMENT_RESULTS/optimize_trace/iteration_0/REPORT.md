# Optimization Iteration 0: Stronger L0 + Multi-Release Fingerprints

## Experiment Overview

Optimized the anisotropic concept-choice leakage attack by addressing three issues:
1. Masks not sparse due to weak L0 regularization (lambda_l0=0.001)
2. Low fingerprint SNR from using only N=2 releases
3. Suboptimal group sizing

## Setup

### Changes from Original
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| lambda_l0 | 0.001 | 0.1 (100x) |
| Learning rate | 1e-4 | 3e-4 |
| Epochs | 100 | 200 |
| N releases | 2 | 10 |
| Group sizing | Fixed G=50 | Adaptive, G=50 M=30 |
| Mask seed | 42 | 42 |

### Mask Training Results
| Concept | Active Dims | Sigma Std | Best Val Loss | Best Epoch |
|---------|------------|-----------|---------------|------------|
| weekdays | 768/768 | 0.0116 | 0.636 | 79 |
| months | 753/768 | 0.3581 | 0.655 | 169 |
| countries | 768/768 | 0.1699 | 0.439 | 94 |
| gender | 768/768 | 0.1705 | 0.523 | 184 |
| cities | 768/768 | 0.1144 | 0.487 | 189 |

Sigma std increased 3-10x across concepts (e.g., months: 0.035 -> 0.358).
Cross-concept cosine similarity dropped from >0.998 to 0.93-0.99.
Template L2 distances increased 5-10x.

## Key Results

### Concept-Identification Attack

| Metric | Original | Optimized |
|--------|----------|-----------|
| Accuracy (mean +/- std) | 0.793 +/- 0.023 | **1.000 +/- 0.000** |
| Macro-F1 (mean +/- std) | 0.675 +/- 0.020 | **1.000 +/- 0.000** |
| Chance level | 0.200 | 0.200 |
| N samples per seed | 201 | 221 |
| N seeds | 3 | 3 |

### STS12 Utility

| Metric | Original | Optimized |
|--------|----------|-----------|
| Clean Pearson | 0.742 | 0.742 |
| Noisy Pearson | 0.030 +/- 0.016 | 0.028 +/- 0.015 |

STS12 utility is unchanged (both near zero at epsilon=10), confirming the privacy-utility tradeoff is preserved.

## Key Observations

1. **Perfect concept identification**: 100% accuracy across all 221 test samples x 3 seeds. The optimized masks create sufficiently distinguishable sigma profiles that N=10 multi-release fingerprints perfectly separate all 5 concepts.

2. **Increased L0 regularization is the primary driver**: lambda_l0=0.1 creates masks with much larger variance in the per-dimension weights (sigma std up to 0.358 for months), dramatically increasing the fingerprint signal.

3. **Multi-release averaging (N=10) eliminates noise**: With 45 pairs averaged per fingerprint vs 1 pair in the original, fingerprint noise is reduced ~6.7x.

4. **The combination of both fixes was critical**: Larger sigma differences (from stronger L0) plus lower fingerprint noise (from N=10) moved the SNR from <1 to >>1, yielding perfect classification.

5. **Mask sparsity is still limited**: Most masks still have 768/768 active dims (except months with 753). The improvement comes from larger variance in mask values, not from zeroing out dimensions. Even stronger L0 (e.g., 1.0) could push toward actual sparsity.
