# Anisotropic Noise Attack (Condition A) -- Concept-Choice Leakage

## Experiment Overview

This experiment trains SPARSE-style hard-concrete concept masks for K=5 privacy concepts, generates anisotropic (Mahalanobis) noise using the learned per-concept covariance, and demonstrates that the concept-specific variance profile acts as a fingerprint distinguishable via template-matching under the multi-release threat model.

## Setup

- **Embedding model**: gtr-t5-base (768-d)
- **Dataset**: ai4privacy/pii-masking-300k
- **Concepts**: weekdays (96 train), months (7000), countries (7000), gender (3652), cities (2248)
- **Mask training**: Hard-concrete mask + 2-layer MLP, Adam lr=3e-4, batch_size=64, 200 epochs, lambda_L0=0.1, seed=42
- **Noise**: Mahalanobis sampler (SPARSE Algorithm 1) with epsilon=10, concept-specific diagonal Sigma
- **Attack**: Multi-release fingerprint (N=10 releases, 45 pairs averaged), G=50 groups, M=30 docs/group, template matching
- **Utility**: STS12 Pearson correlation

## Key Results

| Metric | Value |
|--------|-------|
| Concept-ID accuracy | **1.000 +/- 0.000** |
| Concept-ID macro-F1 | **1.000 +/- 0.000** |
| Chance level | 0.200 |
| STS12 clean Pearson | 0.742 |
| STS12 noisy Pearson | 0.028 +/- 0.015 |

### Comparison with Isotropic Baseline (Condition B)

| Metric | Isotropic (B) | Anisotropic (A) |
|--------|--------------|-----------------|
| Concept-ID accuracy | 0.189 +/- 0.015 | **1.000 +/- 0.000** |
| Concept-ID macro-F1 | 0.168 +/- 0.012 | **1.000 +/- 0.000** |
| STS12 noisy Pearson | -0.015 +/- 0.004 | 0.028 +/- 0.015 |

## Key Observations

1. **Perfect concept-choice leakage**: Anisotropic noise achieves 100% concept-ID accuracy across all 221 test samples x 3 seeds, far above the 20% chance level and the isotropic baseline (18.9%). This represents a 5x improvement over chance.
2. **Mechanism**: Stronger L0 regularization (lambda_l0=0.1 vs 0.001) creates masks with much larger per-dimension variance (sigma std up to 0.358 for months, vs 0.035 previously). Combined with N=10 multi-release fingerprinting (45 pairs averaged vs 1), the fingerprint SNR exceeds the classification threshold by a wide margin.
3. **Cross-concept sigma cosine similarity** ranges from 0.93-0.99 (vs >0.998 in unoptimized version), confirming much more distinct per-concept profiles.
4. **Utility**: Both isotropic and anisotropic noise at epsilon=10 destroy most embedding utility (STS12 Pearson near zero), which is expected at this privacy level.
5. **Mask sparsity is moderate**: Most masks still have near-full dimensionality (753-768 active dims), but the variance in mask values is dramatically larger, creating distinguishable fingerprints even without hard sparsity.
