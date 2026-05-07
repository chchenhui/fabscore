# Covariance Smoothing Mitigation (Condition C) - Optimized

## Experiment Overview

Evaluated the proposed covariance smoothing mitigation for reducing concept-choice leakage via a comprehensive lambda sweep with the same attacker strength (N=10 releases) as Condition A.

```
Sigma_mix = (1 - lambda) * Sigma_{C_k} + lambda * I_d, re-normalized to trace = d = 768
```

## Setup

- **Smoothing parameter sweep**: lambda in {0.2, 0.5, 0.7, 0.9, 0.95, 0.99}
- **Base covariance**: checkpoints_opt/<concept>/seed42/sigma.npy (lambda_l0=0.1 masks)
- **Attack**: N=10 releases (matched to Condition A), G=50 groups, M=200 docs/group (adaptive)
- **STS12 evaluation**: gtr-t5-base, epsilon=10.0, evaluated at lambda={0.2, 0.9, 0.99}
- **Seeds**: 42, 123, 456

## Key Results

### Lambda Sweep (N=10 releases)

| Lambda | Accuracy | Macro-F1 | STS12 Pearson |
|--------|----------|----------|---------------|
| 0.00 (Cond A) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.028 +/- 0.015 |
| 0.20 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.028 +/- 0.015 |
| 0.50 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - |
| 0.70 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - |
| 0.90 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.029 +/- 0.016 |
| 0.95 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - |
| 0.99 | 0.633 +/- 0.047 | 0.468 +/- 0.055 | 0.030 +/- 0.016 |
| 1.00 (Cond B) | 0.189 +/- 0.015 | 0.168 +/- 0.012 | -0.015 +/- 0.004 |

### Decision Rule Evaluation

Mitigation success requires: concept-ID accuracy <= 0.30 AND STS12 Pearson drops by <= 0.02 vs Condition A.

**Result: Mitigation FAILS at all tested lambda values.**

- Best lambda=0.99: accuracy=0.633 (>0.30 threshold), STS12 drop=0.002 (within threshold)
- The accuracy threshold is never met for any lambda < 1.0

## Key Observations

1. **Smoothing is ineffective up to lambda=0.95**: The attack maintains 100% accuracy even when 95% of the covariance is identity. The N=10 multi-release fingerprint has sufficient SNR to detect the residual 5% concept-specific structure.

2. **Rapid phase transition near lambda=1.0**: Accuracy drops from 1.000 (lambda=0.95) to 0.633 (lambda=0.99) to 0.189 (lambda=1.0). Leakage only vanishes in a narrow band near lambda=1.0 where the noise is essentially isotropic.

3. **STS12 utility is lambda-invariant**: All smoothed conditions have nearly identical STS12 Pearson (~0.028-0.030) because trace(Sigma_mix) = d for all lambda. The total noise magnitude is preserved; only the directional structure changes.

4. **Fundamental limitation**: Any meaningful concept-specific structure in the covariance enables the fingerprint attack with sufficient releases. Covariance smoothing is a linear interpolation that cannot eliminate the attack without converging to isotropic noise, which removes the concept-aware protection benefit entirely.

5. **Improvement over original**: The original experiment only tested lambda=0.2 with N=2 releases. This optimized version provides a complete characterization with N=10 releases (fair comparison) and 6 lambda values, demonstrating the fundamental futility of smoothing as a mitigation.
