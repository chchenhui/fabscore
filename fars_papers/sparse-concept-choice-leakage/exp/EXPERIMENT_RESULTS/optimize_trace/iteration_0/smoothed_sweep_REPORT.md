# Optimized Smoothed Mitigation (Condition C) - Lambda Sweep

## Experiment Overview

Optimized the Condition C (covariance smoothing mitigation) experiment by:
1. Fixing N_releases inconsistency: changed from N=2 to N=10 (matching Condition A)
2. Running a comprehensive lambda sweep: {0.2, 0.5, 0.7, 0.9, 0.95, 0.99}
3. Evaluating STS12 utility at representative lambdas (0.2, 0.9, 0.99)

## Setup

- **Base covariance**: checkpoints_opt/<concept>/seed42/sigma.npy (lambda_l0=0.1 masks)
- **Smoothing**: Sigma_mix = (1-lam)*Sigma_Ck + lam*I, renormalized to trace=d=768
- **Attack**: N=10 releases, G=50 groups (adaptive), M=200 docs/group (adaptive)
- **Seeds**: 42, 123, 456
- **STS12**: gtr-t5-base, epsilon=10.0

## Key Results

### Lambda Sweep (N=10 releases, 3 seeds averaged)

| Lambda | Accuracy | Std | Macro-F1 | STS12 Pearson |
|--------|----------|-----|----------|---------------|
| 0.00 (A) | 1.000 | 0.000 | 1.000 | 0.028 |
| 0.20 | 1.000 | 0.000 | 1.000 | 0.028 |
| 0.50 | 1.000 | 0.000 | 1.000 | - |
| 0.70 | 1.000 | 0.000 | 1.000 | - |
| 0.90 | 1.000 | 0.000 | 1.000 | 0.029 |
| 0.95 | 1.000 | 0.000 | 1.000 | - |
| 0.99 | 0.633 | 0.047 | 0.468 | 0.030 |
| 1.00 (B) | 0.189 | 0.015 | 0.168 | -0.015 |

### Decision Rule Evaluation

Mitigation success requires: accuracy <= 0.30 AND STS12 drop <= 0.02 vs Condition A.

- Lambda=0.2 (original): accuracy=1.000 -- FAILS (>> 0.30)
- Lambda=0.9: accuracy=1.000 -- FAILS
- Lambda=0.95: accuracy=1.000 -- FAILS
- Lambda=0.99: accuracy=0.633 -- FAILS (> 0.30)
- STS12 utility is unchanged across all lambdas (~0.028-0.030)

**Conclusion**: Covariance smoothing cannot bring concept-ID accuracy below 0.30 with N=10 releases for any lambda < 1.0. Only lambda=1.0 (isotropic = Condition B) eliminates leakage, but this removes all concept-specific protection structure. This is a fundamental limitation of the smoothing approach.

## Key Observations

1. The attack is robust up to lambda=0.95 with N=10 releases (100% accuracy)
2. Even lambda=0.99 (99% identity + 1% concept-specific) still allows 63.3% accuracy (3.2x chance)
3. STS12 utility is invariant to lambda because all smoothed covariances have trace=d
4. The smoothing linearly interpolates sigma profiles toward identity but the attack signal-to-noise ratio with N=10 releases is high enough to distinguish even tiny residual structure
