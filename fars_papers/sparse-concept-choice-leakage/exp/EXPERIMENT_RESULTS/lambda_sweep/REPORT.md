# Lambda Sweep: Privacy-Leakage vs Utility Trade-Off

## Experiment Overview

Investigated the trade-off between concept-choice leakage and embedding utility as the covariance smoothing parameter lambda varies from 0 (full anisotropic, Condition A) through 0.5. The smoothing formula is:

```
Sigma_mix(lambda) = (1-lambda) * Sigma_{C_k} + lambda * I, re-normalized to trace = d = 768
```

## Setup

- **Smoothing parameters**: lambda in {0, 0.1, 0.2, 0.5}
- **Attack**: Template matching with N=2 releases, G=50 groups, M=200 docs/group
- **Utility**: STS12 Pearson correlation under Mahalanobis noise (epsilon=10.0)
- **Seeds**: 42, 123, 456 (3 seeds for mean +/- std)
- **Base covariance**: checkpoints_opt/<concept>/seed42/sigma.npy (lambda_l0=0.1 masks)
- **Reused results**: lambda=0 from Condition A, lambda=0.2 from Condition C
- **New experiments**: lambda=0.1 and lambda=0.5

## Key Results

| Lambda | Accuracy | Macro-F1 | STS12 Pearson |
|--------|----------|----------|---------------|
| 0.0 | 0.793 +/- 0.023 | 0.675 +/- 0.020 | 0.030 +/- 0.016 |
| 0.1 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.028 +/- 0.015 |
| 0.2 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.028 +/- 0.015 |
| 0.5 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.029 +/- 0.016 |

## Key Observations

1. **Smoothing does not reduce leakage in the tested range**: For lambda in {0.1, 0.2, 0.5}, the attack achieves perfect 1.000 accuracy. Even mixing 50% identity into the covariance leaves enough concept-specific structure for the template-matching attack.

2. **Paradoxical accuracy increase at lambda=0.1**: lambda=0 has accuracy 0.793 while lambda=0.1 achieves 1.000. This is because the original anisotropic attack (lambda=0) used the N=2 default with the original checkpoints, while the smoothed attack uses optimized masks (checkpoints_opt) which have sharper concept-specific covariances. The higher accuracy at lambda>0 reflects the stronger masks, not a benefit of smoothing.

3. **Utility is lambda-invariant**: STS12 Pearson stays constant at ~0.028-0.030 across all lambdas. Since trace(Sigma_mix) = d for all lambda, the total noise magnitude is preserved; only the directional structure changes. This means smoothing costs nothing in utility but also gains nothing in privacy.

4. **No meaningful trade-off exists in this range**: There is no lambda in [0, 0.5] where accuracy drops to near-chance (0.20). The anisotropy threshold where concept identification becomes difficult lies much closer to lambda=1.0 (isotropic), as confirmed by the N=10 optimized sweep where accuracy remained 1.000 up to lambda=0.95 and only dropped to 0.633 at lambda=0.99.

5. **Practical implication**: Covariance smoothing is not a viable mitigation for concept-choice leakage. Any lambda that preserves meaningful concept-aware structure (lambda < ~0.99) is trivially attacked.

## Artifacts

- CSV: `concept_leakage/results/lambda_sweep/lambda_sweep_results.csv`
- Plot 1: `concept_leakage/results/lambda_sweep/accuracy_vs_lambda.png`
- Plot 2: `concept_leakage/results/lambda_sweep/tradeoff_dual_axis.png`
- Attack results: `concept_leakage/results/lambda_sweep/smoothed_attack_lam{X}_results.json`
- STS12 results: `concept_leakage/results/lambda_sweep/smoothed_sts12_lam{X}_results.json`
