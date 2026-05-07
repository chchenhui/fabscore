# Effectiveness Evaluation Report

## Verdict: good

## Summary

The concept-choice leakage attack via anisotropic noise fingerprinting is **strongly confirmed**. Condition A (SPARSE anisotropic noise) achieves **perfect 1.000 concept-identification accuracy** across all seeds, while Condition B (isotropic noise) stays at **0.189 -- below the 0.20 chance level**. The leakage gap (A - B = 0.811) is decisive.

The covariance smoothing mitigation (Condition C) **fails** to reduce leakage to near-chance. Even at the most aggressive smoothing (lambda=0.99), accuracy remains **0.633**, well above the 0.30 success threshold. The attack is perfectly robust (accuracy=1.000) for all lambda <= 0.95. This reveals a fundamental limitation: any residual concept-specific covariance structure is exploitable by a multi-release fingerprint attacker.

## Experiment Feasibility Check

All three conditions ran successfully and produced complete results:

- **Condition A** (Anisotropic Attack): Mask training converged for all 5 concepts. Attack evaluated over 3 seeds with N=10 multi-release groups. STS12 utility evaluated. Results in `EXPERIMENT_RESULTS/anisotropic_attack/RESULTS.json`.
- **Condition B** (Isotropic Baseline): Noise verification confirmed isotropic variance (CV=0.0064). Attack and STS12 evaluated over 3 seeds. Results in `EXPERIMENT_RESULTS/isotropic_baseline/RESULTS.json`.
- **Condition C** (Smoothed Mitigation): Lambda sweep over {0.2, 0.5, 0.7, 0.9, 0.95, 0.99} with N=10 releases and 3 seeds. STS12 evaluated at key lambdas. Results in `EXPERIMENT_RESULTS/smoothed_mitigation/RESULTS.json`.

No infrastructure, environment, or configuration failures occurred. All GPU workloads completed via TrainService.

## Results Analysis

### Unified Comparison Table

| Method | Concept-ID Accuracy (mean +/- std) | Concept-ID Macro-F1 (mean +/- std) | STS12 Pearson (mean +/- std) |
|--------|-------------------------------------|--------------------------------------|-------------------------------|
| Chance (analytic) | 0.200 +/- 0.000 | 0.200 +/- 0.000 | N/A |
| Isotropic B (Sigma=I) | 0.189 +/- 0.015 | 0.168 +/- 0.012 | -0.015 +/- 0.004 |
| Anisotropic A (SPARSE) | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.028 +/- 0.015 |
| Smoothed C (lam=0.2) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.028 +/- 0.015 |
| Smoothed C (lam=0.5) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- |
| Smoothed C (lam=0.7) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- |
| Smoothed C (lam=0.9) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.029 +/- 0.016 |
| Smoothed C (lam=0.95) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- |
| Smoothed C (lam=0.99) | 0.633 +/- 0.047 | 0.468 +/- 0.055 | 0.030 +/- 0.016 |

Configuration: K=5 concepts, epsilon=10.0, N=10 multi-release groups, 3 random seeds (42, 123, 456).

### Decision Rule 1: Leakage Claim

| Criterion | Threshold | Observed | Met? |
|-----------|-----------|----------|------|
| Condition A accuracy >= 0.50 | 0.50 | **1.000** | YES |
| Condition B accuracy <= 0.25 | 0.25 | **0.189** | YES |

**Result: LEAK CONFIRMED**

Condition A achieves perfect concept identification (1.000 accuracy, 1.000 macro-F1) with zero variance across all 3 seeds. This is not marginal -- it is the strongest possible signal. Condition B is slightly below analytic chance (0.189 vs 0.200), confirming that isotropic noise provides zero concept fingerprint. The gap A - B = 0.811 is maximal.

### Decision Rule 2: Mitigation Claim

| Criterion | Threshold | Observed (best lambda=0.99) | Met? |
|-----------|-----------|----------------------------|------|
| Condition C accuracy <= 0.30 | 0.30 | **0.633** | NO |
| STS12 Pearson drop (A to C) <= 0.02 | 0.02 | **0.002** (0.028 -> 0.030) | YES |

**Result: MITIGATION FAILS**

The smoothing lambda sweep reveals a sharp phase transition:
- lambda in [0.0, 0.95]: accuracy = 1.000 (no mitigation effect whatsoever)
- lambda = 0.99: accuracy = 0.633 (reduced but still 2x above the 0.30 threshold)
- lambda = 1.0 (isotropic): accuracy = 0.189 (chance level, but this eliminates concept-awareness entirely)

STS12 utility is essentially invariant to lambda (0.028-0.030 across all settings), so the utility criterion is satisfied. However, the accuracy criterion is never met for any lambda < 1.0 with N=10 releases.

## Statistical Significance

### Leakage (A vs B)

- Condition A: accuracy = 1.000 +/- 0.000 (all 3 seeds: 1.000, 1.000, 1.000)
- Condition B: accuracy = 0.189 +/- 0.015 (seeds: 0.174, 0.209, 0.184)
- The effect size is maximal (d = infinity, since A has zero variance and is at ceiling)
- No statistical test is needed: perfect accuracy across all seeds with zero variance vs near-chance performance is unambiguous

### Mitigation (C at lambda=0.99 vs chance)

- Condition C (lam=0.99): accuracy = 0.633 +/- 0.047
- Chance level: 0.200
- The difference (0.433) is over 9 standard deviations above chance, confirming statistically significant residual leakage
- Even the worst seed would exceed the 0.30 threshold (0.633 - 2*0.047 = 0.539)

### STS12 Utility Stability

- Condition A STS12: 0.028 +/- 0.015
- Condition C (lam=0.99) STS12: 0.030 +/- 0.016
- Difference: 0.002, well within noise and below the 0.02 threshold
- STS12 is essentially determined by the trace constraint (fixed total noise power), not covariance structure

## Verdict Justification

**Verdict: good**

The experiment is "good" because:

1. **Both main and baseline experiments completed successfully** and produced consistent, reproducible results across multiple seeds. This rules out "failed".

2. **Claim 1 (Leakage) is strongly confirmed.** The anisotropic noise fingerprinting attack achieves perfect (1.000) concept-identification accuracy, while the isotropic control stays at chance (0.189). This is the primary hypothesis of the paper and it is validated with the strongest possible evidence.

3. **Claim 2 (Mitigation) reveals a meaningful negative result.** Covariance smoothing fails to mitigate the attack for any lambda < 1.0. This is scientifically valuable: it identifies a fundamental limitation of linear covariance mixing as a defense against multi-release fingerprinting. The sharp phase transition (1.000 accuracy up to lambda=0.95, dropping to 0.633 only at lambda=0.99) demonstrates that the fingerprint attack is highly robust to smoothing.

4. **The negative mitigation result strengthens the overall narrative.** Rather than undermining the paper, it elevates the threat: concept-choice leakage via anisotropic noise is not only real but also resistant to the most natural defense. This motivates more sophisticated mitigations (e.g., per-query randomization, mechanism switching).

5. **Follow-up analyses are well-motivated.** The lambda sweep (already completed as part of optimization) characterizes the full tradeoff. Additional analyses (random anisotropy control, classifier-based attack, token-presence privacy) will further contextualize the finding.

The verdict is "good" rather than "bad" because the experiment conclusively answers the research question and produces strong, clear results. The mitigation failure is not a failure of the experiment -- it is a finding.
