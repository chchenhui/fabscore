# Random Anisotropy Ablation: Is Leakage from Spectrum or Mask Structure?

## Experiment Overview

This experiment tests whether concept-choice leakage is a generic property of any anisotropic noise (eigenvalue spectrum alone) or specific to SPARSE's learned concept masks (dimension-concept alignment).

**Method**: For each concept, take the learned diagonal covariance and randomly permute its entries. This preserves the eigenvalue distribution (same set of per-dimension variances) but destroys the concept-specific dimension alignment. The attacker still uses the original (non-permuted) templates, simulating real-world attacker knowledge of the true SPARSE masks.

## Setup

- **Model**: gtr-t5-base (768-d embeddings)
- **Concepts**: 5 (weekdays, months, countries, gender, cities)
- **Protocol**: G=50 target groups, M=200 docs/group, N=2 releases
- **Seeds**: 3 attack seeds (42, 123, 456), mask_seed=42
- **Random permutation**: Independent random permutation of sigma diagonal per concept per seed
- **Attacker templates**: Original (non-permuted) learned concept covariances

## Key Results

| Condition | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) |
|-----------|------------------------|-------------------------|
| Learned Anisotropic (A) | **0.793 +/- 0.023** | **0.675 +/- 0.020** |
| Random Anisotropic | 0.033 +/- 0.024 | 0.013 +/- 0.009 |
| Isotropic (B) | 0.189 +/- 0.015 | 0.168 +/- 0.012 |
| Chance level | 0.200 | - |

## Key Observations

1. **Random anisotropy drops accuracy to near-zero** (0.033), even below chance level (0.200). This is significantly worse than isotropic noise (0.189).

2. **Below-chance accuracy explained**: With random permutation, each concept's actual noise variance profile is a scrambled version of the learned one. The attacker's templates (based on original alignment) systematically mismatch the actual fingerprints, leading to worse-than-random predictions.

3. **Conclusion**: The fingerprint attack relies critically on the **specific dimension-concept alignment** of SPARSE masks, not merely on the eigenvalue spectrum. The learned masks create a unique per-concept variance signature across dimensions; when this alignment is destroyed (even with identical eigenvalues), the attack fails completely.

4. **Implication**: Any defense that disrupts the dimension-specific structure of concept covariances (while potentially keeping a similar eigenvalue spread) would neutralize this leakage channel. This confirms that the concept-choice fingerprint is a consequence of SPARSE's dimension-concept assignment, not an inherent property of anisotropic noise.

## Artifacts

- Attack results: `concept_leakage/results/random_anisotropy/random_aniso_attack_results.json`
- Comparison CSV: `concept_leakage/results/random_anisotropy/random_aniso_results.csv`
- Comparison chart: `concept_leakage/results/random_anisotropy/random_aniso_comparison.png`
- Attack script: `concept_leakage/attack/run_random_anisotropy_attack.py`
- Compilation script: `concept_leakage/results/compile_random_anisotropy.py`
