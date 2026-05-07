# Token-Presence Privacy Probe Evaluation

## Experiment Overview

This experiment evaluates whether concept-choice leakage (the attacker knowing *which* concept was protected) compromises per-document token-level privacy. A binary MLP probe attempts to predict whether any concept token is present in the original sentence from a single sanitized embedding.

## Setup

- **Probe architecture**: 2-layer MLP (768 -> 256 -> 128 -> 1, sigmoid output)
- **Training**: Adam optimizer, lr=1e-3, batch_size=128, 50 epochs, BCE loss
- **Best model selection**: Best checkpoint by validation AUC
- **Data**: D_plus (label=1, contains concept tokens) vs D_minus (label=0, tokens removed)
- **Embedding model**: sentence-transformers/gtr-t5-base (768-d)
- **Noise parameter**: epsilon=10.0
- **Concepts**: weekdays, months, countries, gender, cities (K=5)
- **Seeds**: 42, 123, 456 (3 seeds per condition per concept)
- **Conditions**:
  - Clean: no noise (upper bound on leakage)
  - Isotropic (B): generalized Laplace noise with Sigma=I
  - Anisotropic (A): Mahalanobis noise with concept-specific diagonal Sigma
  - Smoothed (C, lambda=0.2): Mahalanobis noise with smoothed covariance

## Key Results

| Condition | Token-Presence AUC | Token-Presence Accuracy |
|-----------|-------------------|------------------------|
| Clean (no noise) | 0.8648 +/- 0.0578 | 0.7836 +/- 0.0555 |
| Isotropic (B) | 0.5174 +/- 0.0290 | 0.5116 +/- 0.0175 |
| Anisotropic (A) | 0.5193 +/- 0.0277 | 0.5154 +/- 0.0182 |
| Smoothed (C, lambda=0.2) | 0.5160 +/- 0.0302 | 0.5107 +/- 0.0221 |

### Per-Concept AUC (mean over 3 seeds)

| Concept | Clean | Isotropic | Anisotropic | Smoothed |
|---------|-------|-----------|-------------|----------|
| weekdays | 0.859 | 0.569 | 0.566 | 0.571 |
| months | 0.763 | 0.500 | 0.502 | 0.501 |
| countries | 0.936 | 0.503 | 0.501 | 0.499 |
| gender | 0.868 | 0.511 | 0.512 | 0.506 |
| cities | 0.897 | 0.504 | 0.515 | 0.503 |

## Key Observations

1. **Clean embeddings are informative**: Without noise, the probe achieves AUC=0.86, confirming that concept tokens leave detectable traces in sentence embeddings.

2. **All noise conditions destroy token-level signal**: All three noise conditions (isotropic, anisotropic, smoothed) reduce token-presence AUC to near chance (~0.52), with accuracy near 0.51. This confirms strong token-level privacy under all noise mechanisms.

3. **Anisotropic noise provides equivalent token-level privacy**: Despite concept-choice leakage (the attacker can identify which concept was protected), the anisotropic noise (AUC=0.519) provides essentially the same token-level privacy as isotropic noise (AUC=0.517). The difference is not statistically significant.

4. **Smoothed noise offers no additional token-level benefit**: Smoothed covariance (AUC=0.516) provides token-level privacy equivalent to both anisotropic and isotropic noise.

5. **Contextualizing concept-choice leakage**: The concept-choice leakage finding (Task 7: anisotropic achieves 100% concept-ID accuracy) represents a *metadata* leak about which concept was protected, but does NOT translate into per-document content leakage. The noise still effectively obscures whether any specific concept token was present in the original text.

## Files

- Results CSV: `concept_leakage/results/token_presence/token_presence_results.csv`
- Per-condition JSON: `concept_leakage/results/token_presence/{condition}_results.json`
- Summary JSON: `concept_leakage/results/token_presence/summary.json`
- Probe implementation: `concept_leakage/evaluation/token_presence.py`
- Compile script: `concept_leakage/results/compile_token_presence.py`
