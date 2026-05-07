# EA + PCA-k Removal + Linear Head Control Baseline

## Experiment Overview

Rank-matched PCA control baseline: remove the top k principal components from frozen CBraMod (flatten, 17600-dim) embeddings and train a linear head. This tests whether any improvement from INLP is merely due to generic dimensionality reduction (removing high-variance directions) rather than targeted subject-identity removal.

PCA is fit on training-subject embeddings only per LOSO fold and applied to both train and test embeddings, mirroring the INLP protocol.

## Setup

- **Dataset**: BNCI2014001 (9 subjects, 4-class motor imagery, 22 EEG channels)
- **Protocol**: LOSO (9 folds), 3 seeds [42, 123, 456]
- **Embeddings**: Frozen CBraMod + EA, flatten pooling (17600-dim)
- **PCA-k removal**: `PCA(n_components=k)` fit per fold on train subjects; projection `Z_proj = Z - (Z @ V_k.T) @ V_k`
- **Classifier**: LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
- **k values**: [1, 2, 3, 5, 7, 10]
- **Standardization**: StandardScaler fit on projected train, applied to test

## Key Results

| k | Mean LOSO Acc | Std (folds) | Delta vs EA Baseline (56.27%) |
|---|---|---|---|
| 1 | 55.47% | 7.37% | -0.80 pp |
| 2 | 55.13% | 6.96% | -1.14 pp |
| 3 | 55.26% | 6.92% | -1.01 pp |
| 5 | 54.38% | 6.34% | -1.89 pp |
| 7 | 52.05% | 5.46% | -4.22 pp |
| 10 | 51.47% | 5.42% | -4.80 pp |

Best PCA-k: k=1 at 55.47% (still -0.80 pp below EA baseline of 56.27%).

## Key Observations

1. **All PCA-k values degrade performance** relative to the EA baseline (56.27%). Removing top PCA directions hurts task accuracy.
2. **Monotonic degradation with k**: Performance decreases as more components are removed, from 55.47% (k=1) to 51.47% (k=10).
3. **Fold std decreases with k**: Removing high-variance directions reduces inter-subject variability (from 7.37% at k=1 to 5.42% at k=10) but at the cost of discriminative signal.
4. **Negligible seed variance**: LogisticRegression with lbfgs is nearly deterministic; std across seeds is <0.1%.
5. **Implication for INLP comparison**: If INLP at rank k achieves accuracy above PCA-k at the same k by >=1 pp, the improvement is attributed to targeted subject-identity removal rather than generic regularization. The PCA-k control establishes that generic dimensionality reduction alone is harmful, providing a lower bound for the INLP comparison.

## Files

- Raw results CSV: `project/results/baseline_pca_k.csv`
- Implementation: `project/methods/pca_removal.py`
- Evaluation script: `project/scripts/run_pca_k_baseline.py`
- WandB logs: `wandb/` (offline mode)
