# EA + INLP Subject-Identity Removal + Linear Head

## Experiment Overview

Evaluate the proposed INLP method: iteratively remove linearly decodable subject-identity information from frozen CBraMod embeddings (17600-dim flatten), then train a linear classification head under the LOSO protocol on BNCI2014001 (9 subjects, 4-class motor imagery).

The optimized version uses progressive INLP with inner cross-validation to select the number of INLP iterations and task-head regularization C per fold, preventing over-removal of task-relevant information.

## Setup

- **Embeddings**: Frozen CBraMod with EA, flatten pooling (17600-dim)
- **INLP classifier**: `SGDClassifier(loss='log_loss', max_iter=1000)` for subject-ID prediction
- **INLP mode**: Progressive -- run up to 10 iterations, save intermediate projection matrices
- **Config selection**: Inner LOSO CV on 8 training subjects, sweep iter {1,2,3,4,5,7,10} x C {0.01,0.1,1.0,10.0}
- **Task head**: `LogisticRegression(solver='lbfgs', max_iter=1000, C=selected)`
- **Protocol**: 9-fold LOSO, 3 seeds [42, 123, 456]
- **Projection**: Memory-efficient basis-vector approach (stores orthonormal basis V, projects as Z - Z@V@V.T)

## Key Results

| Method | Mean LOSO Acc | Std (seeds) | Delta vs EA Baseline |
|--------|--------------|-------------|---------------------|
| EA + Linear (baseline) | 56.27% | 0.00% | -- |
| EA + PCA-1 + Linear | 55.47% | -- | -0.80 pp |
| EA + INLP-10 + Linear (original, fixed iter) | 55.18% | 0.28% | -1.09 pp |
| **EA + INLP-CV + Linear (optimized)** | **56.29%** | **0.13%** | **+0.02 pp** |

### Per-Subject Accuracy (averaged across 3 seeds)

| Subject | EA Baseline | INLP (optimized) | Delta |
|---------|------------|-------------------|-------|
| 1 | 60.42% | 61.81% | +1.39 pp |
| 2 | 40.10% | 40.91% | +0.81 pp |
| 3 | 62.50% | 62.44% | -0.06 pp |
| 4 | 50.00% | 50.62% | +0.62 pp |
| 5 | 51.39% | 51.27% | -0.12 pp |
| 6 | 53.99% | 52.89% | -1.10 pp |
| 7 | 63.89% | 64.76% | +0.87 pp |
| 8 | 64.58% | 64.06% | -0.52 pp |
| 9 | 59.55% | 57.87% | -1.68 pp |

### Per-Seed Mean

| Seed | Mean Acc |
|------|---------|
| 42 | 56.38% |
| 123 | 56.35% |
| 456 | 56.15% |

## Config Selection Distribution (across 27 fold-seed combos)

| Config | Count | Fraction |
|--------|-------|----------|
| iter=1, C=0.01 | 7 | 26% |
| iter=2, C=0.01 | 7 | 26% |
| iter=3, C=0.01 | 6 | 22% |
| iter=4, C=0.01 | 3 | 11% |
| iter=7, C=0.01 | 2 | 7% |
| iter=2, C=0.1 | 1 | 4% |
| iter=5, C=0.1 | 1 | 4% |

## Key Observations

1. **Early stopping is critical**: Inner CV selects 1-3 INLP iterations in 74% of folds vs the original 10. Aggressive subject-ID removal destroys task-relevant information.

2. **Low regularization preferred**: C=0.01 is selected in 93% of cases (25/27), vs the default C=1.0. After INLP projection, stronger regularization prevents overfitting.

3. **Improvement over original INLP: +1.11 pp**: The optimized method (56.29%) substantially improves over the fixed-iteration INLP (55.18%), recovering nearly all accuracy lost to over-removal.

4. **Matches but does not clearly exceed baseline**: At +0.02 pp vs the EA baseline (56.27%), optimized INLP essentially matches the baseline. The hypothesis is not strongly supported, but no longer refuted.

5. **Exceeds PCA-k control**: Optimized INLP (56.29%) beats PCA-1 (55.47%) by +0.82 pp, showing that targeted subject-identity removal outperforms naive dimensionality reduction.

6. **Subject-specific patterns**: Subjects 1, 2, 4, 7 benefit from optimized INLP (up to +1.39 pp), while subjects 6, 9 degrade (up to -1.68 pp). No universal improvement.
