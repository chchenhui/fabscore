# Optimized INLP: Early Stopping + C-Tuning via Inner CV

## Experiment Overview

Optimize the INLP method by sweeping INLP iteration count {1,2,3,4,5,7,10} and task-head regularization C {0.01,0.1,1.0,10.0} via inner cross-validation within each LOSO fold. The key insight: the original INLP ran all 10 iterations (removing rank 80), which over-removed task-relevant information. Early stopping preserves more discriminative directions.

## Setup

- **Embeddings**: Frozen CBraMod with EA, flatten pooling (17600-dim)
- **INLP**: Progressive mode -- run 10 iterations, save intermediate projections
- **Config selection**: Inner LOSO CV on 8 training subjects per fold, sweep 7 iter values x 4 C values = 28 configs
- **Task head**: LogisticRegression(solver='lbfgs', max_iter=1000, C=selected)
- **Protocol**: 9-fold LOSO, 3 seeds [42, 123, 456]

## Key Results

| Method | Mean LOSO Acc | Std (seeds) | Delta vs EA Baseline |
|--------|--------------|-------------|---------------------|
| EA + Linear (baseline) | 56.27% | 0.00% | -- |
| EA + INLP-10 + Linear (original) | 55.18% | 0.28% | -1.09 pp |
| **EA + INLP-CV + Linear (optimized)** | **56.29%** | **0.13%** | **+0.02 pp** |

### Per-Seed Mean Accuracy

| Seed | Original INLP | Optimized INLP | Delta |
|------|--------------|----------------|-------|
| 42 | 55.42% | 56.38% | +0.96 pp |
| 123 | 54.87% | 56.35% | +1.48 pp |
| 456 | 55.25% | 56.15% | +0.90 pp |

### Per-Subject Accuracy (averaged across 3 seeds)

| Subject | EA Baseline | Original INLP | Optimized INLP | Delta vs Baseline |
|---------|------------|---------------|----------------|-------------------|
| 1 | 60.42% | 59.43% | 61.81% | +1.39 pp |
| 2 | 40.10% | 42.19% | 40.91% | +0.81 pp |
| 3 | 62.50% | 61.23% | 62.44% | -0.06 pp |
| 4 | 50.00% | 48.89% | 50.62% | +0.62 pp |
| 5 | 51.39% | 50.23% | 51.27% | -0.12 pp |
| 6 | 53.99% | 51.68% | 52.89% | -1.10 pp |
| 7 | 63.89% | 63.31% | 64.76% | +0.87 pp |
| 8 | 64.58% | 61.34% | 64.06% | -0.52 pp |
| 9 | 59.55% | 58.33% | 57.87% | -1.68 pp |

### Config Selection Distribution (across 27 fold-seed combos)

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

1. **Early stopping is critical**: The inner CV overwhelmingly selects 1-3 INLP iterations (74% of folds) vs the original 10. This confirms that aggressive subject-ID removal destroys task-relevant information.

2. **Low regularization preferred**: C=0.01 is selected in 93% of cases (25/27), vs the default C=1.0. After INLP projection reduces the effective dimensionality, stronger regularization prevents overfitting to the remaining directions.

3. **Improvement over original INLP: +1.11 pp**: The optimized method (56.29%) substantially improves over the original INLP (55.18%), recovering nearly all accuracy lost to over-removal.

4. **Matches but does not clearly exceed baseline**: At +0.02 pp vs the EA baseline (56.27%), the optimized INLP essentially matches the baseline. The hypothesis that INLP improves cross-subject generalization is not strongly supported, but it is no longer refuted either.

5. **Subject-specific patterns**: Subjects 1, 2, 4, 7 benefit from optimized INLP (up to +1.39 pp), while subjects 6, 9 degrade (up to -1.68 pp). No universal improvement.
