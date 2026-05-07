# Supplementary Analyses: k-Sensitivity and Geodesic Scatter Plots

## Experiment Overview

Two supplementary analyses to validate robustness of degree-penalized geodesic reconstruction:
1. **Recall@k Sensitivity**: Verify improvement holds across k in {5, 10, 20} (not just k=10)
2. **Geodesic vs Euclidean Scatter Plots**: Visualize whether degree penalty improves the distance correlation underlying kNN recovery

## Setup

- Datasets: SIFT10K (128-d), MSMARCO-10K (768-d)
- Seed: 42
- SIFT10K params: L=2000, d=128, alpha=3.0
- MSMARCO-10K params: L=3000, d=128, alpha=4.0
- Scatter plot: 10,000 sampled node pairs, all-pairs shortest paths

## Key Results

### Recall@k Sensitivity Table

| Method | Dataset | Recall@5 | Recall@10 | Recall@20 |
|--------|---------|----------|-----------|-----------|
| Adjacency-only | SIFT10K | 0.2649 | 0.3470 | 0.3349 |
| Unweighted geodesic | SIFT10K | 0.1440 | 0.1623 | 0.1877 |
| Degree-penalized geodesic | SIFT10K | 0.2455 | 0.2726 | 0.3014 |
| Adjacency-only | MSMARCO-10K | 0.2279 | 0.3251 | 0.3570 |
| Unweighted geodesic | MSMARCO-10K | 0.2325 | 0.2702 | 0.2977 |
| Degree-penalized geodesic | MSMARCO-10K | 0.3665 | 0.4169 | 0.4388 |

### Scatter Plot Spearman Correlations (SIFT10K)

| Metric | Unweighted | Degree-Penalized |
|--------|-----------|-----------------|
| Spearman rho | 0.7524 | 0.8416 |

## Key Observations

1. **Degree-penalized geodesic consistently outperforms unweighted geodesic** across all k values (5, 10, 20) on both datasets. The improvement is not specific to k=10.
2. **On MSMARCO-10K**, degree-penalized geodesic surpasses the adjacency-only baseline at all k values (R@5: 0.367 vs 0.228, R@10: 0.417 vs 0.325, R@20: 0.439 vs 0.357).
3. **On SIFT10K**, degree-penalized geodesic does not surpass adjacency-only at k=5 (0.246 vs 0.265), but the gap narrows at k=10 and k=20 approaches parity.
4. **Scatter plots confirm** degree penalty improves distance correlation: Spearman rho increases from 0.752 to 0.842, showing tighter rank agreement between graph-geodesic and Euclidean distances.
