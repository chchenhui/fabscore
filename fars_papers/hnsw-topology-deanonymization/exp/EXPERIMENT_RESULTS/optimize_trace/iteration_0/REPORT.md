# Optimization Iteration 0: LMDS Bug Fix + Full Hyperparameter Optimization

## Experiment Overview

Two rounds of optimization: (1) Fixed a critical Landmark MDS column indexing bug and optimized SIFT10K hyperparameters, (2) Optimized MSMARCO-10K hyperparameters with the bug already fixed.

## Issues Found and Fixed

### Issue 1: Critical LMDS Column Indexing Bug (SIFT10K + MSMARCO-10K)

In `landmark_mds.py`, the landmark-to-landmark submatrix was extracted as `distance_matrix[:, :L]`, assuming landmarks are the first L columns. Since landmarks are randomly selected, the correct extraction is `distance_matrix[:, landmark_indices]`.

**Fix**: Added `landmark_indices` parameter to `landmark_mds()`.

### Issue 2: Suboptimal Hyperparameters on Both Datasets

SIFT10K: L=256, d=32, alpha=1.0 -> optimized to L=2000, d=128, alpha=3.0.
MSMARCO-10K: L=256, d=32, alpha=1.0 -> optimized to L=3000, d=128, alpha=4.0.

## Key Results

### SIFT10K (128-d)

| Method | Before | After | Improvement |
|--------|--------|-------|-------------|
| Degree-penalized R@10 | 0.0944 | **0.2684** | 2.84x |
| Unweighted R@10 | 0.0638 | 0.1603 | 2.51x |
| Adjacency-only R@10 | 0.3475 | (unchanged) | - |

### MSMARCO-10K (768-d)

| Method | Before | After | Improvement |
|--------|--------|-------|-------------|
| Degree-penalized R@10 | 0.1748 | **0.4164** | 2.38x |
| Unweighted R@10 | 0.0972 | 0.2731 | 2.81x |
| Adjacency-only R@10 | 0.3248 | (unchanged) | - |

## Key Observations

1. On MSMARCO-10K, the degree-penalized method now **surpasses** the adjacency-only baseline (0.416 > 0.325), a qualitative breakthrough.
2. On SIFT10K, degree-penalized approaches adjacency-only (0.268 vs 0.348).
3. The degree-penalized advantage over unweighted is consistent: +67% on SIFT10K, +52% on MSMARCO-10K.
4. Higher alpha works better for high-dimensional data (4.0 for 768-d vs 3.0 for 128-d).
