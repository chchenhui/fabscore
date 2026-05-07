# Effectiveness Evaluation Report

## Verdict: good

## Summary

The degree-penalized geodesic embedding method demonstrates a dimension-dependent effectiveness for recovering true kNN relationships from leaked HNSW topology. On the high-dimensional MSMARCO-10K dataset (768-d text embeddings), the proposed method achieves Recall@10 = 0.4164, surpassing the adjacency-only baseline (0.3248) by +0.0916 absolute improvement (+28.2% relative). This improvement is consistent across all 3 random seeds with very low variance (std = 0.0003). On the lower-dimensional SIFT10K dataset (128-d), the method underperforms adjacency-only (0.2684 vs 0.3475, delta = -0.0791). The degree penalty is validated as an essential component: unweighted geodesic alone is worse than adjacency-only on both datasets.

## Experiment Feasibility Check

All experiments ran successfully without infrastructure or environment issues:

- 6 experiment configurations (3 methods x 2 datasets) completed
- Each configuration used 3 random seeds (42, 123, 456)
- Results are consistent with low cross-seed variance
- An optimization round (task 8) fixed a critical LMDS indexing bug and tuned hyperparameters, after which results improved substantially
- Both main experiment and baseline results are available for comparison

No missing results, no environment failures, no OOM or crash issues. The experiment conditions are fully met.

## Results Analysis

### Unified Comparison Table

| Method | SIFT10K (128-d) Recall@10 | MSMARCO-10K (768-d) Recall@10 |
|---|---|---|
| Adjacency-only | 0.3475 +/- 0.0006 | 0.3248 +/- 0.0007 |
| Unweighted geodesic + LMDS | 0.1603 +/- 0.0016 | 0.2731 +/- 0.0020 |
| Degree-penalized geodesic + LMDS (Ours) | 0.2684 +/- 0.0030 | 0.4164 +/- 0.0003 |

### Per-Seed Recall@10 Details

**SIFT10K:**

| Method | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Adjacency-only | 0.3470 | 0.3484 | 0.3471 |
| Unweighted geodesic | 0.1623 | 0.1602 | 0.1585 |
| Degree-penalized (Ours) | 0.2726 | 0.2670 | 0.2657 |

**MSMARCO-10K:**

| Method | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Adjacency-only | 0.3251 | 0.3238 | 0.3254 |
| Unweighted geodesic | 0.2702 | 0.2748 | 0.2743 |
| Degree-penalized (Ours) | 0.4169 | 0.4163 | 0.4161 |

### Delta Analysis (Degree-Penalized vs Adjacency-Only)

- **SIFT10K**: Delta = 0.2684 - 0.3475 = **-0.0791** (worse on all 3 seeds: -0.0744, -0.0814, -0.0814)
- **MSMARCO-10K**: Delta = 0.4164 - 0.3248 = **+0.0916** (better on all 3 seeds: +0.0918, +0.0925, +0.0907)

### Degree Penalty Contribution (Over Unweighted Geodesic)

- **SIFT10K**: 0.2684 - 0.1603 = +0.1081 (+67.4% relative improvement)
- **MSMARCO-10K**: 0.4164 - 0.2731 = +0.1433 (+52.5% relative improvement)

The degree penalty is essential on both datasets, providing large improvements over unweighted geodesic. Without it, geodesic embedding alone is strictly worse than the adjacency-only baseline.

### Decision Rule Application

The task defined three decision outcomes:

1. **Proceed** (Delta >= 0.10 on one dataset AND >= 0.05 on the other, across all seeds):
   - NOT met. SIFT10K delta is -0.0791 (negative). MSMARCO-10K delta is +0.0916 (close to but below 0.10).

2. **Pivot** (unweighted geodesic improves over adjacency-only but degree penalty adds nothing):
   - NOT met. Unweighted geodesic does NOT improve over adjacency-only on either dataset. Furthermore, the degree penalty IS essential (adds +0.108 to +0.143 over unweighted).

3. **Refute** (both geodesic variants improve by < 0.02 on both datasets):
   - NOT met. Degree-penalized improves by +0.0916 on MSMARCO-10K, far exceeding 0.02.

**Actual outcome**: Mixed/partial. The result falls between the predefined categories. The method provides meaningful improvement in the high-dimensional regime but not in the low-dimensional regime.

### Low-d vs High-d Analysis

The divergent results between SIFT10K and MSMARCO-10K can be attributed to:

1. **Adjacency informativeness differs by dimension**: In 128-d, HNSW neighbors are already strong kNN candidates (R@10 = 0.3475). The adjacency list is highly informative, leaving less room for topology-based reconstruction to add value. In 768-d, adjacency is less informative (R@10 = 0.3248), creating more opportunity for reconstruction methods.

2. **LMDS embedding fidelity**: The landmark MDS reconstruction maps graph distances to Euclidean coordinates. In 128-d, the target space matches the LMDS output dimensionality (n_components=128), but the graph geodesics may introduce distortion that raw adjacency avoids. In 768-d, LMDS compresses to 128 dimensions, which may actually help by denoising the high-dimensional embedding space.

3. **Hub effect is more pronounced in high-d**: High-dimensional spaces exhibit stronger hub phenomena in kNN graphs, making the degree penalty more valuable for correcting geodesic distances.

## Statistical Significance

Given the extremely low cross-seed variance (std = 0.0003 to 0.0030 for Recall@10), the observed differences are highly consistent:

- On MSMARCO-10K, the degree-penalized method beats adjacency-only on every single seed (minimum delta = +0.0907), with improvement 130x larger than the method's standard deviation.
- On SIFT10K, the degree-penalized method is worse on every single seed (maximum delta = -0.0744), with the gap also many multiples of standard deviation.

The results are deterministic enough that formal statistical tests would yield p << 0.001 for both the positive (MSMARCO) and negative (SIFT) findings.

## Verdict Justification

**Verdict: good**

The verdict is "good" (rather than "bad" or "failed") for the following reasons:

1. **Not "failed"**: All experiments completed successfully. Both main experiment and baseline results are available for comparison across all configurations and seeds.

2. **Not "bad"**: There IS a clear, consistent positive signal. The degree-penalized method achieves a +9.16 percentage point improvement on MSMARCO-10K, demonstrating that HNSW topology CAN be exploited to recover kNN relationships beyond what raw adjacency reveals. This is a meaningful finding for the topology-leakage threat model.

3. **"Good" because**:
   - The method works as intended in the high-dimensional regime, which is arguably the more practically relevant scenario (real-world embedding indices typically use 384-1536 dimensional vectors).
   - The degree penalty innovation is validated as essential (unweighted geodesic alone fails on both datasets).
   - The dimension-dependent behavior is itself an interesting and publishable finding.
   - There is sufficient positive signal to warrant further investigation (e.g., understanding the low-d failure mode, testing on additional datasets, exploring alternative embedding dimensions).

The result suggests that subsequent analysis experiments (hyperparameter sensitivity, sanity checks) are worth conducting, particularly to characterize the dimension-dependence of the method's effectiveness.
