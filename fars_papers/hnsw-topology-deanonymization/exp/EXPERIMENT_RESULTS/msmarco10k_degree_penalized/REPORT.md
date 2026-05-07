# Degree-Penalized Geodesic + LMDS on MSMARCO-10K

## Experiment Overview

Evaluate the degree-penalized geodesic embedding method on MSMARCO-10K, a dataset of 10,000 768-dimensional text embeddings from sentence-transformers/msmarco-distilbert-base-v2. This is the critical high-dimensional test: text embeddings have very different intrinsic geometry compared to SIFT features, testing whether HNSW topology in 768-d preserves enough metric information for reconstruction.

## Setup

- **Dataset**: MSMARCO-10K (10,000 vectors, 768-d)
- **HNSW params**: M=32, efConstruction=64
- **Seeds**: 42, 123, 456
- **Reconstruction**: Degree-penalized edge weights (alpha=4.0) + shortest-path distances + Landmark MDS
  - L=3000 landmarks, d=128 target dimensions
  - Edge weight: w(u,v) = 1 + alpha * (log(1+deg(u)) + log(1+deg(v))) / 2
- **Metrics**: Recall@10, Recall@20 (against true kNN), Spearman correlation (50k random pairs)

## Key Results

| Method | Recall@10 | Recall@20 | Spearman rho |
|--------|-----------|-----------|--------------|
| Adjacency-only | 0.3248 +/- 0.0007 | 0.3567 +/- 0.0002 | N/A |
| Unweighted geodesic + LMDS | 0.2731 +/- 0.0020 | 0.3006 +/- 0.0022 | 0.4783 +/- 0.0081 |
| **Degree-penalized geodesic + LMDS** | **0.4164 +/- 0.0003** | **0.4387 +/- 0.0002** | **0.5010 +/- 0.0070** |

## Key Observations

1. **Degree-penalized surpasses the adjacency-only baseline**: Recall@10 of 0.4164 exceeds the adjacency-only baseline (0.3248) by 28%. This is a qualitative breakthrough: the topology-based reconstruction attack recovers more true kNN relationships than the raw HNSW adjacency list provides directly.

2. **Degree penalty provides 52% improvement over unweighted**: Recall@10 increases from 0.2731 (unweighted) to 0.4164 (degree-penalized), confirming that hub-penalty weighting is essential for faithful distance preservation in high-dimensional spaces.

3. **Spearman correlation improves moderately**: From 0.4783 (unweighted) to 0.5010 (degree-penalized), indicating the degree penalty better preserves global distance rank ordering.

4. **Results are highly consistent**: Very low variance across 3 seeds (std < 0.003 for recall), indicating the method is robust to HNSW construction randomness.

5. **Reconstruction takes ~21s per seed**: Dominated by 3000 Dijkstra runs on the 10K-node graph.
