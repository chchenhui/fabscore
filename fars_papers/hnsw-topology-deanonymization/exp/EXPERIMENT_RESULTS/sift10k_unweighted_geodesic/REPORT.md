# Unweighted Geodesic + Landmark MDS Baseline on SIFT10K

## Experiment Overview

Ablation baseline evaluating whether shortest-path (hop) distances on the HNSW layer-0 graph, with unit edge weights, combined with Landmark MDS embedding can recover kNN neighborhoods from topology alone. This isolates the contribution of degree-penalized edge weights in the proposed method.

## Setup

- **Dataset**: SIFT10K (10,000 vectors, d=128)
- **HNSW params**: M=32, efConstruction=64, 3 seeds (42, 123, 456)
- **Reconstruction**: Landmark MDS with L=2000 landmarks, d=128 target dimensions
- **Edge weights**: Unit (1.0) -- unweighted shortest paths
- **kNN evaluation**: k=10 (primary), k=20; FAISS IndexFlatL2 on reconstructed coordinates
- **Spearman correlation**: 50,000 random node pairs, reconstructed Euclidean vs true Euclidean distances

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| Recall@10 | 0.1603 | 0.0016 |
| Recall@20 | 0.1855 | 0.0016 |
| Spearman correlation | 0.7626 | 0.0116 |

### Comparison with Other Methods

| Method | Recall@10 | Recall@20 |
|--------|-----------|-----------|
| Adjacency-only | 0.3475 +/- 0.0006 | 0.3343 +/- 0.0004 |
| Unweighted geodesic + LMDS | 0.1603 +/- 0.0016 | 0.1855 +/- 0.0016 |
| Degree-penalized geodesic + LMDS | 0.2684 +/- 0.0030 | 0.2973 +/- 0.0029 |

## Key Observations

1. The unweighted geodesic + LMDS reconstruction achieves 0.160 Recall@10, below the adjacency-only baseline (0.348) but showing meaningful topology-to-geometry reconstruction.

2. The degree-penalized method achieves 67% higher Recall@10 (0.268 vs 0.160), confirming that hub-penalty weighting is important for recovering local neighborhood structure from HNSW topology.

3. Low variance across seeds (std < 0.002) indicates robust performance.

4. Reconstruction takes ~9s per seed (2000 Dijkstra runs + LMDS on 10K nodes).
