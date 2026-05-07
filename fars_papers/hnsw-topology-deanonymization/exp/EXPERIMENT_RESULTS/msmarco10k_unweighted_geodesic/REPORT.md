# Unweighted Geodesic + Landmark MDS Baseline on MSMARCO-10K

## Experiment Overview

Ablation baseline evaluating whether shortest-path (hop) distances on the HNSW layer-0 graph, with unit edge weights, combined with Landmark MDS embedding can recover kNN neighborhoods from topology alone in the high-dimensional (768-d) text-embedding regime. This isolates the contribution of degree-penalized edge weights in the proposed method.

## Setup

- **Dataset**: MSMARCO-10K (10,000 vectors, d=768, sentence-transformers/msmarco-distilbert-base-v2)
- **HNSW params**: M=32, efConstruction=64, 3 seeds (42, 123, 456)
- **Reconstruction**: Landmark MDS with L=3000 landmarks, d=128 target dimensions
- **Edge weights**: Unit (1.0) -- unweighted shortest paths
- **kNN evaluation**: k=10 (primary), k=20; FAISS IndexFlatL2 on reconstructed coordinates
- **Spearman correlation**: 50,000 random node pairs, reconstructed Euclidean vs true Euclidean distances

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| Recall@10 | 0.2731 | 0.0020 |
| Recall@20 | 0.3006 | 0.0022 |
| Spearman correlation | 0.4783 | 0.0081 |

### Comparison with Other MSMARCO-10K Methods

| Method | Recall@10 | Recall@20 |
|--------|-----------|-----------|
| Adjacency-only | 0.3248 +/- 0.0007 | 0.3567 +/- 0.0002 |
| Unweighted geodesic + LMDS (L=3000, d=128) | 0.2731 +/- 0.0020 | 0.3006 +/- 0.0022 |
| Degree-penalized geodesic + LMDS (L=3000, d=128, alpha=4.0) | 0.4164 +/- 0.0003 | 0.4387 +/- 0.0002 |

## Key Observations

1. The unweighted geodesic + LMDS reconstruction achieves 0.273 Recall@10, below the adjacency-only baseline (0.325) but showing meaningful topology-to-geometry reconstruction.

2. The degree-penalized method achieves 52% higher Recall@10 (0.416 vs 0.273), confirming hub-penalty weighting is critical for accurate reconstruction in high-dimensional spaces.

3. Low variance across seeds (std ~0.002) indicates robust performance.

4. Reconstruction takes ~20s per seed (3000 Dijkstra runs + LMDS on 10K nodes).
