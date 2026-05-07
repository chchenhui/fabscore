# Degree-Penalized Geodesic + LMDS on SIFT10K

## Experiment Overview

Evaluated the proposed degree-penalized geodesic embedding method on SIFT10K (10,000 128-d SIFT descriptors). This is the core proposed attack: given only the HNSW layer-0 adjacency, assign each edge (u,v) a weight `w(u,v) = 1 + alpha * (log(1+deg(u)) + log(1+deg(v))) / 2` to counteract small-world shortcut distortion from high-degree hub nodes. Then compute landmark shortest-path distances and apply Landmark MDS to recover approximate coordinates.

## Setup

- **Dataset**: SIFT10K -- 10,000 vectors, 128 dimensions
- **HNSW params**: M=32, efConstruction=64, seeds=[42, 123, 456]
- **Reconstruction params**: L=2000 landmarks, d=128 target dimensions, alpha=3.0
- **Evaluation**: Recall@10 and Recall@20 against exact kNN (k=20), Spearman rank correlation (50,000 random pairs) between reconstructed and true Euclidean distances

## Key Results

| Method | Recall@10 | Recall@20 | Spearman rho |
|--------|-----------|-----------|--------------|
| Adjacency-only | 0.3475 +/- 0.0006 | 0.3343 +/- 0.0004 | N/A |
| Unweighted geodesic + LMDS | 0.1603 +/- 0.0016 | 0.1855 +/- 0.0016 | 0.7626 +/- 0.0116 |
| **Degree-penalized geodesic + LMDS** | **0.2684 +/- 0.0030** | **0.2973 +/- 0.0029** | **0.8517 +/- 0.0057** |

## Key Observations

1. The degree-penalized method improves Recall@10 by ~67% over the unweighted geodesic baseline (0.2684 vs 0.1603), confirming that hub-penalty weighting better preserves local neighborhood structure.
2. Spearman correlation improves from 0.7626 to 0.8517, indicating degree-penalized geodesic distances are more faithful to true Euclidean distances.
3. The adjacency-only baseline still achieves higher Recall@10 (0.3475) since it directly uses HNSW neighbors which inherently contain true kNN, while geodesic methods reconstruct coordinates then re-search. However, the gap is now much smaller (0.2684 vs 0.3475), and geodesic methods recover full coordinate-space embeddings that enable richer downstream attacks.
4. Results are consistent across 3 random seeds (low std), indicating robustness of the method.
5. The reconstruction pipeline takes ~10s per seed (dominated by 2000 Dijkstra runs).
