# Adjacency-Only Baseline on SIFT10K

## Experiment Overview

Evaluate the adjacency-only baseline on SIFT10K (n=10,000, d=128). This baseline treats the leaked HNSW neighbor lists directly as each node's predicted neighbor set and measures Recall@k against the true kNN computed from the original vectors. It establishes the lower bound of what an attacker learns from the raw leaked topology without any reconstruction.

## Setup

- **Dataset**: SIFT10K -- 10,000 base vectors, 128 dimensions (TexMex corpus)
- **HNSW Parameters**: M=32, efConstruction=64
- **Seeds**: [42, 123, 456] (different insertion orders)
- **Ground Truth**: Exact k-NN (k=20) computed via FAISS IndexFlatL2
- **Metrics**: Recall@k for k in {5, 10, 20}

## Key Results

| Metric     | Mean   | Std    |
|------------|--------|--------|
| Recall@5   | 0.2656 | 0.0006 |
| Recall@10  | 0.3475 | 0.0006 |
| Recall@20  | 0.3343 | 0.0004 |

### Per-Seed Breakdown

| Seed | Recall@5 | Recall@10 | Recall@20 |
|------|----------|-----------|-----------|
| 42   | 0.2649   | 0.3470    | 0.3349    |
| 123  | 0.2663   | 0.3484    | 0.3341    |
| 456  | 0.2656   | 0.3471    | 0.3339    |

## Key Observations

1. **Low variance across seeds**: std < 0.001 for all k values, indicating the baseline is stable regardless of insertion order.
2. **Recall@20 < Recall@10**: With an average node degree of ~16 (max 2M=64 neighbors), many nodes have fewer than 20 HNSW neighbors, so the adjacency list cannot supply enough candidates for k=20 recall.
3. **Recall@10 ~ 0.35**: The raw HNSW adjacency captures about 35% of true 10-nearest neighbors. This establishes the lower bound for the proposed reconstruction methods.
4. **Degree statistics**: min=1, max=76-97, mean=15.9-16.0 across seeds.
