# Adjacency-Only Baseline on MSMARCO-10K

## Experiment Overview

Evaluate the adjacency-only baseline on MSMARCO-10K (n=10,000, d=768) to test whether HNSW topology leakage remains meaningful in a high-dimensional text-embedding regime. For each node, the first k neighbors from its HNSW adjacency list are used as predicted kNN and compared against exact kNN ground truth.

## Setup

- **Dataset**: MS MARCO v1.1 passage corpus (microsoft/ms_marco on HuggingFace)
  - 626,907 unique passages extracted from train split
  - 10,000 randomly sampled (seed=42)
  - Embedded with `sentence-transformers/msmarco-distilbert-base-v2` (d=768)
- **HNSW Parameters**: M=32, efConstruction=64
- **Seeds**: [42, 123, 456] (different insertion orders)
- **Ground Truth**: Exact 20-NN via FAISS IndexFlatL2
- **Metrics**: Recall@{5, 10, 20}

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| Recall@5 | 0.2284 | 0.0009 |
| **Recall@10** | **0.3248** | **0.0007** |
| Recall@20 | 0.3567 | 0.0002 |
| Avg Degree | 21.04 | 0.21 |

## Key Observations

1. **MSMARCO-10K vs SIFT10K Comparison**: The adjacency-only baseline achieves Recall@10 = 0.3248 on MSMARCO-10K (768-d), comparable to SIFT10K's 0.3475 (128-d). The ~6.5% drop suggests topology leakage is slightly less effective in higher dimensions but still substantial.
2. **Recall@20 slightly higher than Recall@10**: 0.3567 > 0.3248, consistent with the fact that HNSW neighbors (avg degree ~21) cover more of the true 20-NN set than the true 10-NN set.
3. **Low variance across seeds**: Std < 0.001 for all recall metrics, indicating robust results independent of insertion order.
4. **Average degree ~21**: Higher than SIFT10K's ~16, likely due to the higher dimensionality causing more symmetric neighbor relationships in the HNSW graph.
