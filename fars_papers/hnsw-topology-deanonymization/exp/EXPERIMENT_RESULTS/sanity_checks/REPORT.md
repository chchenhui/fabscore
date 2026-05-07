# Sanity Checks: ER Random Graph Control and M Sensitivity

## Experiment Overview

Two sanity checks to validate that HNSW topology reconstruction results are driven by genuine metric structure rather than trivial graph properties. All experiments use SIFT10K (n=10000, d=128).

## Setup

- **Dataset**: SIFT10K (10,000 128-d SIFT descriptors)
- **HNSW params**: M=32, efConstruction=64, seed=42
- **Reconstruction params**: alpha=1.0, L=256 landmarks, d=32 components
- **Metric**: Recall@10 against exact kNN ground truth

### Adjacency-only Recall Definition

For comparing across different M values, adjacency-only Recall@10 measures the fraction of true 10-NN that appear anywhere in a node's full neighbor set: `|true_top10 ∩ all_neighbors| / 10`. This avoids penalizing denser graphs where sorted-by-ID truncation to k=10 would be arbitrary.

## Key Results

### 1. Erdos-Renyi Random Graph Control

| Metric | Value |
|--------|-------|
| HNSW avg degree | 15.97 |
| ER avg degree | 15.94 |
| Chance Recall@10 | 0.0010 |
| ER Adjacency-only Recall@10 | 0.0013 (1.29x chance) |
| ER Degree-penalized Recall@10 | 0.0011 (1.09x chance) |

**Conclusion**: Both ER methods yield near-chance Recall@10, confirming that metric structure in the HNSW topology is essential for reconstruction. A random graph with identical size and average degree contains no useful metric information.

### 2. M Sensitivity (M in {16, 32, 64})

| M | Avg Degree | Adjacency-only Recall@10 | Degree-penalized Recall@10 |
|---|-----------|--------------------------|---------------------------|
| 16 | 15.37 | 0.5167 | 0.1826 |
| 32 | 15.85 | 0.5159 | 0.1869 |
| 64 | 97.05 | 0.8743 | 0.1331 |

**Adjacency-only**: Increases substantially from M=16/32 (~0.516) to M=64 (0.874). M=16 and M=32 have similar avg degree (~15-16) because efConstruction=64 limits neighbor acquisition for larger M; the jump occurs at M=64 where the layer-0 capacity (128) is substantially filled.

**Degree-penalized geodesic**: Does NOT increase monotonically. M=64 (0.133) is lower than M=16 (0.183) and M=32 (0.187). The very dense graph (avg degree 97) produces short, non-discriminative geodesic distances, degrading MDS embedding quality.

## Key Observations

1. **ER control validates metric dependence**: The >400x gap between HNSW Recall@10 (~0.52) and ER Recall@10 (~0.001) confirms that reconstruction success is driven by metric structure encoded in HNSW edges, not by generic graph properties.

2. **Adjacency-only scales with graph density**: More HNSW neighbors trivially capture more true kNN, validating that denser indices leak more metric information through their topology.

3. **Geodesic reconstruction has a density sweet spot**: Overly dense graphs (M=64, avg degree ~97) compress geodesic distances into a narrow range, making MDS embedding less discriminative. This suggests an optimal M range for geodesic-based reconstruction, likely where graph diameter is large enough for meaningful distance variation.

4. **efConstruction=64 is a bottleneck for M=32**: With efConstruction=64, both M=16 (layer-0 capacity 32) and M=32 (capacity 64) produce similar effective graphs (~16 avg degree). Higher efConstruction would differentiate them more.
