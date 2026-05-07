# Hyperparameter Ablation: Alpha and Landmark Sensitivity

## Experiment Overview

Ablation study examining the sensitivity of the degree-penalized geodesic method to two key hyperparameters:
- **Hub-penalty coefficient alpha**: Controls how much high-degree (hub) nodes are penalized in edge weights
- **Number of landmarks L**: Controls how many landmark nodes are used for Dijkstra + LMDS

Conducted on SIFT10K with seed=42 and d=32 for computational efficiency.

## Setup

- Dataset: SIFT10K (10,000 vectors, 128-d)
- HNSW adjacency: seed=42, M=32, efConstruction=64
- Embedding dimension: d=32
- Evaluation: Recall@10 and Spearman correlation vs true kNN

## Key Results

### Alpha Sensitivity (L=256 fixed)

| alpha | Recall@10 | Spearman | Time (s) |
|-------|-----------|----------|----------|
| 0.00  | 0.1212    | 0.8019   | 0.94     |
| 0.25  | 0.1565    | 0.8368   | 1.10     |
| 0.50  | 0.1731    | 0.8455   | 1.23     |
| 1.00  | 0.1852    | 0.8514   | 1.24     |
| 2.00  | 0.1932    | 0.8550   | 1.14     |
| 4.00  | 0.1964    | 0.8571   | 1.14     |

### Landmark Sensitivity (alpha=1.0 fixed)

| L    | Recall@10 | Spearman | Time (s) |
|------|-----------|----------|----------|
| 64   | 0.1309    | 0.8020   | 0.21     |
| 128  | 0.1653    | 0.8444   | 0.39     |
| 256  | 0.1852    | 0.8514   | 0.84     |
| 512  | 0.1955    | 0.8545   | 1.87     |
| 1024 | 0.1988    | 0.8559   | 4.27     |

## Key Observations

1. **Alpha effect**: Recall@10 increases monotonically with alpha from 0.0 to 4.0, with diminishing returns beyond alpha=1.0. The degree penalty is clearly beneficial: alpha=0.0 (unweighted) yields 0.1212 vs 0.1964 at alpha=4.0 -- a 62% relative improvement. alpha=1.0 captures 85% of the total alpha-0-to-4 improvement range.

2. **Landmark effect**: Recall@10 increases with L but with strong diminishing returns. Going from L=64 to L=256 gains +0.054 recall, but L=256 to L=1024 gains only +0.014. Time scales roughly linearly with L.

3. **Default assessment**: Both defaults (alpha=1.0, L=256) are reasonable:
   - alpha=1.0 is within 0.011 of the best (alpha=4.0), and captures most improvement over unweighted
   - L=256 is within 0.014 of the best (L=1024), and offers the best accuracy-speed trade-off (0.8s vs 4.3s)

4. **Spearman correlation** tracks Recall@10 closely, increasing from 0.80 to 0.86 across both sweeps, confirming that the hyperparameters affect both neighborhood recovery and distance preservation similarly.

## Artifacts

- `results/analysis_alpha_sensitivity.json` -- per-alpha results
- `results/analysis_landmark_sensitivity.json` -- per-L results
- `results/analysis_hyperparameter_summary.json` -- combined summary
- `results/figures/alpha_sensitivity.pdf` -- alpha sweep plot
- `results/figures/landmark_sensitivity.pdf` -- landmark sweep plot (dual-axis)
