# Adam + Collocation Resampling Baseline on Ice-Shelf Inverse Problem

## Experiment Overview

Strong first-order baseline: Adam optimizer with per-step collocation resampling for the full compute budget on the 1D ice-shelf inverse problem. This evaluates whether Adam alone with resampling can achieve good accuracy without quasi-Newton refinement.

## Setup

- **Method**: Adam (lr=1e-3, betas=(0.9, 0.999)) with per-step collocation resampling
- **Problem**: 1D ice-shelf inverse problem -- infer B(x) from noisy u(x), h(x)
- **Network**: MLP [1, 20, 20, 20, 20, 20, 20, 3], tanh activations, Xavier init, FP64
- **Collocation**: N=1001 points, uniform [0,1] cubed, resampled every step
- **Loss**: gamma * PDE_loss + (1-gamma) * data_loss, gamma/(1-gamma) = 0.1 (gamma ~ 0.0909)
- **Data**: k=401 observation points, noise_level=0.3, ground-truth B(x)=1
- **Compute budget**: 30,000 forward+backward gradient evaluations
- **Seeds**: 3 seeds (0, 1, 2) controlling both network init and data noise
- **Best checkpoint**: Selected by lowest B_err during training (evaluated every 500 steps)

## Key Results

| Seed | B_err | u_err | h_err | Best Step |
|------|-------|-------|-------|-----------|
| 0 | 6.51e-4 | 2.02e-2 | 5.13e-2 | 27500 |
| 1 | 2.45e-4 | 1.60e-3 | 9.11e-3 | 19500 |
| 2 | 1.69e-3 | 1.17e-2 | 5.46e-2 | 500 |

**Mean +/- Std (from best checkpoints):**
- B_err: 8.63e-4 +/- 6.10e-4
- u_err: 1.12e-2 +/- 7.61e-3
- h_err: 3.83e-2 +/- 2.07e-2

## Key Observations

1. **B_err variability**: B_err varies substantially across seeds (2.45e-4 to 1.69e-3), with a coefficient of variation ~71%. Seed 2 achieves its best B_err at step 500 and degrades afterward, highlighting the importance of best-checkpoint selection.

2. **Non-monotonic B_err**: For seed 2, B_err increased from 1.69e-3 (step 500) to ~4.7e-3 (step 30000), while u_err and h_err continued to decrease. This suggests Adam+resampling can overfit to data loss at the expense of B recovery.

3. **Loss convergence**: Total loss converges to ~0.168 for all seeds. The PDE loss fluctuates due to collocation resampling (expected behavior).

4. **Clustering analysis**: With only 3 seeds, k-means clustering is not applicable (falls back to median/IQR reporting). More seeds would be needed for clustering fraction analysis.

5. **Compute budget**: Each seed consumed exactly 30,000 gradient evaluations as designed. Total wall-clock time: ~7.5 minutes for all 3 seeds on 1 A100 GPU.
