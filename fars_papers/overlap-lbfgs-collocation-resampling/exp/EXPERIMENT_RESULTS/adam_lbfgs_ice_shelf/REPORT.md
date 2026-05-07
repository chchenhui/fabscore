# Adam Warmstart + Fixed-Collocation L-BFGS Baseline on Ice-Shelf Inverse Problem

## Experiment Overview

Two-phase hybrid baseline: Adam optimizer with per-step collocation resampling (warmstart), then switch to standard full-batch L-BFGS with fixed collocation points (refinement). This is the standard workaround for combining resampling reliability with L-BFGS convergence when practitioners cannot use resampling during L-BFGS.

## Setup

- **Method**: Phase 1: Adam (lr=1e-3) + ResampleSampler; Phase 2: L-BFGS + FixedSampler
- **L-BFGS config**: history_size=20, max_iter=20, strong_wolfe, tolerance_grad=1e-11, tolerance_change=1e-14
- **Problem**: 1D ice-shelf inverse problem -- infer B(x) from noisy u(x), h(x)
- **Network**: MLP [1, 20, 20, 20, 20, 20, 20, 3], tanh activations, Xavier init, FP64
- **Collocation**: N=1001 points, uniform [0,1] cubed, resampled during Adam, fixed during L-BFGS
- **Loss**: gamma * PDE_loss + (1-gamma) * data_loss, gamma/(1-gamma) = 0.1 (gamma ~ 0.0909)
- **Data**: k=401 observation points, noise_level=0.3, ground-truth B(x)=1
- **Compute budget**: B=30,000 gradient evaluations total (adam_budget=15,000 + L-BFGS remainder)
- **Seeds**: 3 seeds (0, 1, 2) controlling both network init and data noise
- **Best checkpoint**: Selected by lowest B_err during training (evaluated every 500 budget evals + after each L-BFGS step)

## Key Results

| Seed | B_err | u_err | h_err | Best Step | Adam Evals | L-BFGS Steps | L-BFGS Closures | Termination |
|------|-------|-------|-------|-----------|------------|--------------|-----------------|-------------|
| 0 | 1.33e-3 | 1.65e-2 | 5.81e-2 | 500 | 15000 | 620 | 15000 | gradient_tolerance |
| 1 | 4.15e-4 | 1.04e-3 | 6.70e-3 | 7500 | 15000 | 623 | 15000 | gradient_tolerance |
| 2 | 1.69e-3 | 1.17e-2 | 5.46e-2 | 500 | 15000 | 620 | 15000 | gradient_tolerance |

**Mean +/- Std (from best checkpoints):**
- B_err: 1.15e-3 +/- 5.38e-4
- u_err: 9.73e-3 +/- 6.45e-3
- h_err: 3.98e-2 +/- 2.35e-2

## Key Observations

1. **Best checkpoints all from Adam phase**: For all 3 seeds, the best B_err was achieved during the Adam+resampling warmstart phase (steps 500 or 7500), not during L-BFGS refinement. L-BFGS reduced total loss but did not improve B_err.

2. **L-BFGS convergence behavior**: L-BFGS completed ~620 outer steps using all 15,000 allocated closure calls before hitting gradient tolerance. Each step consumed ~24 closure calls (line-search evaluations), consistent with strong Wolfe conditions on a non-convex landscape.

3. **L-BFGS loss vs B_err divergence**: During the L-BFGS phase, total loss decreased smoothly (from ~0.185 to ~0.160 for seed 0), but B_err increased from its best value. This indicates L-BFGS optimizes the combined loss surface in a direction that improves data fit at the expense of B recovery -- the fixed collocation set may bias the PDE loss landscape.

4. **Comparison with Adam-only baseline**: The Adam-only resampling baseline achieved B_err = 8.63e-4 +/- 6.10e-4. The Adam->L-BFGS hybrid achieves B_err = 1.15e-3 +/- 5.38e-4, slightly worse despite using L-BFGS refinement. The best checkpoints are from the Adam phase in both cases.

5. **Budget accounting**: All seeds consumed exactly 30,000 gradient evaluations (15,000 Adam + 15,000 L-BFGS closure calls). This confirms the compute budget is identical to the Adam-only baseline.
