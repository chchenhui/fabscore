# Adam Warmstart + Fixed-Collocation L-BFGS Baseline on 2D Poisson Forward Problem

## Experiment Overview

Two-phase hybrid baseline: Adam optimizer with per-step collocation resampling (warmstart), then switch to standard full-batch L-BFGS with fixed collocation points (refinement). This mirrors the ice-shelf Adam→L-BFGS baseline but on a canonical 2D Poisson forward PDE, testing whether the proposed overlap-resampled L-BFGS method generalizes beyond inverse problems.

## Setup

- **PDE**: -Laplacian(u) = f on [0,1]^2, Dirichlet u=0 on boundary
- **Exact solution**: u*(x,y) = sin(pi*x)*sin(pi*y), f = 2*pi^2*sin(pi*x)*sin(pi*y)
- **Network**: MLP [2, 50, 50, 50, 50, 1], tanh activations, Xavier init, FP64
- **Phase 1 (Adam + resampling)**: Adam lr=1e-3, ResampleSampler2D N=2000 uniform on [0,1]^2, 15000 gradient evaluations
- **Phase 2 (L-BFGS + fixed collocation)**: torch.optim.LBFGS, history_size=20, strong_wolfe, tolerance_grad=1e-11, tolerance_change=1e-14, N=2000 fixed points drawn at transition, 15000 gradient evaluations (closure calls)
- **Boundary**: 800 fixed points (200 per edge), lambda_bc=1.0
- **Compute budget**: B_poisson = 30000 gradient evaluations total (50/50 split)
- **Evaluation**: Relative L2 error on 100x100 grid, every 500 budget evals
- **Seeds**: 0, 1, 2
- **Best checkpoint**: Selected by lowest relative L2 error during training

## Key Results

| Seed | Best rel L2 | Best Step | Adam Evals | L-BFGS Steps | L-BFGS Closures | Termination |
|------|-------------|-----------|------------|--------------|-----------------|-------------|
| 0    | 4.63e-4     | 30000     | 15000      | 620          | 15000           | gradient_tolerance |
| 1    | 2.40e-4     | 21005     | 15000      | 626          | 15000           | gradient_tolerance |
| 2    | 3.23e-4     | 21506     | 15000      | 624          | 15000           | gradient_tolerance |

**Mean +/- Std: 3.42e-4 +/- 9.19e-5**

## Budget Calibration

- **B_poisson = 30000 gradient evaluations** (total budget consumed by this method)
- All seeds consumed exactly 30000 gradient evaluations (15000 Adam + 15000 L-BFGS closure calls)
- The Adam-only resampling baseline (Task 5) used the same 30000 budget, so no re-run needed for budget calibration

## Key Observations

1. **Dramatic improvement over Adam-only baseline**: Adam→L-BFGS achieves rel L2 = 3.42e-4 vs Adam-only = 5.94e-3, a ~17x improvement. Unlike the ice-shelf problem where L-BFGS hurt B_err, here L-BFGS substantially improves the solution quality.

2. **Best checkpoints from L-BFGS phase**: For all 3 seeds, the best rel L2 was achieved during the L-BFGS phase (steps 21005-30000), not during Adam warmstart. This contrasts sharply with the ice-shelf result where best checkpoints were all from the Adam phase.

3. **L-BFGS convergence behavior**: L-BFGS completed ~620-626 outer steps using all 15000 allocated closure calls before hitting gradient tolerance. Each step consumed ~22-24 closure calls (strong Wolfe line search), consistent with the ice-shelf behavior.

4. **Forward vs inverse problem dynamics**: On the forward Poisson problem (no data loss, only PDE + BC), L-BFGS with fixed collocation steadily improves solution quality. The pathological behavior seen on ice-shelf (L-BFGS reducing total loss but degrading B_err) does not occur here, likely because the forward problem has a cleaner loss landscape without the data-PDE tension of inverse problems.

5. **Error reduction trajectory**: Rel L2 error progressed from ~14.9% (step 500, Adam) → ~1.0% (step 15000, end of Adam) → ~0.034% (best during L-BFGS). The L-BFGS phase achieved ~30x additional error reduction beyond Adam's endpoint.

6. **Comparison with ice-shelf**: On ice-shelf, Adam→L-BFGS (B_err=1.15e-3) was worse than Adam-only (B_err=8.63e-4). On 2D Poisson, Adam→L-BFGS (rel L2=3.42e-4) is dramatically better than Adam-only (rel L2=5.94e-3). This establishes that L-BFGS refinement is highly effective on forward problems but can hurt on inverse problems.
