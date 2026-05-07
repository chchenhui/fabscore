# Adam + Resampling Baseline on 2D Poisson Forward Problem

## Experiment Overview

Evaluate the Adam optimizer with per-step uniform collocation resampling on a canonical 2D Poisson forward PDE problem. This establishes the first-order optimizer baseline for the 2D Poisson benchmark used to test generalization of overlap-resampled L-BFGS beyond the 1D ice-shelf inverse problem.

## Setup

- **PDE**: -Laplacian(u) = f on [0,1]^2, Dirichlet u=0 on boundary
- **Exact solution**: u*(x,y) = sin(pi*x)*sin(pi*y), f = 2*pi^2*sin(pi*x)*sin(pi*y)
- **Network**: MLP [2, 50, 50, 50, 50, 1], tanh activations, Xavier init, FP64
- **Optimizer**: Adam, lr=1e-3
- **Collocation**: N=2000 interior points, resampled uniformly on [0,1]^2 every iteration
- **Boundary**: 800 fixed points (200 per edge), lambda_bc=1.0
- **Compute budget**: 30000 gradient evaluations
- **Evaluation**: Relative L2 error on 100x100 grid, every 500 steps
- **Seeds**: 0, 1, 2
- **Best checkpoint**: Selected by lowest relative L2 error during training

## Key Results

| Seed | Best rel L2 | Best Step | Total Evals |
|------|-------------|-----------|-------------|
| 0    | 7.01e-3     | 25000     | 30000       |
| 1    | 5.15e-3     | 25500     | 30000       |
| 2    | 5.65e-3     | 30000     | 30000       |

**Mean +/- Std: 5.94e-3 +/- 7.84e-4**

## Key Observations

1. Training loss decreased from ~8.3 to ~1e-4 over 30000 steps, with noisy per-step fluctuations typical of collocation resampling.
2. Relative L2 error improved from ~14.9% at step 500 to ~0.6% by step 25000-30000.
3. All seeds converged without NaN/Inf issues.
4. Best checkpoints were found late in training (step 25000-30000), suggesting the error was still slowly decreasing at budget exhaustion.
5. BC loss converged to ~3-5e-5, much smaller than PDE loss (~1e-4 to 1e-3), indicating boundary conditions are well satisfied.
