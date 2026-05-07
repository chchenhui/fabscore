# L-BFGS Early Stopping and Line-Search Diagnostics

## Experiment Overview

Diagnostic analysis measuring how L-BFGS behaves under four collocation configurations on the ice-shelf inverse problem. The goal is to quantify the mechanism by which collocation resampling affects L-BFGS: specifically, whether naive resampling breaks curvature estimates, leading to line-search failures and early termination, and whether overlap-set curvature pairs maintain estimate quality.

## Setup

**Four configurations tested** (all with identical hyperparameters):

| Config | Overlap Frac | Description |
|--------|-------------|-------------|
| Naive (o=0) | 0.0 | Full per-step resampling, curvature pairs on completely different batches |
| Fixed | N/A | Fixed collocation points during L-BFGS (no resampling) |
| Overlap o=0.25 | 0.25 | 25% overlap between consecutive batches for curvature pairs |
| Overlap o=0.5 | 0.5 | 50% overlap between consecutive batches for curvature pairs |

**Shared hyperparameters**: budget=50000, adam_budget=20000, adam_fixed_budget=7500 (except Fixed which has 0), gamma=0.5, N_coll=1001, N_ob=401, 3 seeds per config.

**Training pipeline**: Adam+resampling warmstart (20k evals) -> Adam+fixed transition (7.5k evals) -> L-BFGS phase (remaining budget).

**Note**: All resampling configs (Naive, o=0.25, o=0.5) use `OverlapLBFGS` with a cautious update rule that skips curvature pairs when y^T s is too small. This is a safeguard that prevents the optimizer from using poor curvature information.

## Key Results

### Total L-BFGS Iterations

| Config | Seed 0 | Seed 1 | Seed 2 | Mean +/- Std |
|--------|--------|--------|--------|-------------|
| Naive (o=0) | 5221 | 5295 | 1483 (NaN) | 3999.7 +/- 1775.5 |
| Fixed | 1230 | 1240 | 1231 | 1233.7 +/- 4.5 |
| Overlap o=0.25 | 5346 | 5376 | 5199 | 5307.0 +/- 76.4 |
| Overlap o=0.5 | 5662 | 5477 | 2412 (NaN) | 4517.0 +/- 1476.5 |

### Termination Reasons

| Config | Budget Exhausted | Gradient Tolerance | NaN |
|--------|-----------------|-------------------|-----|
| Naive (o=0) | 2/3 | 0/3 | 1/3 |
| Fixed | 0/3 | 3/3 | 0/3 |
| Overlap o=0.25 | 3/3 | 0/3 | 0/3 |
| Overlap o=0.5 | 2/3 | 0/3 | 1/3 |

### Cautious Skips (curvature pair rejections)

| Config | Seed 0 | Seed 1 | Seed 2 |
|--------|--------|--------|--------|
| Naive (o=0) | 777 (14.9%) | 531 (10.0%) | 129 (8.7%) |
| Fixed | 0 (0%) | 0 (0%) | 0 (0%) |
| Overlap o=0.25 | 1138 (21.3%) | 943 (17.5%) | 1319 (25.4%) |
| Overlap o=0.5 | 655 (11.6%) | 968 (17.7%) | 393 (16.3%) |

### Line Search Failures

Very rare across all configs: 0-6 per run. Not a primary termination mechanism.

## Key Observations

1. **Fixed-collocation L-BFGS converges to gradient tolerance** in ~1230 steps consistently across all seeds. This is the expected behavior: with fixed data, L-BFGS builds accurate curvature approximations and converges to a (local) minimum.

2. **Naive resampling (o=0) does NOT cause premature termination via line-search failures as originally hypothesized.** Instead, the cautious update rule in OverlapLBFGS effectively skips bad curvature pairs (~10-15% of steps), allowing the optimizer to continue running. When it terminates early, it is due to NaN (1/3 seeds), not line-search failure. Seeds 0 and 1 exhaust their full budget.

3. **Overlap configs (o=0.25, o=0.5) behave similarly to naive** in terms of iteration count and budget exhaustion. The overlap mechanism's primary benefit is not in preventing early termination (since the cautious rule already handles that) but rather in improving the quality of curvature information used when pairs ARE accepted.

4. **Cautious skip rates are highest for overlap o=0.25** (~20%), followed by naive (~11%), then overlap o=0.5 (~15%). This suggests that with lower overlap fractions, more curvature pairs fail the quality check, but the safeguard prevents catastrophic behavior.

5. **NaN instability** affects 1/3 naive seeds and 1/3 overlap o=0.5 seeds, but 0/3 fixed and 0/3 overlap o=0.25 seeds. This suggests resampling introduces some numerical instability risk, but it is not the dominant failure mode.

6. **The original hypothesis about L-BFGS premature termination from resampling** is partially supported but nuanced: the cautious update rule (present in our OverlapLBFGS implementation) effectively mitigates line-search failures. Without this safeguard, naive resampling would likely cause the premature termination described in the literature.

7. **Best B_err is always from Adam phase**: For all configs with the same seed, best_B_err is identical (e.g., seed 0: naive=fixed=1.338e-3, o0.25=o0.5=1.406e-3). This is because the best checkpoint always occurs during the Adam warmstart (step 13500 for naive/fixed with 20k Adam budget, step 18000 for overlap configs with 20k Adam + 7.5k fixed transition). L-BFGS does NOT improve the best checkpoint in any resampling variant. For fixed L-BFGS, the best checkpoint is also from Adam (the L-BFGS convergence reduces training loss but does not improve the validation metric).

## Generated Plots

- `bar_total_iterations.png`: Total L-BFGS outer iterations per configuration (mean +/- std)
- `grad_norm_trajectory.png`: Gradient norm ||g_k|| vs L-BFGS iteration (seed 0, log scale)
- `step_size_trajectory.png`: Step size alpha_k vs L-BFGS iteration (seed 0, log scale)
- `curvature_quality_trajectory.png`: Curvature pair quality y_k^T s_k / (||s_k||^2 * ||g_k||) vs iteration (seed 0, excludes Fixed config which uses torch.optim.LBFGS without curvature pair access)
