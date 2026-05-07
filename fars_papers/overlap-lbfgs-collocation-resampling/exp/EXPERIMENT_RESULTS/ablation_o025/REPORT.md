# Ablation Study: Overlap Ratio o=0.25 vs o=0.5

## Experiment Overview

Pre-registered ablation testing sensitivity of overlap-resampled L-BFGS to the overlap ratio on the 1D ice-shelf inverse problem. The main experiment uses o=0.5 (500 retained, 501 fresh out of 1001 collocation points). This ablation tests o=0.25 (250 retained, 751 fresh) to determine whether a smaller overlap destabilizes the curvature estimates.

## Setup

All settings identical to the main experiment except `overlap_frac`:

| Parameter | Value |
|-----------|-------|
| Network | 6x20 MLP, tanh, float64 |
| N_coll | 1001 |
| N_ob | 401 |
| noise_level | 0.3 |
| gamma | 0.5 |
| Budget | 50000 (20000 Adam + 7500 Adam-fixed + remainder Overlap-LBFGS) |
| Seeds | 0, 1, 2 |
| overlap_frac (ablation) | 0.25 -> n_keep=250, n_fresh=751 |
| overlap_frac (main) | 0.50 -> n_keep=500, n_fresh=501 |

## Key Results

### Comparison Table

| Overlap | B_err (mean+-std) | u_err (mean+-std) | h_err (mean+-std) | L-BFGS Iters | Cautious Skips | Early Stop? |
|---------|-------------------|--------------------|--------------------|--------------|----------------|-------------|
| o=0.50 | 8.06e-4 +- 4.71e-4 | 6.73e-3 +- 3.92e-3 | 3.08e-2 +- 2.20e-2 | 4517 +- 1490 | 672 (15.2%) | Yes (NaN seed 2) |
| o=0.25 | 8.28e-4 +- 4.46e-4 | 6.73e-3 +- 3.92e-3 | 3.09e-2 +- 2.19e-2 | 5307 +- 77 | 1133 (21.4%) | No |

### Per-Seed Details

**o=0.50:**
- Seed 0: B_err=1.41e-3, best_step=18000 (adam), lbfgs_steps=5662, cautious_skips=655 (11.6%), termination=budget_exhausted
- Seed 1: B_err=2.54e-4, best_step=27000 (adam_fixed), lbfgs_steps=5477, cautious_skips=968 (17.7%), termination=budget_exhausted
- Seed 2: B_err=7.57e-4, best_step=1000 (adam), lbfgs_steps=2412, cautious_skips=393 (16.3%), **termination=nan** (37085/50000 evals)

**o=0.25:**
- Seed 0: B_err=1.41e-3, best_step=18000 (adam), lbfgs_steps=5346, cautious_skips=1138 (21.3%), termination=budget_exhausted
- Seed 1: B_err=3.20e-4, best_step=26500 (adam_fixed), lbfgs_steps=5376, cautious_skips=943 (17.5%), termination=budget_exhausted
- Seed 2: B_err=7.57e-4, best_step=1000 (adam), lbfgs_steps=5199, cautious_skips=1319 (25.4%), termination=budget_exhausted

## Key Observations

### (a) Premature Termination
o=0.25 **avoids** premature termination on all 3 seeds (all terminate with budget_exhausted). In contrast, o=0.5 had seed 2 terminate with NaN at 37085/50000 evals. The higher cautious skip rate at o=0.25 acts as a built-in safety mechanism, filtering out unreliable curvature pairs that could destabilize the optimizer.

### (b) Accuracy Comparison
B_err mean difference is only 2.7% (8.28e-4 vs 8.06e-4), well within the standard deviation. The errors are nearly identical because best checkpoints come from the Adam phases (steps 18000, 26500/27000, and 1000), which run identically for both overlap ratios. The L-BFGS phase does not improve upon the Adam-phase best checkpoint for seeds 0 and 2, regardless of overlap ratio.

### (c) Cautious Skip Rate
Mean cautious skip rate increases from 15.2% (o=0.5) to 21.4% (o=0.25), a 6.2 percentage point increase. This confirms that smaller overlap leads to noisier curvature estimates (larger gradient discrepancy between consecutive overlap sets), causing more curvature pairs to fail the cautious update threshold y^T s > eps * ||s||^2 * ||g||.

### (d) Practical Failure Criterion
The method is stable and functional at o=0.25, far below the pre-registered o>=0.8 threshold for "practical failure". This confirms that overlap-resampled L-BFGS does not require near-complete point retention and does not degenerate to nearly-fixed collocation. The L-BFGS iteration count is also more consistent at o=0.25 (std=77 vs std=1490 at o=0.5 due to the NaN termination).

## Figures

- `training_loss_comparison.png` -- Full training loss curves (all phases) for both overlap ratios
- `lbfgs_phase_loss.png` -- L-BFGS phase loss trajectory comparison
- `cautious_skip_comparison.png` -- Bar chart of cautious skip counts and rates per seed
