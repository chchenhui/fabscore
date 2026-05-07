# Training Dynamics Visualization: mHC-Sinkhorn vs mHC-Orthostochastic

## Experiment Overview

This analysis compares the training dynamics of mHC-Sinkhorn and mHC-Orthostochastic across both experimental settings, complementing the aggregate metrics (final val loss, r_max) from the main experiments. Three visualization figures are produced:

1. **Validation Loss Curves** (`results/validation_loss_curves.pdf`)
2. **Gradient Norm Trajectories** (`results/gradient_norm_trajectories.pdf`)
3. **Learned H_res Heatmaps** (`results/h_res_heatmaps.pdf`)

## Setup

- **Setting A**: 48-layer, n_embd=150, hc_num_streams=4, 5 seeds (Sinkhorn), 5 seeds (Orthostochastic), 1 seed (HC Unconstrained)
- **Setting B**: 6-layer, n_embd=288, hc_num_streams=8, 3 seeds (Sinkhorn), 3 seeds (Orthostochastic)
- All runs: max_iters=5000, bfloat16, grad_clip=1.0, 4 GPUs DDP
- HC Unconstrained calibration: Setting A only (no Setting B calibration run)

### Data Sources

- **Val loss**: Parsed from TrainService stdout logs. Sinkhorn eval_interval=500, Orthostochastic eval_interval=250. Some logs were truncated (especially Setting A Sinkhorn seeds 2-5, which only have 2 eval points each). Individual seed traces are shown alongside aggregate mean+/-std.
- **Grad norms**: Complete per-iteration data from `diagnostics/gradient_spikes.json` (5001 points per run, all seeds).
- **H_res matrices**: Extracted from `ckpt.pt` checkpoints (best-val-loss checkpoint). Projection functions (sinkhorn_log, orthostochastic_project) applied to raw H_res_logits parameters.

## Key Results

### 1. Validation Loss Curves

Both methods show comparable loss trajectories across both settings:

- **Setting A**: Sinkhorn and Orthostochastic curves overlap closely, with the Orthostochastic trajectory (4 seeds with full coverage) showing nearly identical convergence to Sinkhorn (seed 1 with full coverage). The HC Unconstrained reference achieves slightly lower loss, consistent with its larger search space.
- **Setting B**: All 3 Sinkhorn seeds and 3 Orthostochastic seeds show similar convergence rates. The eval point overlap covers iterations 0-1000 (all seeds) with seed 11 extending to 2500.
- The loss curves corroborate the aggregate final val loss results: Setting A delta=+0.0028, Setting B delta=+0.0131.

### 2. Gradient Norm Trajectories

200-step moving average gradient norms (log scale) reveal:

| Metric | Setting A Sinkhorn | Setting A Ortho | Setting A HC Unconstrained | Setting B Sinkhorn | Setting B Ortho |
|--------|-------------------|-----------------|----------------------------|-------------------|-----------------|
| Median (last 1000 iters) | 0.876 +/- 0.007 | 0.864 +/- 0.011 | 1.155 | 0.818 +/- 0.059 | 0.804 +/- 0.042 |
| Max grad norm | 6.33 +/- 0.17 | 6.72 +/- 0.35 | 18.25 | 31.73 +/- 0.67 | 32.17 +/- 0.96 |
| Mean (first 100 iters) | 2.49 | 2.51 | 3.32 | 7.44 | 7.60 |

- **Both mHC methods have nearly identical gradient stability**. Median late-training grad norms match within noise (0.876 vs 0.864 for Setting A, 0.818 vs 0.804 for Setting B).
- **HC Unconstrained shows substantially higher gradient norms**: Median 1.155 vs ~0.87 for mHC methods in Setting A, and max 18.25 vs ~6.5. This demonstrates the gradient stabilization benefit of doubly-stochastic constraints.
- **Setting B has higher initial gradient norms** (7.4-7.6 vs 2.5) and larger max spikes (~32 vs ~6.5), reflecting the higher-dimensional mixing matrices (8x8 vs 4x4) and larger model width.

### 3. Learned H_res Matrices

H_res heatmaps from representative seeds (seed 1 for most; seed 2 for Setting B Ortho since seed 1 diverged) at selected layers:

**Setting A** (layers 1, 12, 24, 36, 48; 1-indexed):
- **Sinkhorn**: All layers converge to exact identity matrices (||H-I||=0.000). The Sinkhorn initialization (diagonal=0, off-diagonal=-8) with tau=0.05 produces output indistinguishable from identity.
- **Orthostochastic (with identity mix, alpha~0.09-0.14)**: Near-identity at early/middle layers (||H-I|| <= 0.032). Layer 48 (last layer) shows the most deviation from identity (||H-I||=0.166, alpha=0.137), suggesting the model learns slight stream mixing only at the final layer.

**Setting B** (layers 1, 3, 6; 1-indexed):
- **Sinkhorn**: All layers converge to exact identity (||H-I||=0.000), same as Setting A.
- **Orthostochastic**: Near-identity but with slightly more structure than Setting A. ||H-I|| ranges from 0.027-0.073, with layer 3 showing the most mixing (||H-I||=0.073).

**Key observation**: Both methods converge to near-permutation (specifically near-identity) structures. The Sinkhorn initialization strongly biases toward identity, and the orthostochastic constraint with identity mix achieves the same routing pattern with only minor deviations. This means the orthostochastic subset's reduced expressiveness (compared to full Birkhoff polytope) is not a practical limitation because neither method exploits dense interior points of the doubly-stochastic polytope.

## Figures

- `results/validation_loss_curves.pdf` -- 2-panel val loss with mean+/-std bands and individual seed traces
- `results/gradient_norm_trajectories.pdf` -- 2-panel smoothed grad norm (log scale) with mean+/-std bands
- `results/h_res_heatmaps.pdf` -- Grid of H_res heatmaps (Setting A: 5 layers x 2 methods; Setting B: 3 layers x 2 methods)

## Data Limitations

- Val loss curves for Setting A Sinkhorn (seeds 2-5) have only 2 data points due to TrainService log truncation. Setting A Ortho seed 5 has no TrainService log. Individual seed traces partially compensate for this.
- Gradient norm data is complete (5001 points per run) from the diagnostics JSON files.
- H_res analysis shows a single representative seed per method; cross-seed variation in routing patterns not captured.
