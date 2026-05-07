# HC Unconstrained Calibration Run

## Experiment Overview

Single-seed calibration run of unconstrained Hyper-Connections (HC without doubly-stochastic projection) on Setting A (48-layer, n=4). The purpose is to establish a gradient spike baseline for interpreting the stability metrics of constrained methods (Sinkhorn and orthostochastic mHC).

## Setup

- **Config**: `orthostochastic_mhc_experiments/configs/setting_a_hc_unconstrained.py`
- **Model**: 48 layers, n_embd=150, n_head=6, hc_num_streams=4, ~20.8M params
- **Training**: 5000 iters, batch_size=8, grad_accum=4, 4 GPUs (DDP), bfloat16
- **Seed**: 1
- **Key difference from mHC**: `mhc=False`, `hc_disable=False` -- uses vanilla HC without any doubly-stochastic constraint on H_res

## Key Results

| Metric | Unconstrained HC | mHC-Sinkhorn (5-seed mean) | mHC-Orthostochastic (5-seed mean) |
|--------|------------------|---------------------------|----------------------------------|
| Best Val Loss | 4.7104 | 4.7615 +/- 0.0094 | 4.7642 +/- 0.0125 |
| r_max | **3.1387** | 1.9125 +/- 0.2436 | 1.8665 +/- 0.1305 |
| r_max ratio vs constrained | -- | **1.64x** | **1.65x** |

## Key Observations

1. **Gradient spike severity**: The unconstrained HC r_max (3.1387) is approximately 1.64x higher than the constrained methods (~1.91 Sinkhorn, ~1.87 Orthostochastic). This confirms that the doubly-stochastic constraint meaningfully reduces gradient spikes.

2. **Spike distribution**: Out of 4800 spike ratio measurements:
   - 10 exceeded 2.0 (0.21%)
   - 2 exceeded 2.5 (0.04%)
   - 1 exceeded 3.0 (0.02%) -- the r_max event
   - Mean spike ratio: 1.037, p95: 1.329, p99: 1.647

3. **Training stability**: Despite higher gradient spikes, the unconstrained HC run did NOT diverge. Training completed all 5000 iterations with decreasing loss. This is consistent with the grad_clip=1.0 clipping preventing catastrophic divergence.

4. **Val loss**: The unconstrained HC achieved a slightly lower best val loss (4.7104) than both constrained methods (~4.76). This is expected -- the unconstrained model has more freedom in H_res parameters, but with only 1 seed this difference is not statistically significant.

5. **Calibration conclusion**: The r_max values from constrained methods (Sinkhorn: 1.91, Orthostochastic: 1.87) are meaningfully lower than unconstrained HC (3.14), confirming that the doubly-stochastic constraint provides measurable gradient stability improvement (~39% reduction in peak spike ratio). Both constrained methods show similar stability characteristics.
