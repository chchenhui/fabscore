# mHC-Sinkhorn Baseline: Setting B (6-layer, hc_num_streams=8)

## Experiment Overview

Baseline experiment for mHC-Sinkhorn on the 6-layer nanoGPT configuration (Setting B) with hc_num_streams=8. This setting tests the wider-streams regime (n=8) where permutation-mixture parameterizations become factorial. The Sinkhorn-Knopp projection with 10 iterations and tau=0.05 is used for the H_res doubly-stochastic constraint.

## Setup

- **Model**: nanoGPT, 6 layers, n_embd=288, n_head=6, block_size=1024 (~20M params)
- **Dataset**: FineWeb10B (GPT-2 tokenized, 10 train shards used)
- **Training**: max_iters=5000, batch_size=32, grad_accum=4, bfloat16
- **Optimizer**: AdamW, lr=6e-4 with cosine decay to 6e-5, warmup=200 iters
- **mHC config**: hc_num_streams=8, mhc_h_res_proj="sinkhorn", sinkhorn_iters=10, sinkhorn_tau=0.05
- **Hardware**: 4x GPU (DDP), ~18 minutes per run
- **Seeds**: 1, 2, 3

## Key Results

| Metric | Mean | Std | Seed 1 | Seed 2 | Seed 3 |
|--------|------|-----|--------|--------|--------|
| Final val loss | 4.2529 | 0.0164 | 4.2656 | 4.2589 | 4.2343 |
| Best val loss | 4.2495 | 0.0133 | 4.2554 | 4.2589 | 4.2343 |
| r_max (grad spike) | 1.9508 | 0.2130 | 2.1832 | 1.9041 | 1.7650 |
| DS error (last 200) | ~0.0 | 0.0 | ~0.0 | ~0.0 | ~0.0 |

## Key Observations

1. **Val loss**: Final validation loss of 4.2529 +/- 0.0164 across 3 seeds. The best val loss (4.2495) is nearly identical to final, indicating continued improvement throughout training without overfitting.
2. **Gradient stability**: r_max ~1.95, comparable to Setting A Sinkhorn (1.91), indicating stable training without gradient spikes.
3. **DS constraint**: Sinkhorn achieves effectively zero DS error (~6e-8), as expected with 10 Sinkhorn iterations.
4. **Convergence**: All 3 seeds converged to similar val losses (spread ~0.03), indicating stable and reproducible training.
5. **Efficiency**: Each run completed in ~18 minutes on 4 GPUs (5000 iters at ~200ms/iter).
