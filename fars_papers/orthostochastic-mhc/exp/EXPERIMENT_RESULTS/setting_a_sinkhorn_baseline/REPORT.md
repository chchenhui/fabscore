# mHC-Sinkhorn Baseline on Setting A (48-Layer, hc_num_streams=4)

## Experiment Overview

Baseline experiment for the orthostochastic mHC project. Trains nanoGPT with mHC using Sinkhorn-Knopp H_res projection on FineWeb10B. 5 seeds establish reference metrics for comparison with orthostochastic variant.

## Setup

- **Model**: nanoGPT, 48 layers, n_embd=150, n_head=6, ~20.8M params
- **Data**: FineWeb10B (10 train shards used, 1B tokens; 1 val shard, 100M tokens)
- **Training**: max_iters=5000, batch_size=8, grad_accum=4, block_size=1024
- **Optimizer**: AdamW, lr=6e-4, cosine decay to 6e-5, warmup=200 iters
- **Precision**: bfloat16
- **HC Config**: hc_num_streams=4, mhc=True, mhc_h_res_proj="sinkhorn", sinkhorn_iters=10, sinkhorn_tau=0.05
- **Hardware**: 4x GPU per run (DDP), 5 runs in parallel
- **Seeds**: [1, 2, 3, 4, 5]

## Key Results

| Metric | Mean | Std | Per-seed values |
|--------|------|-----|-----------------|
| Final val loss | 4.7631 | 0.0116 | [4.7778, 4.7599, 4.7721, 4.7503, 4.7552] |
| Best val loss | 4.7615 | 0.0094 | [4.7699, 4.7599, 4.7721, 4.7503, 4.7552] |
| r_max | 1.9125 | 0.2436 | [1.684, 1.774, 1.795, 2.023, 2.286] |
| DS error (avg last 200) | 0.0000 | 0.0000 | [0.0, 0.0, 0.0, 0.0, 0.0] |

## Key Observations

1. **Validation loss**: Consistent across seeds with very low std (0.0116), indicating stable training.
2. **r_max**: Moderate gradient spike ratios (mean 1.91), indicating good training stability. Values range from 1.68 to 2.29.
3. **DS error**: Exactly 0.0 for all seeds -- Sinkhorn-Knopp produces exact doubly-stochastic matrices (in bfloat16 precision, errors round to zero).
4. **Training duration**: ~55 minutes per run on 4 GPUs.
5. **Data note**: Used 10 of 103 available training shards (1B of 10B tokens) to fit in container memory. With 5000 iters * 32K tokens/iter = 164M tokens consumed, 1B available tokens provides sufficient diversity.
