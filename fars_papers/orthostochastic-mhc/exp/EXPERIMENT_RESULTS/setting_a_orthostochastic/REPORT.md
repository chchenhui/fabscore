# mHC-Orthostochastic on Setting A (48-Layer, hc_num_streams=4)

## Experiment Overview

Core hypothesis test for the orthostochastic mHC project. Trains nanoGPT with mHC using orthostochastic H_res projection (Newton-Schulz orthogonalization + elementwise squaring) on FineWeb10B. 5 seeds for comparison with the Sinkhorn baseline.

**Note**: Results updated after optimization iteration 0 which fixed a critical initialization bug and tuned hyperparameters. See `optimize_trace/iteration_0/REPORT.md` for details.

## Setup

- **Model**: nanoGPT, 48 layers, n_embd=150, n_head=6, ~20.8M params
- **Data**: FineWeb10B (10 train shards, 1B tokens; 1 val shard, 100M tokens)
- **Training**: max_iters=5000, batch_size=8, grad_accum=4, block_size=1024
- **Optimizer**: AdamW, lr=6e-4, cosine decay to 6e-5, warmup=200 iters
- **Precision**: bfloat16
- **HC Config**: hc_num_streams=4, mhc=True, mhc_h_res_proj="orthostochastic"
- **NS Config**: ns_steps=15, ns_eps=1e-7, ns_coeffs=(3.0, -3.2, 1.2)
- **Identity Mix**: mhc_residual_identity_mix=True, mhc_residual_alpha=0.1
- **Eval**: eval_interval=250 (20 evaluations), best checkpoint saved on val loss
- **Hardware**: 4x GPU per run (DDP), 5 runs in parallel (20 GPUs total)
- **Seeds**: [1, 2, 3, 4, 5]

## Key Results

### Orthostochastic Results

| Metric | Mean | Std | Per-seed values |
|--------|------|-----|-----------------|
| Final val loss | 4.7656 | 0.0117 | [4.7604, 4.7613, 4.7819, 4.7755, 4.7488] |
| Best val loss | 4.7642 | 0.0125 | [4.7604, 4.7546, 4.7819, 4.7755, 4.7488] |
| r_max | 1.8665 | 0.1305 | [1.704, 2.021, 1.717, 1.956, 1.935] |
| DS max_row_err | 0.0061 | 0.0007 | [0.0057, 0.0059, 0.0053, 0.0062, 0.0073] |
| Orth residual (max) | 0.0078 | 0.0005 | [0.0082, 0.0073, 0.0073, 0.0084, 0.0077] |

### Comparison with Sinkhorn Baseline

| Metric | Sinkhorn (mean +/- std) | Orthostochastic (mean +/- std) | Delta |
|--------|------------------------|-------------------------------|-------|
| Final val loss | 4.7631 +/- 0.0116 | 4.7656 +/- 0.0117 | +0.0025 |
| Best val loss | 4.7615 +/- 0.0094 | 4.7642 +/- 0.0125 | +0.0028 |
| r_max | 1.9125 +/- 0.2436 | 1.8665 +/- 0.1305 | -0.0460 |
| DS error | 0.0000 +/- 0.0000 | 0.0061 +/- 0.0007 | +0.0061 |
| Orth residual | N/A | 0.0078 +/- 0.0005 | - |

### Decision Rule

- sigma_S = 0.0094 (Sinkhorn std of best val loss)
- Delta = +0.0028
- **PROCEED**: delta (0.0028) <= 0.5*sigma_S (0.0047)

## Key Observations

1. **Validation loss**: Orthostochastic now matches Sinkhorn closely (+0.003 best val loss, well within noise). The 92% gap reduction came from fixing the H_res initialization.
2. **Gradient stability (r_max)**: Slightly better than Sinkhorn (1.87 vs 1.91).
3. **DS error**: Small but nonzero (~0.006), within the 1e-2 acceptable range.
4. **Training duration**: ~97 min per run on 4 GPUs (vs ~55 min Sinkhorn, ~70 min original ortho) due to ns_steps=15.
5. **Identity mix**: The (1-alpha)*I + alpha*S interpolation with alpha=0.1 provides a smooth transition from identity to learned DS matrix.
