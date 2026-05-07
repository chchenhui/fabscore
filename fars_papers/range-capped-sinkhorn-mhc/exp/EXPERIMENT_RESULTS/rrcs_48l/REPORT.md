# RRCS (Range-Capped Sinkhorn) Experiment -- Optimized (r_cap=2.0)

## Experiment Overview

Implemented and evaluated the Range-Capped Sinkhorn (RRCS) method on a 48-layer nanoGPT (~20M params) trained on FineWeb10B for 5000 iterations. After the initial experiment with r_cap=30 showed no improvement (gradients still ~0), the r_cap was reduced to 2.0, which successfully restores gradient flow through the Sinkhorn projection.

RRCS rescales Z = logits/tau so that max(Z)-min(Z) <= r_cap before Sinkhorn iterations. With r_cap=2.0, the effective off-diagonal magnitude is exp(-2) ~ 0.14, giving meaningful mixing in the doubly-stochastic routing matrix.

## Setup

- **Model**: nanoGPT 48-layer, n_embd=150, n_head=6, ~20.8M params
- **Dataset**: FineWeb10B (900M train tokens, 100M val tokens, GPT-2 tokenizer)
- **Training**: 5000 iters, batch_size=8, grad_accum=4, lr=6e-4 cosine decay to 6e-5, warmup=200
- **mHC config**: hc_num_streams=4, sinkhorn_tau=0.05, sinkhorn_iters=10
- **RRCS config**: mhc_rrcs=True, mhc_r_cap=2.0
- **Seeds**: 42, 123, 456
- **Hardware**: 1x A100-SXM4-80GB per seed (3 runs in parallel)
- **Runtime**: ~3.3 hours per seed

## Key Results

### Validation Loss
| Seed | Best Val Loss | Final Val Loss |
|------|---------------|----------------|
| 42   | 4.7761        | 4.7761         |
| 123  | 4.7863        | 4.7949         |
| 456  | 4.7626        | 4.7626         |
| **Mean +/- Std** | **4.775 +/- 0.010** | **4.778 +/- 0.013** |

### Gradient Flow (H_res_logits)
- **Median gradient norm (post-warmup)**: ~4.1e-6 for all seeds
- Gradients are **meaningfully nonzero** -- 2.2 billion times larger than with r_cap=30

### Parameter Drift (H_res_logits)
- **||H_res_logits(T=5000) - H_res_logits(T=0)||_F**: ~4.19 mean across seeds
- Drift is **massive** -- 20 million times larger than with r_cap=30

### RRCS Scaling Factor (s)
- **s ~ 0.0124** (= 2.0 / ~162) -- much more aggressive capping than the original 0.1875
- Sinkhorn input range is effectively reduced from ~162 to 2.0

### Doubly-Stochastic Properties
- **DS error**: ~1e-6 (still negligible -- Sinkhorn projection is accurate)
- **Entropy**: ~0.933 (meaningful mixing -- NOT a permutation matrix)

### Training Stability
- **Grad norm spike ratio**: 1.28 - 1.34 (very stable)
- No NaN, no loss explosions

## Key Observations

1. **RRCS with r_cap=2.0 restores gradient flow**: The critical insight is that r_cap=30 was far too conservative. For a 4x4 DS matrix, exp(-30) ~ 1e-13 still produces a near-exact permutation. With r_cap=2.0, exp(-2) ~ 0.14 gives meaningful off-diagonal entries and nonzero gradients.

2. **H_res actually learns**: Parameters drift by ~4.19 Frobenius norm units. Individual logits change by 0.01-0.3, with the most active layers showing substantial routing adaptation.

3. **DS matrices have non-trivial routing**: Entropy of 0.93 means streams are being mixed, not just permuted. This is the qualitative behavior RRCS was designed to enable.

4. **Val loss is unchanged**: 4.775 vs 4.771 baseline -- within seed variance. At this small scale (20M params, 5K iters), the learned routing provides comparable but not yet better performance. The key achievement is that routing IS learned, which was the stated goal.

5. **All success criteria from the proposal are met**:
   - Gradient increase >= 100x: 2.2 billion x (PASS)
   - Param drift >= 10x: 20 million x (PASS)
   - Val loss not worse by > 0.5x baseline std: 0.004 < 0.005 (PASS)

## Comparison with All Conditions

| Condition | Best Val Loss | H_res Grad Median | H_res Drift | Entropy |
|-----------|--------------|-------------------|-------------|---------|
| **RRCS r_cap=2.0** | **4.775 +/- 0.010** | **4.1e-6** | **4.19** | **0.933** |
| RRCS r_cap=5.0 | 4.791 +/- 0.015 | 1.7e-5 | 5.19 | * |
| RRCS r_cap=8.0 | 4.804 +/- 0.015 | 7.6e-6 | 5.35 | * |
| RRCS r_cap=30 | 4.771 +/- 0.009 | 1.9e-15 | 2.1e-7 | 0.0 |
| mHC default | 4.772 +/- 0.011 | 0.0 | 0.0 | 0.0 |
| Cap-init | 4.772 +/- 0.010 | 1.6e-15 | 3.4e-10 | ~0 |
