# mHC Default Baseline Experiment

## Experiment Overview

Ran the unmodified mHC baseline (sinkhorn_tau=0.05, sinkhorn_iters=10) on a 48-layer nanoGPT (~20M params) trained on FineWeb10B for 5000 iterations. This establishes the reference condition demonstrating that H_res_logits gradients completely vanish under the default Sinkhorn temperature, confirming the core motivation for Range-Capped Sinkhorn (RRCS).

## Setup

- **Model**: nanoGPT 48-layer, n_embd=150, n_head=6, ~20.8M params
- **Dataset**: FineWeb10B (900M train tokens, 100M val tokens, GPT-2 tokenizer)
- **Training**: 5000 iters, batch_size=8, grad_accum=4, lr=6e-4 cosine decay to 6e-5, warmup=200
- **mHC config**: hc_num_streams=4, sinkhorn_tau=0.05, sinkhorn_iters=10
- **Seeds**: 42, 123, 456
- **Hardware**: 1x A100-SXM4-80GB per seed (3 runs in parallel)
- **Runtime**: ~2.9 hours per seed
- **Diagnostics**: logged every 10 steps (H_res gradient norms, Sinkhorn conditioning, DS error, entropy)

## Key Results

### Validation Loss
| Seed | Best Val Loss | Final Val Loss |
|------|---------------|----------------|
| 42   | 4.7717        | 4.7717         |
| 123  | 4.7859        | 4.7938         |
| 456  | 4.7577        | 4.7577         |
| **Mean +/- Std** | **4.771 +/- 0.011** | **4.774 +/- 0.015** |

### Gradient Flow (H_res_logits)
- **Median gradient norm (all layers, all steps post-warmup)**: 0.0 for all seeds
- **Mean gradient norm**: 0.0 for all seeds
- Gradients are **exactly zero** due to Sinkhorn outputting a numerically exact permutation matrix

### Parameter Drift (H_res_logits)
- **||H_res_logits(T=5000) - H_res_logits(T=0)||_F**: 0.0 for all seeds (across all 96 HC layers)
- H_res_logits **never change** during training

### Sinkhorn Input Conditioning
- **Log-range r = max(Z) - min(Z)**: 160.0 for all layers, all steps, all seeds
- This equals 8.0 / 0.05 = 160, the initial conditioning (diagonal=0, off-diagonal=-8, divided by tau=0.05)
- Since logits never change, the conditioning never changes

### Doubly-Stochastic Error
- **Max row-sum deviation**: 0.0
- **Max col-sum deviation**: 0.0
- The Sinkhorn output is numerically exact identity matrix (a perfect permutation)

### H_res Entropy
- **Mean row entropy**: 0.0 (exact permutation = zero entropy)

### Training Stability
- **Grad norm spike ratio**: 1.22 - 1.55 (very stable, no spikes)
- No NaN, no loss explosions

## Key Observations

1. **Complete gradient vanishing**: With sinkhorn_tau=0.05 and the default initialization (diagonal=0, off-diagonal=-8), the Sinkhorn-projected H_res is an exact identity matrix. Autograd gradients through Sinkhorn are exactly 0.0 -- the H_res_logits parameters are effectively frozen.

2. **Routing is fixed throughout training**: Since gradients are zero, H_res_logits never move from initialization. The 4-stream residual routing is always identity (each stream maps to itself), meaning the mHC "width connections" provide no benefit over standard residual connections.

3. **The model still trains well**: Despite frozen routing, the model achieves val loss ~4.77. This is because the attention and MLP layers still learn normally -- only the H_res routing is frozen. The depth connections (H_post) use softmax (not Sinkhorn) and do receive gradients.

4. **Root cause is extreme conditioning**: r = max(Z) - min(Z) = 160 means the Sinkhorn matrix is exp(160) times more peaked on one entry than others. This makes the output a numerically exact permutation, with zero gradient everywhere.
