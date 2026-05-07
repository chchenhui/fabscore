# Fixed Tau Cap-Init Control Experiment

## Experiment Overview

Ran the fixed-tau_cap-init control condition on a 48-layer nanoGPT (~20M params) trained on FineWeb10B for 5000 iterations. This condition uses a constant effective temperature tau_cap_init = 0.2667, computed from the initialization log-range (r_init=160) and r_cap=30:

```
tau_cap_init = tau_base * max(1, r_init / r_cap)
             = 0.05 * max(1, 160 / 30)
             = 0.05 * 5.333
             = 0.2667
```

This matches what RRCS would compute at initialization, but does NOT adapt during training. The purpose is to test the confounder "it is just a better constant tau" -- if this condition matches RRCS on all metrics, per-step adaptation is unnecessary.

## Setup

- **Model**: nanoGPT 48-layer, n_embd=150, n_head=6, ~20.8M params
- **Dataset**: FineWeb10B (900M train tokens, 100M val tokens, GPT-2 tokenizer)
- **Training**: 5000 iters, batch_size=8, grad_accum=4, lr=6e-4 cosine decay to 6e-5, warmup=200
- **mHC config**: hc_num_streams=4, sinkhorn_tau=0.2667, sinkhorn_iters=10
- **Seeds**: 42, 123, 456
- **Hardware**: 1x A100-SXM4-80GB per seed (3 runs in parallel)
- **Runtime**: ~2.9-3.0 hours per seed
- **Diagnostics**: logged every 10 steps (H_res gradient norms, Sinkhorn conditioning, DS error, entropy)

## Key Results

### Validation Loss
| Seed | Best Val Loss | Final Val Loss |
|------|---------------|----------------|
| 42   | 4.7715        | 4.7715         |
| 123  | 4.7842        | 4.7918         |
| 456  | 4.7596        | 4.7596         |
| **Mean +/- Std** | **4.772 +/- 0.010** | **4.774 +/- 0.013** |

### Gradient Flow (H_res_logits)
- **Median gradient norm (all layers, post-warmup)**: ~1.6e-15 for all seeds
- Gradients are numerically negligible (order 1e-15), effectively zero

### Parameter Drift (H_res_logits)
- **||H_res_logits(T=5000) - H_res_logits(T=0)||_F**: ~3.4e-10 (effectively zero)
- H_res_logits do not meaningfully change during training

### Sinkhorn Input Conditioning
- **Log-range r = max(Z) - min(Z)**: 29.996 for all layers, all steps, all seeds
- Exactly as expected: r_init / (tau_cap_init / tau_base) = 160 / (0.2667/0.05) = 160/5.333 = 30

### Doubly-Stochastic Error
- **Max row-sum deviation**: 0.0
- **Max col-sum deviation**: 0.0
- The Sinkhorn output is still a near-exact permutation matrix

### H_res Entropy
- **Mean row entropy**: ~6.5e-12 (effectively zero -- permutation matrix)

### Training Stability
- **Grad norm spike ratio**: 1.25 - 1.52 (very stable, no spikes)
- No NaN, no loss explosions

## Key Observations

1. **Still frozen routing**: Despite reducing tau from 0.05 to 0.2667 (a 5.3x increase), the Sinkhorn input range is still ~30, which is large enough that the output is a near-exact permutation matrix. The H_res_logits gradients are ~1e-15 (vs exactly 0.0 for the default), so routing parameters remain effectively frozen.

2. **Identical val loss to default**: Val loss 4.774 +/- 0.013, virtually identical to the mHC default (4.774 +/- 0.015). This is expected since H_res is frozen in both conditions -- the model trains identically through the non-H_res parameters (attention, MLP).

3. **Range capping at initialization is insufficient**: The cap-init approach targets r_cap=30 exactly, but r=30 is still far too large for meaningful gradient flow. The Sinkhorn output at r=30 is exp(30) ~ 10^13 times more peaked on one entry vs others, still a numerical permutation. To restore gradient flow, per-step adaptive capping (RRCS) is needed to handle the evolving logit landscape as training progresses.

4. **This condition does NOT test the intended hypothesis**: The cap-init condition was designed to test "is a constant higher tau sufficient?" But with the mHC initialization (diag=0, off-diag=-8), even tau=0.2667 gives r=30, which still produces exact permutations. The condition effectively demonstrates that a 5.3x tau increase is not enough to break the permutation-matrix regime. For the cap-init to actually serve as a meaningful control, either r_cap would need to be much smaller (e.g., r_cap=5) or the H_res logits would need to evolve first (which they can't, since gradients are zero).

## Comparison with mHC Default Baseline

| Metric | mHC Default (tau=0.05) | Cap-Init (tau=0.2667) |
|--------|------------------------|------------------------|
| Val Loss (mean +/- std) | 4.774 +/- 0.015 | 4.774 +/- 0.013 |
| H_res grad median | 0.0 | ~1.6e-15 |
| H_res param drift | 0.0 | ~3.4e-10 |
| Sinkhorn range | 160.0 | 30.0 |
| DS error | 0.0 | 0.0 |
| Entropy | 0.0 | ~6.5e-12 |
| Grad spike ratio | 1.2-1.5 | 1.3-1.5 |
