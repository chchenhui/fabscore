# r_cap Ablation Study: RRCS Sensitivity to Range Cap Threshold

## Experiment Overview

Ablation study varying the range cap threshold r_cap in RRCS (Range-Restricted Capped Sinkhorn) across three values: r_cap=20 (tighter cap), r_cap=30 (default), and r_cap=40 (looser cap). All runs use seed=42 on the same 48-layer nanoGPT architecture trained for 5000 iterations. The r_cap=30 result is reused from the Main Experiment (Task 3).

The goal is to answer: How sensitive is RRCS to the choice of r_cap? Is there a clear optimal range, or is performance robust across values?

## Setup

- **Model**: nanoGPT 48-layer, n_embd=150, n_head=6, ~20.8M params
- **Dataset**: FineWeb10B (900M train tokens, 100M val tokens, GPT-2 tokenizer)
- **Training**: 5000 iters, batch_size=8, grad_accum=4, lr=6e-4 cosine decay, warmup=200
- **mHC config**: hc_num_streams=4, sinkhorn_tau=0.05, sinkhorn_iters=10, mhc_rrcs=True
- **Ablation variable**: mhc_r_cap in {20.0, 30.0, 40.0}
- **Seed**: 42 (single seed per r_cap)
- **Hardware**: 1x GPU per run, ~3.2 hours each
- **Diagnostics**: diag_interval=10 (every 10 steps)

## Key Results

### Comparison Table

| r_cap | Val Loss | H_res Grad Median | ||dH_res||_F | DS Error | Entropy | % Steps Capped | s_mean |
|-------|----------|-------------------|-------------|----------|---------|----------------|--------|
| 20    | 4.7708   | 4.33e-11          | 7.91e-03    | 0.0      | ~1.2e-7 | 100%           | 0.125  |
| 30    | 4.7722   | 1.82e-15          | 2.10e-07    | 0.0      | 0.0     | 100%           | 0.1875 |
| 40    | 4.7712   | 1.05e-19          | 1.22e-11    | 0.0      | ~2.9e-16| 100%           | 0.25   |

### Scaling Factors

The initial Sinkhorn logit range is ~160, so the scaling factor s = r_cap / range:
- r_cap=20: s ~ 0.125 (scale down by 8x)
- r_cap=30: s ~ 0.1875 (scale down by ~5.3x)
- r_cap=40: s ~ 0.25 (scale down by 4x)

### Gradient Flow Ordering

A clear monotonic relationship exists: smaller r_cap produces larger gradients:
- r_cap=20: grad_median = 4.33e-11 (4 orders of magnitude larger than r_cap=30)
- r_cap=30: grad_median = 1.82e-15 (baseline)
- r_cap=40: grad_median = 1.05e-19 (4 orders of magnitude smaller than r_cap=30)

Each 10-unit increase in r_cap reduces gradients by ~4 orders of magnitude. This follows from the exponential relationship: gradients scale as exp(-r_cap) through the Sinkhorn softmax.

### Parameter Drift

Drift follows the same exponential pattern:
- r_cap=20: ||dH_res||_F = 7.91e-03 (meaningful but small)
- r_cap=30: ||dH_res||_F = 2.10e-07 (negligible)
- r_cap=40: ||dH_res||_F = 1.22e-11 (effectively zero)

### Validation Loss

All three r_cap values achieve essentially identical validation loss (~4.771), differing by < 0.002. This confirms that at these large cap values, the routing matrix is still a near-permutation and does not meaningfully affect the model's representational capacity. The val loss is dominated by the residual stream, not the H_res routing.

## Key Observations

1. **RRCS is extremely sensitive to r_cap in this regime**: Each 10-unit change in r_cap shifts gradient magnitude by ~4 orders of magnitude. This exponential sensitivity arises because the Sinkhorn projection maps range-r_cap logits through softmax, where off-diagonal entries scale as exp(-r_cap).

2. **All tested r_cap values (20, 30, 40) are too large for meaningful routing**: Even r_cap=20 produces gradients of only ~4e-11 and parameter drift of ~8e-3. Compare this to r_cap=2.0 from the optimization experiment (Task 4/5) which achieves gradients of ~4e-6 and drift of ~4.2. The transition to "effective" routing happens at much smaller r_cap values.

3. **Val loss is robust across r_cap in this regime**: Since all three caps still produce near-permutation matrices, the model's performance is unaffected. The routing simply doesn't contribute meaningfully to the loss landscape at these cap values.

4. **100% of steps are capped for all three values**: The raw logit range (~160) exceeds all tested thresholds, so RRCS capping is always active. The s_mean values (0.125, 0.1875, 0.25) are constant, confirming the range doesn't change much during training with these cap values.

5. **Practical implication**: For RRCS to be effective, r_cap should be much smaller (e.g., r_cap=2.0 as found in the optimization study). The r_cap=20-40 range tested here acts almost identically to uncapped Sinkhorn (r_cap=inf), with only exponentially small differences in gradient flow.

## Figures

- `results/figures/rcap_ablation_val_loss.png` -- Val loss curves (nearly overlapping)
- `results/figures/rcap_ablation_grad_norm.png` -- H_res gradient norm time series (log scale, clear separation)
- `results/figures/rcap_ablation_log_range.png` -- Sinkhorn input log-range over training
