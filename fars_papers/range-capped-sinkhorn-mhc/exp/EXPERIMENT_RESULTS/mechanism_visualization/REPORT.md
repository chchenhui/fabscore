# RRCS Gradient Vanishing Prevention Mechanism - Visualization Report

## Experiment Overview

This analysis produces diagnostic figures demonstrating the core RRCS hypothesis: controlling the log-range of Sinkhorn inputs prevents extreme exponentiation and restores gradient flow into H_res_logits. Six figures compare three conditions (mHC default, fixed tau cap-init, RRCS r_cap=2.0) using diagnostic logs from the main experiments (3 seeds x 3 conditions, 5000 iterations each).

## Setup

- **Model**: 48-layer nanoGPT (~20M params) with mHC (Manifold-Constrained Hyper-Connections)
- **Dataset**: FineWeb10B (900M train tokens)
- **Conditions**:
  1. mHC Default: tau=0.05, sinkhorn_iters=10 (unmodified)
  2. Fixed tau Cap-Init: tau=0.2667, constant effective temperature
  3. RRCS: r_cap=2.0, per-step adaptive range capping
- **Seeds**: 42, 123, 456 (primary plots use seed=42, shaded regions show seed variance)
- **Diagnostic interval**: Every 10 steps (501 data points per run)
- **Script**: `mhc_repo/diagnostics/plot_results.py`

## Key Results

### Figure 1: H_res Gradient Norm Time Series (`hres_grad_norm_timeseries.png`)
- **mHC Default**: H_res gradient norms are exactly 0.0 for all layers across all 5000 iterations. Complete gradient vanishing.
- **Cap-Init**: H_res gradient norms are ~1e-15 to ~1e-21, effectively zero. The constant temperature approach does not solve the problem.
- **RRCS**: H_res gradient norms are ~4e-6 (orders of magnitude above the other two conditions). Gradients flow meaningfully through Sinkhorn projection.

### Figure 2a: Sinkhorn Log-Range Distribution (`sinkhorn_log_range_distribution.png`)
- mHC Default: log-range locked at 160 across all layers and steps.
- Cap-Init: log-range locked at ~30 (the initialization range scaled by constant tau).
- RRCS (raw): log-range grows to ~162 (parameters actually learn).
- RRCS (post-cap): effective range is ~2.0 (the cap in action), keeping Sinkhorn outputs soft.

### Figure 2b: Sinkhorn Log-Range Time Series (`sinkhorn_log_range_timeseries.png`)
- mHC Default and Cap-Init show constant log-ranges throughout training (160 and 30 respectively).
- RRCS raw range increases slightly over training (parameters drifting).
- RRCS post-cap effective range stays near 2.0, demonstrating the adaptive capping.

### Figure 3: H_res Parameter Drift (`hres_param_drift.png`)
- **mHC Default**: Zero cumulative drift (gradient is zero, parameters never change).
- **Cap-Init**: Negligible drift (~1e-10 Frobenius, gradient too small to move parameters).
- **RRCS**: Substantial cumulative drift with final Frobenius norm ~4.19. Parameters actively learn throughout training.

### Figure 4: Global Gradient Norm (`global_grad_norm.png`)
- All three conditions have similar global gradient norm profiles (dominated by transformer parameters, not H_res).
- Spike ratios: mHC Default ~1.42, Cap-Init ~1.42, RRCS ~1.34.
- RRCS shows marginally better stability (slightly lower spike ratio).

### Figure 5: Theoretical Range Cap Mechanism (`range_cap_mechanism.png`)
Four-panel illustration:
- (a) Uncapped: exp(Z) for Z in [-200, 0] shows values below 1e-30 for most of the range.
- (b) Range capped (r_cap=30): entries stay above exp(-30) ~ 9e-14, still quite small.
- (c) RRCS (r_cap=2.0): entries stay above exp(-2) ~ 0.135, keeping soft routing.
- (d) Gradient chain schematic showing where near-zero exp(Z) kills gradients.

## Key Observations

1. The core mechanism is confirmed: uncapped Sinkhorn with tau=0.05 produces Z values with range ~160, causing exp(Z_min) to underflow to 0, creating hard permutation matrices with zero gradients.

2. Even the cap-init approach (constant tau=0.2667, range ~30) produces near-zero gradients (~1e-15). The range of 30 still makes exp(-30) ~ 9e-14 negligibly small for gradient purposes.

3. Only RRCS with r_cap=2.0 produces meaningful gradients (~4e-6), because exp(-2) ~ 0.135 keeps the Sinkhorn output soft enough (entropy ~0.93) for gradients to flow.

4. The parameter drift metric directly demonstrates that H_res_logits "actually learn" under RRCS (drift 4.19) vs being frozen under mHC default (drift 0.0) and cap-init (drift ~3e-10).

5. Global gradient norms are similar across conditions, confirming that the gradient vanishing is specific to the Sinkhorn projection path, not a global training instability.
