# Diagnostic: Overlap-Resampled L-BFGS from Scratch (No Warmstart) on 2D Poisson

## Experiment Overview

This diagnostic isolates the effect of the Adam warmstart by running overlap-resampled L-BFGS directly from random initialization on the 2D Poisson benchmark. The main experiments use a 3-phase approach (Adam+resampling -> Adam+fixed -> overlap-LBFGS); this experiment skips Phase 1 and Phase 2 entirely.

## Setup

- **Architecture**: 4x50 MLP (layers=[2,50,50,50,50,1]), float64, tanh activation
- **Optimizer**: OverlapLBFGS from random initialization (no Adam phase)
  - overlap_frac=0.5, N=2000 collocation points
  - history_size=20, cautious updates (eps=1e-6)
  - Wolfe line search: c1=1e-4, c2=0.9, max_ls=20
- **Lambda_bc**: 10.0 (matching the L-BFGS phase of warmstart experiments)
- **Budget**: 50000 gradient evaluations (same as warmstart experiments)
- **Seeds**: 0, 1, 2
- **Evaluation**: Every 500 gradient evals; best checkpoint by lowest rel L2 error

## Key Results

### Comparison Table

| Configuration | Rel. L2 Error (mean +/- std) | L-BFGS Iters | Cautious Skips | Termination |
|---|---|---|---|---|
| Overlap-LBFGS from scratch | 1.36e-2 +/- 1.74e-2 | 9816 +/- 3589 | 2752 +/- 542 | budget_exhausted |
| Overlap-LBFGS + warmstart | 7.03e-4 +/- 1.14e-4 | 8240 +/- 7 | 240 +/- 4 | budget_exhausted |
| Adam -> fixed-LBFGS | 3.42e-4 +/- 9.19e-5 | 623 +/- 2 | 0 +/- 0 | gradient_tolerance |

### Per-Seed From-Scratch Results

| Seed | Rel L2 | Best Step | LBFGS Iters | Cautious Skips | LS Failures | Status |
|---|---|---|---|---|---|---|
| 0 | 9.48e-4 | 34500 | 12331 | 3400 (27.6%) | 2 | OK |
| 1 | 1.67e-3 | 42508 | 12377 | 2782 (22.5%) | 6 | OK |
| 2 | 3.83e-2 | 11003 | 4740 | 2074 (43.7%) | 107 | Diverged |

## Key Observations

1. **From-scratch partially converges**: Seeds 0 and 1 reach rel L2 of 9.5e-4 and 1.7e-3 respectively -- within 1.3x to 2.4x of the warmstart result (7.03e-4). However, seed 2 gets trapped and eventually diverges (107 line search failures, best rel L2 only 3.83e-2).

2. **High cautious skip rate without warmstart**: From-scratch has 22-44% cautious skip rate vs 2.9% with warmstart. This means the overlap-LBFGS curvature estimates are frequently invalidated when starting from random weights, indicating the loss landscape is poorly conditioned at random initialization.

3. **Warmstart is critical for stability**: The from-scratch approach has 1/3 seeds diverging, while all 3 warmstart seeds converge consistently. Mean rel L2 is 19x worse (1.36e-2 vs 7.03e-4) with very high variance (std/mean = 1.28 vs 0.16).

4. **The overlap mechanism does provide independent benefit**: Even without warmstart, seeds 0 and 1 achieve results that are significantly better than Adam-only (5.94e-3), and seed 0 (9.48e-4) approaches the warmstart result. This suggests the overlap mechanism is not purely dependent on warmstart -- it can function independently, but with reduced reliability.

5. **Warmstart role**: The Adam warmstart is **critical** rather than merely helpful. It provides: (a) initialization near a reasonable basin, reducing cautious skips 10x; (b) stability against divergence; (c) consistent convergence across seeds. The full method's gains come from **both** warmstart and overlap mechanism together.

## Files

- `RESULTS.json`: Structured results and comparison data
- `comparison_table.json`: Raw comparison table data
- `from_scratch_vs_warmstart.png`: Training loss and rel L2 curves
