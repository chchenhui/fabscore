# Optimization Iteration 0: Overlap-Resampled L-BFGS on 2D Poisson

## Experiment Overview

Optimized the overlap-resampled L-BFGS method for the 2D Poisson forward problem. The original method (rel L2: 5.56e-3) was 16x worse than fixed-LBFGS (3.42e-4) and comparable to Adam-only (5.94e-3). Four fixes applied:

1. **3-phase training**: Added Adam+fixed-collocation intermediate phase (5000 budget) to stabilize the model before L-BFGS
2. **Higher overlap fraction**: Increased from 0.5 to 0.9, reducing stochastic noise in the loss landscape
3. **Lambda_bc scheduling**: lambda_bc=1.0 (Adam), 5.0 (Adam+fixed), 10.0 (L-BFGS) to enforce boundary conditions during L-BFGS
4. **Increased budget**: 50000 total (15000 Adam + 5000 Adam+fixed + 30000 overlap-LBFGS)

## Setup

- Budget: 50000 (15000 Adam + 5000 Adam+fixed + 30000 overlap-LBFGS)
- overlap_frac: 0.9
- lambda_bc: 1.0 (Adam), 5.0 (Adam+fixed), 10.0 (L-BFGS)
- Network: [2, 50, 50, 50, 50, 1], tanh, FP64
- N_coll: 2000, N_per_edge: 200
- Seeds: 0, 1, 2

## Key Results

| Seed | Best rel L2 | Best Step | Cautious Skips | Termination |
|------|------------|-----------|----------------|-------------|
| 0 | 8.29e-4 | 46506 | 243 (2.9%) | budget_exhausted |
| 1 | 7.29e-4 | 49001 | 243 (3.0%) | budget_exhausted |
| 2 | 5.52e-4 | 48501 | 234 (2.8%) | budget_exhausted |

**Mean rel L2: 7.03e-4 +/- 1.14e-4** (7.9x improvement over original 5.56e-3)

### Comparison

| Method | rel L2 (mean +/- std) | vs Fixed-LBFGS |
|--------|----------------------|----------------|
| Adam-only (Task 5) | 5.94e-3 +/- 7.84e-4 | 17.4x worse |
| **Original** overlap-LBFGS (Task 7) | 5.56e-3 +/- 3.62e-3 | 16.3x worse |
| **Optimized** overlap-LBFGS | **7.03e-4 +/- 1.14e-4** | **2.1x worse** |
| Fixed-LBFGS (Task 6) | 3.42e-4 +/- 9.19e-5 | baseline |

## Key Observations

1. **Dramatic improvement**: rel L2 dropped from 5.56e-3 to 7.03e-4 (7.9x better). The optimized overlap-LBFGS is now within 2x of the fixed-LBFGS baseline, a massive reduction from the original 16x gap.

2. **Cautious skip rate**: Dropped from 27-34% to 2.8-3.0%. The 0.9 overlap fraction means only 10% of collocation points change per step, providing much more consistent curvature estimates.

3. **Best checkpoints from L-BFGS phase**: All seeds now achieve their best rel L2 during the overlap-LBFGS phase (steps 46K-49K), confirming that L-BFGS is now effectively driving convergence rather than being stuck as in the original.

4. **Seed consistency**: Standard deviation dropped from 3.62e-3 to 1.14e-4 (32x reduction), showing much more consistent behavior across seeds.

5. **Remaining gap to fixed-LBFGS (2.1x)**: Expected due to the inherent stochasticity of resampling -- even 10% point turnover prevents the exact convergence that fixed collocation achieves (loss ~1e-6 vs ~5e-5).
