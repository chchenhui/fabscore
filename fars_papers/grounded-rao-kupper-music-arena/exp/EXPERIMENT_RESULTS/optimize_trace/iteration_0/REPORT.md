# GRK Optimization Iteration 0

## Experiment Overview

Optimized the Grounded Rao-Kupper (GRK) model on Music Arena by:
1. Adding probability renormalization to handle P(TIE) clipping
2. Extending the model with per-system tie propensity parameters (gamma_k)
3. Tuning hyperparameters via 5-fold CV
4. Improving optimizer convergence (tighter tolerances, better initialization)

## Issues Found and Fixed

### Issue 1: Probability Renormalization
P(TIE) = 1 - P(A) - P(B) - P(BOTH_BAD) can be clipped to epsilon, breaking the probability simplex. Fixed by renormalizing all 4 probabilities to sum to 1 after clipping.

### Issue 2: Per-System Tie Propensity (gamma extension)
The original model uses a single global lambda. Extended with per-system gamma_k:
lam_ij = base_lam * exp(gamma_i + gamma_j), regularized with L2 on gamma.
This allows different systems to have different tie tendencies while preserving the grounded BOTH_BAD structure.

### Issue 3: CV-Tuned Regularization
Added 5-fold CV to select best l2_gamma. Best: l2_gamma=0.1 (CV NLL=1.0340 vs baseline 1.0355).

### Issue 4: Optimizer Improvements
Tighter convergence (ftol=1e-14, gtol=1e-10, maxiter=10000), better lambda initialization (log(0.3) instead of 0).

## Key Results

### Global Test Set (983 battles)

| Metric | Original GRK | Optimized GRK | Change |
|--------|-------------|---------------|--------|
| 4-way NLL | 0.9556 | **0.9526** | -0.0030 |
| BOTH_BAD Brier | 0.1862 | 0.1867 | +0.0005 |
| BOTH_BAD ECE | 0.1542 | **0.1502** | -0.0040 |
| TIE per-class NLL | 2.4883 | **2.4592** | -0.0291 |
| A-wins NLL | 0.6936 | **0.6899** | -0.0037 |
| B-wins NLL | 0.7209 | **0.7160** | -0.0049 |
| GRK vs AB-MNL NLL diff | -0.0789 | **-0.0819** | larger margin |

### Stratified Results

| Split | Original NLL | Optimized NLL | Change |
|-------|-------------|---------------|--------|
| Global | 0.9556 | **0.9526** | -0.0030 |
| Instrumental | 0.9137 | **0.9111** | -0.0026 |
| Vocal | 1.2953 | **1.2884** | -0.0069 |

### All pairwise bootstrap comparisons remain significant (95% CIs exclude zero).

## Key Observations

1. The gamma extension improves NLL across all three test splits.
2. The improvement is primarily in TIE prediction (NLL 2.49 -> 2.46) and A/B win prediction.
3. BOTH_BAD Brier is essentially unchanged (0.1862 vs 0.1867) -- the core BOTH_BAD modeling is already optimal.
4. The per-system gamma values are small (range: -0.065 to +0.050), consistent with strong regularization.
5. The GRK vs AB-MNL advantage widened from -0.0789 to -0.0819 in NLL.

## Model Configuration

- use_gamma=True, l2_beta=0.0, l2_gamma=0.1
- Lambda: 1.281 (vs original 1.313)
- Total parameters: 25 (12 beta + 12 gamma + 1 lambda) vs original 13
