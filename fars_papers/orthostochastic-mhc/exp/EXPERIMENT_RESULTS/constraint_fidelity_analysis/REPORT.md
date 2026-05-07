# Constraint Fidelity Analysis: DS Error and Orthogonality Residual Trajectories

## Experiment Overview

This analysis characterizes the doubly-stochastic (DS) constraint fidelity and orthogonality
residual trajectories throughout training for mHC-Sinkhorn and mHC-Orthostochastic across
both architectural settings. The key mechanistic question: does the finite-step Newton-Schulz
iteration produce sufficiently orthogonal matrices O such that O*O remains approximately
doubly-stochastic, especially in mixed-precision (bfloat16) training?

## Setup

- **Setting A**: 48-layer, n_embd=150, hc_num_streams=4 (~20M params)
- **Setting B**: 6-layer, n_embd=288, hc_num_streams=8 (~20M params)
- **Training**: 5000 iterations, bfloat16, diagnostics logged every 10 steps
- **DS Error**: max(max_row_deviation, max_col_deviation) from doubly-stochastic constraint
- **Orth Residual**: ||O O^T - I||_F for Newton-Schulz orthogonal matrix O
- **Seeds**: Setting A: 5 seeds per method; Setting B: 3 seeds (Sinkhorn), 5 seeds (Ortho)

## Key Results

### Setting A Sinkhorn
- DS Error (final): 0.000000 +/- 0.000000
- DS Error (max over training): 0.000000
- DS Error always < 1e-2: True
- DS Error always < 1e-3: True

### Setting A Ortho
- DS Error (final): 0.006074 +/- 0.000657
- DS Error (max over training): 0.008471
- DS Error always < 1e-2: True
- DS Error always < 1e-3: False
- Orth Residual (final): 0.007785 +/- 0.000459
- Orth Residual (max over training): 0.010639

### Setting B Sinkhorn
- DS Error (final): 0.000000 +/- 0.000000
- DS Error (max over training): 0.000000
- DS Error always < 1e-2: True
- DS Error always < 1e-3: True

### Setting B Ortho
- DS Error (final): 0.005289 +/- 0.000795
- DS Error (max over training): 0.007525
- DS Error always < 1e-2: True
- DS Error always < 1e-3: False
- Orth Residual (final): 0.007793 +/- 0.000751
- Orth Residual (max over training): 0.011161

## Key Observations

1. **Sinkhorn DS error is exactly 0**: The Sinkhorn-Knopp projection produces exactly
   doubly-stochastic matrices by construction, so DS error is 0 throughout training
   (Setting A: 0.0; Setting B: ~6e-8 from floating point).

2. **Orthostochastic DS error remains bounded below Refute threshold**: The Newton-Schulz
   based orthostochastic projection produces small but nonzero DS error. Across both settings,
   the mean DS error stays below 0.01 (the Refute threshold) at all steps:
   - Setting A: max mean DS error = 0.0085, final = 0.0061 +/- 0.0007
   - Setting B: max mean DS error = 0.0075, final = 0.0053 +/- 0.0008
   However, DS error exceeds the Proceed threshold (1e-3) in both settings.

3. **Orthogonality residual is small but nonzero**: The finite-step Newton-Schulz iteration
   (15 steps for Setting A, 20 for Setting B) produces near-orthogonal matrices:
   - Setting A: ||O O^T - I||_F final = 0.0078 +/- 0.0005, max = 0.0106
   - Setting B: ||O O^T - I||_F final = 0.0078 +/- 0.0008, max = 0.0112
   These residuals are ~1% of ||I||_F, confirming the Newton-Schulz approximation is adequate.

4. **Stability over training**: Both DS error and orthogonality residual remain bounded
   and do not diverge over the course of 5000 training iterations, even with bfloat16
   mixed-precision training. No monotonic growth is observed, indicating the constraint
   violation does not accumulate.

5. **Per-layer uniformity**: The per-layer bar charts show that DS error and orthogonality
   residual are relatively uniform across layers, with no systematic trend of early vs late
   layers exhibiting worse constraint fidelity.

## Figures

- `results/ds_error_trajectories.pdf` - DS error over training for both methods and settings
- `results/orthogonality_residual_trajectories.pdf` - Orth residual over training for ortho method
- `results/per_layer_diagnostics.pdf` - Per-layer diagnostics at final iteration
