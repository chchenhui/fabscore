# ASED Optimization — Iteration 0

## Experiment Overview

Optimized the Anisotropic Spectral Error Dressing (ASED) method by extending the within-degree anisotropy model from 2 mu-bins to 3 mu-bins and adding degree-band-dependent weights (4 bands). The goal was to improve CRPS, particularly in the extra-tropics 30-60 region where the original ASED only achieved 0.5% improvement over SED (below the 1% target).

## Setup

- **Variable**: Z500, Lead time: 5 days, Grid: 0.25 deg
- **Forecast**: GraphCast 2020, bias-corrected
- **Calibration**: 2020 odd months, Evaluation: 2020 even months
- **Ensemble**: M=50, 5 seeds (0-4)
- **Key changes**:
  - 3 uniform mu-bins: [0, 0.33], [0.33, 0.67], [0.67, 1.0] (vs original 2 bins at mu=0.5)
  - 4 log-spaced degree bands: l=[10,23], [24,59], [60,146], [147,359] (vs original single global band)
  - n_spectrum_bins=10, l_min=10 (unchanged)

## Key Results

### Comparison Table (mean +/- std across 5 seeds)

| Metric | Optimized ASED | Original ASED | SED | GP-1200km |
|--------|---------------|---------------|-----|-----------|
| CRPS Global | **139.60 +/- 0.52** | 141.03 +/- 0.54 | 143.80 +/- 0.52 | 143.47 +/- 0.74 |
| CRPS Extra-tropics 30-60 | **181.54 +/- 0.46** | 182.03 +/- 0.52 | 182.90 +/- 0.51 | 182.94 +/- 0.90 |
| SSR Global | 1.123 +/- 0.001 | 1.123 +/- 0.001 | 1.129 +/- 0.001 | 1.127 +/- 0.002 |
| SSR Extra-tropics 30-60 | 1.010 +/- 0.003 | 0.997 +/- 0.003 | 1.009 +/- 0.002 | 1.009 +/- 0.006 |

### Improvement over SED

| Region | Optimized ASED | Original ASED | Target |
|--------|---------------|---------------|--------|
| CRPS Global | **-2.92%** | -1.9% | >= 1% |
| CRPS Extra-tropics | **-0.75%** | -0.5% | >= 1% |

### Improvement over Original ASED

- CRPS Global: -1.43 points (-1.0%)
- CRPS Extra-tropics: -0.49 points (-0.27%)

### Anisotropy Band Weights (3 bins: zonal / intermediate / meridional)

| Band | l range | w_zonal | w_intermediate | w_meridional | Ratio z/m |
|------|---------|---------|----------------|--------------|-----------|
| Planetary | 10-23 | 130.3 | 127.6 | 31.6 | 4.12 |
| Synoptic | 24-59 | 5.70 | 4.31 | 0.95 | 5.99 |
| Mesoscale | 60-146 | 0.054 | 0.046 | 0.013 | 4.03 |
| Small-scale | 147-359 | 6.6e-4 | 5.6e-4 | 3.6e-4 | 1.82 |

## Key Observations

1. The 3-bin approach reveals that the zonal and intermediate bins have similar power (~5.8 vs ~5.6 globally), while the meridional bin has much less (~1.4). The original 2-bin split at mu=0.5 was suboptimal because it lumped intermediate modes with either zonal or meridional modes.

2. The synoptic band (l=24-59) shows the strongest anisotropy ratio (5.99:1), confirming that mid-latitude storm track structures dominate the directional error pattern at these scales.

3. The g_lm multiplier range expanded from [0.65, 1.42] (original) to [0.26, 1.59] (optimized), indicating much stronger variance redistribution.

4. CRPS global improvement increased from 1.9% to 2.9% over SED. Extra-tropics improved from 0.5% to 0.75%, but still below the 1% target.

5. The extra-tropics SSR moved from 0.997 (near-perfect) to 1.010 (slightly over-dispersed), suggesting the stronger anisotropy redistribution creates slightly too much spread in extra-tropics for some seeds.

6. The fundamental limitation is that SH perturbations are global — they cannot selectively target extra-tropics. The extra-tropics CRPS is inherently harder to improve than global CRPS because the perturbation structure must match finer-scale storm-track error patterns that SH-based methods can only approximate.
