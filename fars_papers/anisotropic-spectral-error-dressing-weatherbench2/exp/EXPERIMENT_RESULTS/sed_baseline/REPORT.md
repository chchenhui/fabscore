# SED Baseline Experiment Report

## Experiment Overview

Evaluate the degree-only Spectral Error Dressing (SED) baseline on WB2 Z500 at 5-day lead time. SED matches the empirical residual degree power spectrum C_l from historical forecast errors and samples isotropic (within-degree uniform) Gaussian perturbations in spherical harmonic space. This is the direct comparator for ASED: any CRPS improvement from ASED over SED is attributable to within-degree anisotropy matching.

## Setup

- **Variable**: Geopotential at 500 hPa (Z500)
- **Lead time**: 5 days
- **Forecast model**: GraphCast (2020)
- **Verification**: ERA5 (2020)
- **Grid**: 0.25 deg equiangular (721 x 1440)
- **Calibration split**: 2020 odd months (Jan, Mar, May, Jul, Sep, Nov) — 368 time steps
- **Evaluation split**: 2020 even months (Feb, Apr, Jun, Aug, Oct, Dec) — 354 time steps
- **Bias correction**: Spatial bias field estimated on calibration split, subtracted from forecasts
- **SH transform**: pyshtools, lmax=359, DH2 sampling
- **C_l estimation**: Degree power spectrum averaged over calibration residuals, smoothed with 10 log-spaced l-bins
- **Perturbation sampling**: i.i.d. N(0,1) in real SH basis, scaled by sqrt(C_l) — same variance for all m within each l
- **Variance matching**: Global alpha = sqrt(V_res / V_eta) where V_res = mean(r_bc^2) = 109068.28
- **Ensemble size**: M=50
- **Seeds**: 5 random seeds [0, 1, 2, 3, 4]

## Key Results

| Metric | Mean | Std |
|--------|------|-----|
| CRPS Global (m^2/s^2) | 143.80 | 0.52 |
| CRPS Extra-tropics 30-60 (m^2/s^2) | 182.90 | 0.51 |
| Spread-Skill Ratio Global | 1.129 | 0.001 |
| Spread-Skill Ratio Extra-tropics 30-60 | 1.009 | 0.002 |

## Key Observations

- SED CRPS is comparable to GP-1200km (143.80 vs 143.47 global, 182.90 vs 182.94 extra-tropics), within noise margins.
- SED has lower seed-to-seed variance (std 0.52 vs 0.74 global CRPS), likely because the empirical C_l spectrum is a better fit to the residual structure than the parametric GP kernel.
- Spread-skill ratio ~1.13 globally and ~1.01 in extra-tropics, indicating slight over-dispersion globally but near-perfect calibration in extra-tropics — identical pattern to GP-1200km.
- The empirical C_l is dominated by low degrees (l<20), with power dropping rapidly at higher degrees, consistent with large-scale forecast error structure.
- Variance matching alpha ~1.18, indicating the raw SH perturbations slightly underestimate the residual variance (corrected by global scaling).
