# Isotropic GP-1200km Perturbation Baseline

## Experiment Overview

Evaluates an isotropic Gaussian-process correlated noise perturbation baseline on WeatherBench 2 Z500 at 5-day lead time. This baseline uses a fixed 1200 km horizontal decorrelation length scale (matching GenCast's GraphCast-Perturbed GP perturbations) to generate spatially correlated but isotropic perturbations. Variance is globally matched to the calibration residual variance.

Reference: "GenCast: Diffusion-based ensemble forecasting for medium-range weather" (Price et al., 2023), https://arxiv.org/abs/2312.15796

## Setup

- **Variable**: Geopotential at 500 hPa (Z500)
- **Lead time**: 5 days
- **Base model**: GraphCast (WB2 2020 forecasts)
- **Grid**: 0.25 deg equiangular (721 x 1440)
- **Calibration split**: 2020 odd months (Jan, Mar, May, Jul, Sep, Nov)
- **Evaluation split**: 2020 even months (Feb, Apr, Jun, Aug, Oct, Dec)
- **Ensemble size**: M = 50
- **Random seeds**: [0, 1, 2, 3, 4]
- **Decorrelation length**: L = 1200 km
- **GP power spectrum**: C_l^GP = exp(-l(l+1) * (L/R)^2), R = 6371 km
- **Max SH degree**: lmax = 359
- **Variance matching**: alpha = sqrt(V_res / V_eta), V_res = 109068.28
- **Bias correction**: Spatial bias field estimated on calibration split

## Key Results

| Metric | Mean +/- Std (5 seeds) |
|--------|----------------------|
| CRPS global | 143.47 +/- 0.74 |
| CRPS extratropics 30-60 | 182.94 +/- 0.90 |
| Spread-skill ratio global | 1.127 +/- 0.002 |
| Spread-skill ratio extratropics | 1.009 +/- 0.006 |

Comparison with deterministic baseline:

| Method | CRPS Global | CRPS Extra-tropics 30-60 |
|--------|-------------|--------------------------|
| Deterministic (bias-corrected) | 160.93 | 239.96 |
| Isotropic GP-1200km | 143.47 +/- 0.74 | 182.94 +/- 0.90 |

## Key Observations

1. **CRPS improvement over deterministic**: The GP perturbation ensemble reduces CRPS by ~10.8% globally and ~23.8% in extra-tropics compared to the deterministic baseline. This confirms that even naive isotropic noise substantially improves probabilistic skill.

2. **Spread-skill ratio**: Global SSR is 1.13 (slightly overdispersive), while extra-tropics SSR is close to 1.01 (near-perfect calibration in the 30-60 latitude band). The isotropic GP tends to be slightly overdispersive globally, likely because the same perturbation magnitude is applied everywhere including the tropics where forecast errors are smaller.

3. **Low variance across seeds**: Std across 5 seeds is small (<1% of mean for CRPS), confirming that M=50 members provide stable estimates.

4. **GP spectral properties**: The L=1200km GP spectrum drops below 1% of peak power at l=11, meaning perturbations are very smooth (large-scale only). This is a strong smoothness prior that does not match the actual residual error spectrum.
