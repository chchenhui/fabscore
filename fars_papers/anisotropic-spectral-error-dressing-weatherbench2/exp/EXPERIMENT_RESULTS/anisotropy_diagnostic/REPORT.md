# Calibration Anisotropy Index and Spectral Diagnostics

## Experiment Overview

Computed the pre-registered calibration anisotropy index A_cal and produced spectral diagnostics of GraphCast Z500 residuals at 5-day lead time. This analysis quantifies within-degree anisotropy in the residuals to interpret ASED's effectiveness.

## Setup

- **Variable**: Geopotential at 500 hPa (Z500)
- **Lead time**: 5 days
- **Calibration split**: 2020 odd months (368 time steps)
- **SH resolution**: lmax = 359
- **Anisotropy threshold**: mu = |m|/l, split at 0.5 (low-mu: quasi-zonal, high-mu: quasi-meridional)
- **Analysis degrees**: l >= 10

## Key Results

| Metric | Value |
|--------|-------|
| A_cal | -0.2762 +/- 0.0053 |
| \|A_cal\| < 0.1? | No (significantly anisotropic) |
| Quasi-zonal (A_cal < 0)? | Yes |
| ASED w_low / w_high (2-bin) | 5.785 / 2.780 = 2.08x |

## Key Observations

1. **Strong anisotropy**: A_cal = -0.276, far exceeding the 0.1 isotropic threshold (52x the standard error). This confirms that GraphCast Z500 residuals have significant within-degree anisotropy.

2. **Quasi-zonal structure**: The negative A_cal indicates P_low > P_high, meaning low-order (quasi-zonal, |m|/l < 0.5) SH coefficients carry more power than high-order (quasi-meridional) ones. This is consistent with the expected zonal structure of weather forecast errors.

3. **ASED justification**: The strong anisotropy validates the ASED method's approach of redistributing perturbation variance across SH orders. The 2-bin ASED weights (w_low/w_high = 2.08) correctly amplify low-mu perturbations and attenuate high-mu perturbations.

4. **Scale dependence**: The anisotropy contrast A(l) varies with degree, being strongest at synoptic scales (l ~ 10-40). The optimized 4-band ASED captures this by using degree-band-dependent weights.

5. **Spectral structure**: The empirical degree power spectrum C_l shows a steep power-law decay, with the GP-1200km spectrum providing a smooth approximation that overestimates power at high degrees (small scales).

## Output Files

- `ased/results/anisotropy_diagnostic.json` — Full per-degree results
- `ased/figures/degree_spectrum.pdf` — Degree power spectrum + anisotropy contrast
- `ased/figures/anisotropy_profile.pdf` — P_low/P_high profiles + ratio with ASED weights
