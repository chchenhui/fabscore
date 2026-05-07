# Anisotropic Spectral Error Dressing (ASED) — Main Experiment

## Experiment Overview

Evaluated the proposed ASED method on WeatherBench 2 Z500 at 5-day lead time. ASED extends degree-only SED by adding within-degree anisotropy profiles that redistribute variance across spherical harmonic orders m while preserving the degree spectrum C_l. The optimized version uses 3 mu-bins (zonal, intermediate, meridional) and 4 log-spaced degree bands, allowing scale-dependent anisotropy capture.

## Setup

- **Variable**: Geopotential at 500 hPa (Z500)
- **Lead time**: 5 days
- **Forecast model**: GraphCast (2020), bias-corrected
- **Verification**: ERA5
- **Grid**: 0.25 deg (721x1440)
- **Calibration**: 2020 odd months (Jan, Mar, May, Jul, Sep, Nov)
- **Evaluation**: 2020 even months (Feb, Apr, Jun, Aug, Oct, Dec)
- **Ensemble size**: M=50
- **Seeds**: 5 (0-4), report mean +/- std
- **ASED parameters**: lmax=359, l_min=10, n_bins=10, n_mu_bins=3 (uniform: [0,0.33], [0.33,0.67], [0.67,1.0]), n_aniso_bands=4 (log-spaced)

## Key Results

### Anisotropy Profile (3 bins: zonal / intermediate / meridional)

| Band | l range | w_zonal | w_intermediate | w_meridional | Ratio z/m |
|------|---------|---------|----------------|--------------|-----------|
| Planetary | 10-23 | 130.3 | 127.6 | 31.6 | 4.12 |
| Synoptic | 24-59 | 5.70 | 4.31 | 0.95 | 5.99 |
| Mesoscale | 60-146 | 0.054 | 0.046 | 0.013 | 4.03 |
| Small-scale | 147-359 | 6.6e-4 | 5.6e-4 | 3.6e-4 | 1.82 |

The synoptic band (l=24-59) shows the strongest anisotropy ratio (5.99:1), confirming mid-latitude storm-track structures dominate directional error patterns at these scales.

### Evaluation Metrics (mean +/- std across 5 seeds)

| Metric | ASED | SED | GP-1200km | IFS-ENS |
|--------|------|-----|-----------|---------|
| CRPS Global | **139.60 +/- 0.52** | 143.80 +/- 0.52 | 143.47 +/- 0.74 | 117.34 |
| CRPS Extra-tropics 30-60 | **181.54 +/- 0.46** | 182.90 +/- 0.51 | 182.94 +/- 0.90 | 174.75 |
| SSR Global | 1.123 +/- 0.001 | 1.129 +/- 0.000 | 1.127 +/- 0.002 | 0.989 |
| SSR Extra-tropics 30-60 | 1.010 +/- 0.003 | 1.009 +/- 0.002 | 1.009 +/- 0.006 | 1.010 |

### Improvement over SED

- CRPS Global: -4.20 points (2.9% improvement)
- CRPS Extra-tropics: -1.36 points (0.7% improvement)

## Key Observations

1. ASED consistently improves CRPS over SED across all 5 seeds, both globally and in extra-tropics.
2. The 3-bin mu decomposition reveals that zonal and intermediate modes have similar power (~5.8 vs ~5.6), while meridional modes carry much less (~1.4). The 2-bin split at mu=0.5 was suboptimal because it mixed intermediate modes with both groups.
3. Degree-band-dependent weights capture that synoptic-scale anisotropy (ratio 5.99) is much stronger than planetary (4.12) or small-scale (1.82), enabling targeted variance redistribution at storm-track scales.
4. The g_lm multiplier range expanded from [0.65, 1.42] (2-bin, 1-band) to [0.26, 1.59] (3-bin, 4-band), indicating much stronger directional variance redistribution.
5. CRPS global improvement nearly doubled from 1.9% to 2.9% over SED. Extra-tropics improved from 0.5% to 0.7%.
6. All three sanity checks passed: SED degeneracy, per-degree conservation, and total SH variance match.

## Sanity Check Results

| Check | Result | Detail |
|-------|--------|--------|
| SED degeneracy (w_low=w_high => g=1) | PASS | max |g-1| = 0.0 |
| Per-degree conservation (mean g = 1) | PASS | max deviation = 1.51e-14 |
| Total SH variance (ASED = SED) | PASS | rel diff = 2.76e-12 |
