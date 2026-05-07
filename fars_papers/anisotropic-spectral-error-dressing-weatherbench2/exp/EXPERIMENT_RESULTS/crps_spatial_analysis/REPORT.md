# Spatial CRPS Difference Analysis: ASED vs SED

## Experiment Overview

This analysis computes gridpoint-level CRPS for SED (degree-only spectral error dressing) and ASED (optimized 3-bin 4-band anisotropic SED) to test the physical hypothesis that ASED gains concentrate in the extra-tropics (storm-track regions 30-60 degrees) where forecast errors are directionally organized.

## Setup

- Variable: Z500 (geopotential at 500 hPa)
- Lead time: 5 days
- Ensemble size: M=50, seed=0
- Grid: 0.25 deg equiangular (721 x 1440)
- Evaluation split: 2020 even months (354 time steps)
- CRPS definition: CRPS(lat,lon) = mean_t[E|X-y| - 0.5*E|X-X'|] (WB2 convention)
- ASED configuration: 3 mu-bins, 4 log-spaced degree bands, l_min=10

## Key Results

### Global Statistics
| Metric | Value |
|--------|-------|
| Global mean CRPS_SED (unweighted) | 176.56 m^2/s^2 |
| Global mean CRPS_ASED (unweighted) | 173.39 m^2/s^2 |
| Global mean ΔCRPS (SED - ASED) | 3.17 m^2/s^2 |
| Gridpoints where ASED better | 82.4% |

### Zonal Band ΔCRPS (SED - ASED, positive = ASED better)
| Latitude Band | Mean ΔCRPS |
|---------------|------------|
| Polar S (60-90°S) | 1.41 |
| Extra-tropics SH (30-60°S) | 1.53 |
| **Tropics (30°S-30°N)** | **7.05** |
| Extra-tropics NH (30-60°N) | 0.94 |
| Polar N (60-90°N) | 1.11 |

### Zonal-Mean Peak Values
| Latitude | Zonal-mean ΔCRPS |
|----------|-----------------|
| ~15°N | 8.38 (maximum) |
| ~0° (equator) | 8.32 |
| ~15°S | 7.74 |
| ~45°N | -0.30 (slight SED advantage) |

## Key Observations

1. **ASED improves over SED almost everywhere**: 82.4% of gridpoints show positive ΔCRPS (ASED better), with a global mean improvement of 3.17 m^2/s^2.

2. **Largest improvement in the tropics, not extra-tropics**: The tropical band (30°S-30°N) shows mean ΔCRPS of 7.05, roughly 5-7x larger than the extra-tropical bands (0.94-1.53). This is contrary to the initial hypothesis that improvements would concentrate in storm-track regions.

3. **Extra-tropics show modest but positive improvement**: Both NH (0.94) and SH (1.53) extra-tropical bands show positive ΔCRPS, indicating ASED does help in these regions, but the effect is much smaller than in the tropics.

4. **Slight SED advantage around 45°N**: A narrow band around 45°N shows near-zero or slightly negative ΔCRPS, meaning SED slightly outperforms ASED in the core NH storm track.

5. **Hypothesis assessment**: The anisotropy hypothesis — that within-degree directional redistribution would primarily benefit storm-track regions — is not supported by the spatial pattern. Instead, ASED's variance redistribution appears to benefit regions with strong zonal structure (tropics) more than regions with directionally organized but more chaotic error patterns (extra-tropics). The anisotropy profile captures tropical zonal structure (trade winds, ITCZ) at least as effectively as mid-latitude storm-track anisotropy.

## Output Files
- `ased/figures/crps_spatial_diff.pdf` — Two-panel figure (spatial map + zonal-mean)
- `ased/figures/crps_spatial_diff.png` — PNG copy
- `ased/results/crps_spatial_sed.npy` — Gridpoint CRPS for SED (721x1440)
- `ased/results/crps_spatial_ased.npy` — Gridpoint CRPS for ASED (721x1440)
- `ased/results/crps_spatial_diff_summary.json` — Summary statistics
