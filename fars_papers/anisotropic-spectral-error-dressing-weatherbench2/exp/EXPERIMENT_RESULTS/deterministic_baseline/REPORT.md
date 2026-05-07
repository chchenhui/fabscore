# Deterministic Bias-Corrected GraphCast Baseline

## Experiment Overview

Evaluate the deterministic bias-corrected GraphCast forecast on WeatherBench 2 for Z500 at 5-day lead time. For a single deterministic forecast (M=1), CRPS reduces to latitude-weighted MAE. The bias correction is estimated on the calibration split (2020 odd months) and applied to the evaluation split (2020 even months).

## Setup

- **Model**: GraphCast (deterministic, no ensemble)
- **Variable**: Geopotential at 500 hPa (Z500), units: m^2/s^2
- **Lead time**: 5 days
- **Grid**: 0.25 deg equiangular (721 x 1440)
- **Calibration split**: 2020 odd months (Jan, Mar, May, Jul, Sep, Nov) -- 368 init times
- **Evaluation split**: 2020 even months (Feb, Apr, Jun, Aug, Oct, Dec) -- 354 init times
- **Bias correction**: Spatial bias field b(x) = mean_t[f(t,x) - o(t+l,x)] estimated on calibration split
- **Data sources**:
  - GraphCast: `gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr`
  - ERA5: `gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`

## Key Results

| Metric | Value |
|--------|-------|
| CRPS global (bias-corrected) | 160.93 m^2/s^2 |
| CRPS extra-tropics 30-60 deg (bias-corrected) | 239.96 m^2/s^2 |
| CRPS global (raw, no bias correction) | 161.86 m^2/s^2 |
| CRPS extra-tropics 30-60 deg (raw) | 240.31 m^2/s^2 |

Bias field statistics: mean = -25.33, std = 39.92, min = -192.90, max = 120.40

## Key Observations

1. For M=1, CRPS equals latitude-weighted MAE since the spread term is zero.
2. Bias correction provides a small improvement (~0.6% globally, ~0.15% extra-tropics).
3. Extra-tropical errors are ~49% higher than global average, reflecting stronger weather variability at mid-latitudes.
4. The bias field has non-trivial spatial structure (std ~40 m^2/s^2) with a negative mean, indicating GraphCast slightly overestimates Z500 on average.
5. These values serve as the lower-bound reference for all subsequent ensemble methods (SED, ASED, GP noise).
