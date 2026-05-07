# IFS-ENS 50-Member Ensemble Reference Evaluation

## Experiment Overview

Evaluate the ECMWF IFS-ENS operational 50-member ensemble on WeatherBench 2 Z500 at 5-day lead time as a context reference. IFS-ENS provides an upper-bound calibration reference for what a well-tuned operational ensemble achieves. This is reported for context only and is not part of the decision rule comparing GraphCast-based perturbation methods.

## Setup

- **Data source**: `gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`
- **Variable**: Geopotential at 500 hPa (Z500)
- **Lead time**: 5 days
- **Ensemble size**: 50 members (operational IFS-ENS)
- **Evaluation period**: 2020 even months (Feb, Apr, Jun, Aug, Oct, Dec)
- **Initialization times**: 00z and 12z (differs from GraphCast's 06z/18z)
- **Truth**: ERA5 reanalysis at valid times
- **Bias correction**: None (evaluated as-is)
- **Aligned evaluation pairs**: 354 (of 364 even-month init times; 10 dropped due to missing ERA5 valid times)
- **Grid**: 0.25 deg equiangular (721 x 1440)

## Key Results

| Metric | Global | Extra-tropics 30-60 deg |
|--------|--------|------------------------|
| CRPS | 117.34 | 174.75 |
| Spread-Skill Ratio | 0.989 | 1.010 |

## Key Observations

1. **IFS-ENS achieves substantially lower CRPS** than all GraphCast-based perturbation methods (deterministic: 160.93, GP-1200km: 143.47, SED: 143.80), as expected for a well-tuned operational NWP ensemble.
2. **Near-perfect calibration**: SSR of 0.989 globally and 1.010 in extra-tropics indicates the IFS-ENS ensemble spread closely matches its forecast error — a hallmark of a mature operational system with perturbed initial conditions and stochastic physics.
3. **Comparison caveat**: IFS-ENS uses 00z/12z initialization times while GraphCast uses 06z/18z. The initialization frequency, model formulation, and data assimilation system are fundamentally different. This row serves only as a calibration reference.
4. **Evaluation runtime**: ~70 minutes streaming from GCS (354 time steps, 50 members each, processed in chunks of 2 to manage memory).
