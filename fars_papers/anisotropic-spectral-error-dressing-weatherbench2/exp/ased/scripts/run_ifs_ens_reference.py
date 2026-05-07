# Evaluate IFS-ENS 50-member operational ensemble on WB2 Z500 @ 5-day.
# Streams data from GCS in small chunks to avoid OOM (50 members x 721x1440 is ~200MB/step).
# IFS-ENS is a context reference (upper bound), not a direct competitor.
# IFS-ENS init times are 00z/12z (vs GraphCast 06z/18z).
# No bias correction, no seed variance. Single deterministic result.

import gc
import json
import os
import sys
import time as _time

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ased.data.wb2_loader import load_era5_z500
from ased.evaluation.metrics import extratropics_30_60_region
from weatherbench2 import metrics as wb2_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GCS_IFS_ENS = "weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"
LEAD_TIME_DAYS = 5
STREAM_CHUNK = 2


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Opening IFS-ENS Zarr store (lazy)...")
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(GCS_IFS_ENS, gcs=fs)
    ds = xr.open_zarr(store, consolidated=True)

    td = np.timedelta64(LEAD_TIME_DAYS, "D")
    da = ds["geopotential"].sel(level=500, prediction_timedelta=td)

    all_times = da.time.values
    year_mask = (all_times >= np.datetime64("2020-01-01")) & (
        all_times < np.datetime64("2021-01-01")
    )
    month_vals = pd.DatetimeIndex(all_times).month
    even_mask = np.isin(month_vals, [2, 4, 6, 8, 10, 12])
    sel_mask = year_mask & even_mask
    sel_indices = np.where(sel_mask)[0]
    init_times = all_times[sel_mask]
    n_eval_total = len(init_times)
    print(f"  IFS-ENS 2020 even-month init times: {n_eval_total}")

    print("Loading ERA5 Z500 truth (2020, cached)...")
    t0 = _time.time()
    era5 = load_era5_z500(year=2020)
    era5_times = era5.time.values
    print(f"  Loaded in {_time.time() - t0:.1f}s")

    valid_times = init_times + td
    common = np.intersect1d(valid_times, era5_times)
    fc_keep = np.isin(valid_times, common)
    sel_indices = sel_indices[fc_keep]
    init_times = init_times[fc_keep]
    valid_times_aligned = init_times + td
    n_eval = len(init_times)
    print(f"  After alignment with ERA5: {n_eval} evaluation pairs")

    era5_truth_mask = np.isin(era5_times, valid_times_aligned)
    era5_truth = era5.isel(time=era5_truth_mask)
    era5_truth = era5_truth.assign_coords(time=init_times)
    del era5
    gc.collect()

    region_extratropics = extratropics_30_60_region()
    skill_metric = wb2_metrics.CRPSSkill(ensemble_dim="realization")
    spread_metric = wb2_metrics.CRPSSpread(ensemble_dim="realization")
    regions = {"global": None, "extratropics": region_extratropics}
    accum = {f"{m}_{r}": 0.0 for m in ["skill", "spread"] for r in regions}
    total_time = 0
    n_chunks = (n_eval + STREAM_CHUNK - 1) // STREAM_CHUNK

    print(f"\nStreaming evaluation ({n_chunks} chunks of {STREAM_CHUNK})...")
    t_start = _time.time()

    for ci in range(n_chunks):
        start = ci * STREAM_CHUNK
        end = min(start + STREAM_CHUNK, n_eval)
        cs = end - start

        chunk_indices = sel_indices[start:end]
        for attempt in range(3):
            try:
                ens_da = da.isel(time=chunk_indices).compute()
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Retry {attempt+1} for chunk {ci+1}: {e}")
                    _time.sleep(10 * (attempt + 1))
                else:
                    raise

        ens_da = ens_da.rename({"number": "realization"})
        ens_da = ens_da.sortby("latitude")
        chunk_times = init_times[start:end]
        ens_ds = xr.Dataset({
            "geopotential": xr.DataArray(
                ens_da.values,
                dims=["time", "realization", "latitude", "longitude"],
                coords={
                    "time": chunk_times,
                    "realization": np.arange(ens_da.sizes["realization"]),
                    "latitude": ens_da.latitude.values,
                    "longitude": ens_da.longitude.values,
                },
            )
        })

        truth_chunk = era5_truth.isel(time=slice(start, end))

        for rname, region in regions.items():
            skill_res = skill_metric.compute_chunk(ens_ds, truth_chunk, region=region)
            spread_res = spread_metric.compute_chunk(ens_ds, truth_chunk, region=region)
            accum[f"skill_{rname}"] += float(skill_res["geopotential"].mean("time").values) * cs
            accum[f"spread_{rname}"] += float(spread_res["geopotential"].mean("time").values) * cs
            del skill_res, spread_res

        total_time += cs
        del ens_da, ens_ds, truth_chunk
        gc.collect()

        if (ci + 1) % 20 == 0 or ci + 1 == n_chunks:
            elapsed = _time.time() - t_start
            rate = total_time / elapsed * 3600
            print(f"    Chunk {ci+1}/{n_chunks} done ({total_time}/{n_eval} steps, {elapsed:.0f}s, ~{rate:.0f} steps/hr)")

    for k in accum:
        accum[k] /= total_time

    crps_global = accum["skill_global"] - 0.5 * accum["spread_global"]
    crps_extra = accum["skill_extratropics"] - 0.5 * accum["spread_extratropics"]
    ssr_global = accum["spread_global"] / accum["skill_global"] if accum["skill_global"] != 0 else float("nan")
    ssr_extra = accum["spread_extratropics"] / accum["skill_extratropics"] if accum["skill_extratropics"] != 0 else float("nan")

    results = {
        "crps_global": crps_global,
        "crps_extratropics_30_60": crps_extra,
        "spread_skill_ratio_global": ssr_global,
        "spread_skill_ratio_extratropics_30_60": ssr_extra,
    }

    total_elapsed = _time.time() - t_start
    print(f"\n  Total evaluation time: {total_elapsed:.1f}s")
    print(f"\n=== IFS-ENS Results ===")
    print(f"  CRPS global:       {results['crps_global']:.4f}")
    print(f"  CRPS extratropics: {results['crps_extratropics_30_60']:.4f}")
    print(f"  SSR global:        {results['spread_skill_ratio_global']:.4f}")
    print(f"  SSR extratropics:  {results['spread_skill_ratio_extratropics_30_60']:.4f}")

    output = {
        "method": "ifs_ens_operational_ensemble",
        "variable": "geopotential_500hPa",
        "lead_time_days": LEAD_TIME_DAYS,
        "evaluation_months": [2, 4, 6, 8, 10, 12],
        "grid": "0.25deg_721x1440",
        "bias_corrected": False,
        "ensemble_size": 50,
        "init_times": "00z_12z",
        "n_eval_times": n_eval,
        "note": "Context reference only. IFS-ENS uses 00z/12z init times vs GraphCast 06z/18z. Direct comparison should be interpreted cautiously.",
        "metrics": results,
    }

    out_path = os.path.join(RESULTS_DIR, "ifs_ens_reference.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
