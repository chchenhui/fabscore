# Load GraphCast forecasts, ERA5 truth, and IFS-ENS data from WeatherBench 2 Zarr stores on GCS.
# Provides functions for selecting Z500, lead times, calibration/evaluation splits, bias correction.
# GraphCast dims: (time, prediction_timedelta, level, lat, lon) — renamed to latitude/longitude
# ERA5 dims: (time, level, latitude, longitude) — native WB2 convention
# IFS-ENS dims: (time, number, prediction_timedelta, level, latitude, longitude) — number->realization
# All outputs use (latitude, longitude) to match WB2 metrics library.
# Data is downloaded in time-chunks to avoid OOM / timeout on large requests.

import numpy as np
import xarray as xr
import gcsfs

GCS_GRAPHCAST = "weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
GCS_ERA5 = "weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
GCS_IFS_ENS = "weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"

CACHE_DIR = "/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/anisotropic-spectral-error-dressing-weatherbench2/exp/ased/data/.cache"


def _open_zarr(gcs_path):
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.GCSMap(gcs_path, gcs=fs)
    return xr.open_zarr(store, consolidated=True)


def _chunked_compute(da, dim="time", chunk_size=30):
    import os
    n = da.sizes[dim]
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = da.isel({dim: slice(start, end)}).compute()
        chunks.append(chunk)
        print(f"    Downloaded {end}/{n} time steps")
    return xr.concat(chunks, dim=dim)


def load_graphcast_z500(year=2020, lead_time_days=5):
    import os
    cache_path = os.path.join(CACHE_DIR, f"graphcast_z500_{year}_lead{lead_time_days}d.nc")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        ds = xr.open_dataset(cache_path)
        return ds

    ds = _open_zarr(GCS_GRAPHCAST)
    td = np.timedelta64(lead_time_days, "D")
    da = ds["geopotential"].sel(level=500, prediction_timedelta=td)
    times = da.time.values
    year_mask = (times >= np.datetime64(f"{year}-01-01")) & (
        times < np.datetime64(f"{year + 1}-01-01")
    )
    da = da.isel(time=year_mask)
    da = da.rename({"lat": "latitude", "lon": "longitude"})
    da = da.sortby("latitude")
    print(f"  Downloading {da.sizes['time']} time steps from GCS...")
    da = _chunked_compute(da, dim="time", chunk_size=30)
    result = da.to_dataset(name="geopotential")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_netcdf(cache_path, engine="h5netcdf")
    print(f"  Cached to {cache_path}")
    return result


def load_era5_z500(year=2020):
    import os
    cache_path = os.path.join(CACHE_DIR, f"era5_z500_{year}.nc")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        ds = xr.open_dataset(cache_path)
        return ds

    ds = _open_zarr(GCS_ERA5)
    da = ds["geopotential"].sel(level=500)
    times = da.time.values
    year_mask = (times >= np.datetime64(f"{year}-01-01")) & (
        times < np.datetime64(f"{year + 1}-01-01")
    )
    da = da.isel(time=year_mask)
    da = da.sortby("latitude")
    print(f"  Downloading {da.sizes['time']} time steps from GCS...")
    da = _chunked_compute(da, dim="time", chunk_size=60)
    result = da.to_dataset(name="geopotential")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_netcdf(cache_path, engine="h5netcdf")
    print(f"  Cached to {cache_path}")
    return result


def load_ifs_ens_z500(year=2020, lead_time_days=5, months=None):
    import os
    suffix = "" if months is None else f"_months{''.join(str(m) for m in months)}"
    cache_path = os.path.join(CACHE_DIR, f"ifs_ens_z500_{year}_lead{lead_time_days}d{suffix}.nc")
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        ds = xr.open_dataset(cache_path)
        return ds

    ds = _open_zarr(GCS_IFS_ENS)
    td = np.timedelta64(lead_time_days, "D")
    da = ds["geopotential"].sel(level=500, prediction_timedelta=td)
    times = da.time.values
    import pandas as pd
    year_mask = (times >= np.datetime64(f"{year}-01-01")) & (
        times < np.datetime64(f"{year + 1}-01-01")
    )
    if months is not None:
        month_vals = pd.DatetimeIndex(times).month
        month_mask = np.isin(month_vals, months)
        year_mask = year_mask & month_mask
    da = da.isel(time=year_mask)
    da = da.rename({"number": "realization"})
    da = da.sortby("latitude")
    n_time = da.sizes["time"]
    n_members = da.sizes["realization"]
    print(f"  Downloading {n_time} time steps x {n_members} members from GCS...")
    da = _chunked_compute_with_retry(da, dim="time", chunk_size=5)
    result = da.to_dataset(name="geopotential")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_netcdf(cache_path, engine="h5netcdf")
    print(f"  Cached to {cache_path}")
    return result


def _chunked_compute_with_retry(da, dim="time", chunk_size=5, max_retries=3):
    import time as _time
    n = da.sizes[dim]
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for attempt in range(max_retries):
            try:
                chunk = da.isel({dim: slice(start, end)}).compute()
                chunks.append(chunk)
                print(f"    Downloaded {end}/{n} time steps")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"    Retry {attempt+1}/{max_retries} for steps {start}-{end}: {e}")
                    _time.sleep(wait)
                else:
                    raise
    return xr.concat(chunks, dim=dim)


def split_calibration_evaluation(
    ds,
    calib_months=(1, 3, 5, 7, 9, 11),
    eval_months=(2, 4, 6, 8, 10, 12),
):
    months = ds.time.dt.month
    calib = ds.sel(time=months.isin(calib_months))
    evalu = ds.sel(time=months.isin(eval_months))
    return calib, evalu


def align_forecast_truth(forecast, truth, lead_time_days=5):
    td = np.timedelta64(lead_time_days, "D")
    valid_times = forecast.time.values + td
    truth_times = truth.time.values
    common = np.intersect1d(valid_times, truth_times)
    fc_mask = np.isin(valid_times, common)
    tr_mask = np.isin(truth_times, common)
    fc_aligned = forecast.isel(time=fc_mask)
    tr_aligned = truth.isel(time=tr_mask)
    tr_aligned = tr_aligned.assign_coords(time=fc_aligned.time.values)
    return fc_aligned, tr_aligned


def compute_bias_correction(forecast_calib, truth_calib):
    residual = forecast_calib["geopotential"] - truth_calib["geopotential"]
    bias = residual.mean("time")
    return bias


def apply_bias_correction(forecast, bias):
    ds = forecast.copy()
    ds["geopotential"] = ds["geopotential"] - bias
    return ds
