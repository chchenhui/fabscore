#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pandas as pd


REQUIRED_COLS = {"case", "key", "value", "unit", "source_bibkey"}


def load_clusters_from_xlsx(path: str) -> pd.DataFrame:
    """
    Load clustered variable sheets from an Excel workbook where
    each sheet corresponds to a single variable (key).
    Sheets named 'VARIABILITY' or 'README' are ignored.
    Expected columns per sheet: case, key, value, unit, source_bibkey
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        if sheet.upper() in {"VARIABILITY", "README"}:
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        if not REQUIRED_COLS.issubset(df.columns):
            # Skip sheets that do not present the required columns
            continue
        frames.append(df[list(REQUIRED_COLS)])

    if not frames:
        raise RuntimeError("No valid variable sheets found in the Excel file.")

    return pd.concat(frames, axis=0, ignore_index=True)


def load_clusters_from_dir(path: str) -> pd.DataFrame:
    """
    Load clustered variable CSVs from a directory where each CSV
    corresponds to a single variable (key).
    Expected columns per CSV: case, key, value, unit, source_bibkey
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Directory not found: {path}")

    frames = []
    for fn in os.listdir(path):
        if not fn.lower().endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(path, fn))
        if not REQUIRED_COLS.issubset(df.columns):
            continue
        frames.append(df[list(REQUIRED_COLS)])

    if not frames:
        raise RuntimeError("No valid CSVs with required columns were found in the directory.")

    return pd.concat(frames, axis=0, ignore_index=True)


def aggregate_parameters(df: pd.DataFrame, agg: str = "median", category_source: str | None = None) -> pd.DataFrame:
    """
    Aggregate clustered values to characteristic parameters per (case, key).
    Aggregation uses median by default (or mean if requested).
    The 'category' field is filled from an optional category_source CSV if available.
    Output columns: case, key, value, unit, category, source_bibkey
    """
    if agg not in {"median", "mean"}:
        raise ValueError("aggregate must be 'median' or 'mean'")

    # Ensure value column is numeric (coerce non-numeric to NaN for safe aggregation)
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out_rows = []
    for (case, key), g in df.groupby(["case", "key"], dropna=False):
        # Drop NaNs before aggregation
        g_vals = g["value"].dropna()
        if g_vals.empty:
            val = float("nan")
        else:
            if agg == "mean":
                val = float(g_vals.mean())
            else:
                val = float(g_vals.median())

        # Use the most frequent unit and source_bibkey for stability
        unit = str(g["unit"].mode().iloc[0]) if not g["unit"].empty else ""
        src = str(g["source_bibkey"].mode().iloc[0]) if not g["source_bibkey"].empty else ""

        out_rows.append(
            {
                "case": case,
                "key": key,
                "value": val,
                "unit": unit,
                "category": "",          # filled later if mapping exists
                "source_bibkey": src,
            }
        )

    out = pd.DataFrame(out_rows, columns=["case", "key", "value", "unit", "category", "source_bibkey"])

    # Optional category backfill from an existing CSV (e.g., a prior data_sources.csv)
    cat_maps = {}
    candidate_paths = [category_source] if category_source else []
    # As a convenience, also check a local data_sources.csv if present and no explicit source was given
    if not candidate_paths:
        default_local = "data_sources.csv"
        if os.path.exists(default_local):
            candidate_paths = [default_local]

    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                old = pd.read_csv(p)
                if {"case", "key", "category"}.issubset(old.columns):
                    # later mappings override earlier ones
                    cat_maps.update({(r["case"], r["key"]): r["category"] for _, r in old.iterrows()})
            except Exception:
                # If anything goes wrong, just skip category backfill
                pass

    if cat_maps:
        out["category"] = [cat_maps.get((r["case"], r["key"]), "") for _, r in out.iterrows()]

    return out



def main():
    ap = argparse.ArgumentParser(description="Aggregate clustered variable data into model-ready parameters.")
    ap.add_argument("--source_xlsx", type=str, default="data_sources_clustered.xlsx",
                    help="Excel workbook where each sheet is one variable (default: data_sources_clustered.xlsx).")
    ap.add_argument("--source_dir", type=str, default=None,
                    help="Directory with per-variable CSVs (if provided, overrides --source_xlsx).")
    ap.add_argument("--aggregate", choices=["median", "mean"], default="median",
                    help="Aggregation function for clustered values (default: median).")
    ap.add_argument("--out_csv", type=str, default="data_sources.csv",
                    help="Output CSV file (default: data_sources.csv).")
    ap.add_argument("--category_source", type=str, default=None,
                    help="Optional CSV to backfill 'category' per (case,key).")
    args = ap.parse_args()

    if args.source_dir:
        df = load_clusters_from_dir(args.source_dir)
    else:
        df = load_clusters_from_xlsx(args.source_xlsx)

    out = aggregate_parameters(df, agg=args.aggregate, category_source=args.category_source)

    # Ensure output directory exists if a path is provided
    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Sort for readability and write
    out = out.sort_values(["case", "key"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} aggregated parameters to {args.out_csv} using {args.aggregate}.")

if __name__ == "__main__":
    main()