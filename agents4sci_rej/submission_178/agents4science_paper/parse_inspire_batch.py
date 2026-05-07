#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

# ---------- small utils ----------
def norm(s: str) -> str:
    s = str(s)
    s = s.replace("²", "2")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def find_sheet(xls: pd.ExcelFile, want_names, fallback_patterns):
    names = xls.sheet_names
    for w in want_names:
        if w in names:
            return w
    for n in names:
        if any(re.search(p, norm(n)) for p in fallback_patterns):
            return n
    return None

def header_row_by_required_cols(df_raw: pd.DataFrame, required_keywords, max_search=120):
    req = [norm(k) for k in required_keywords]
    up = min(max_search, len(df_raw))
    for i in range(up):
        row = [norm(v) for v in df_raw.iloc[i].tolist()]
        ok = True
        for kw in req:
            if not any(kw in cell for cell in row):
                ok = False; break
        if ok:
            return i
    return None

def trim_tail(df: pd.DataFrame, key_cols):
    def empty_row(r):
        for c in key_cols:
            v = r.get(c)
            if pd.isna(v): 
                continue
            if isinstance(v, str) and norm(v) == "":
                continue
            return False
        return True
    last_good = -1
    for idx, r in df.iterrows():
        if not empty_row(r):
            last_good = idx
    return df.loc[:last_good].copy() if last_good >= 0 else df

# ---------- single-file parser ----------
def parse_one(xlsx_path: Path, sheet_ref=None, sheet_tgt=None, verbose=False) -> pd.DataFrame:
    """
    Return DataFrame with columns:
    ['source_file','Climate','Type of building','Age of construction',
     'consumption_before','consumption_after','energy_savings_percentage']
    """
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        if verbose: print(f"[ERROR] open {xlsx_path.name}: {e}")
        return None

    # choose sheets
    ref_sheet = sheet_ref or find_sheet(
        xls,
        want_names=["Reference buildings simulations"],
        fallback_patterns=[r"\breference\b", r"\bbaseline\b"]
    )
    tgt_sheet = sheet_tgt or find_sheet(
        xls,
        want_names=["Target buildings simulations"],
        fallback_patterns=[r"\btarget\b", r"\bretrofit\b", r"\bafter\b"]
    )
    if ref_sheet is None or tgt_sheet is None:
        if verbose: print(f"[WARN] sheets not found in {xlsx_path.name}: {xls.sheet_names}")
        return None

    # load raw
    df_ref_raw = pd.read_excel(xls, sheet_name=ref_sheet, header=None)
    df_tgt_raw = pd.read_excel(xls, sheet_name=tgt_sheet, header=None)

    # locate header
    common_keys = ["climate", "type of building", "age of construction"]
    hdr_ref = header_row_by_required_cols(df_ref_raw, common_keys)
    hdr_tgt = header_row_by_required_cols(df_tgt_raw, common_keys)
    if hdr_ref is None or hdr_tgt is None:
        if verbose: print(f"[WARN] header not found in {xlsx_path.name}")
        return None

    # read with header
    df_ref = pd.read_excel(xlsx_path, sheet_name=ref_sheet, header=hdr_ref, engine="openpyxl").dropna(how="all")
    df_tgt = pd.read_excel(xlsx_path, sheet_name=tgt_sheet, header=hdr_tgt, engine="openpyxl").dropna(how="all")

    # trim trailing empties
    df_ref = trim_tail(df_ref, df_ref.columns[:5])
    df_tgt = trim_tail(df_tgt, df_tgt.columns[:5])

    # pick key cols
    def pick(cols, tokens_all):
        for c in cols:
            if all(t in norm(c) for t in tokens_all):
                return c
        return None

    key_climate = pick(df_ref.columns, ["climate"])
    key_btype   = None
    for c in df_ref.columns:
        nc = norm(c)
        if "type" in nc and "building" in nc:
            key_btype = c; break
    key_age     = pick(df_ref.columns, ["age","construction"])

    if not all([key_climate, key_btype, key_age]):
        if verbose: print(f"[WARN] key cols missing in {xlsx_path.name}")
        return None

    # before consumption
    key_cons_b = None
    for c in df_ref.columns:
        nc = norm(c)
        if ("kwh" in nc and "m2" in nc) and (("consumption" in nc) or ("average" in nc) or ("energy use" in nc)):
            key_cons_b = c; break
    if key_cons_b is None:
        if verbose: print(f"[WARN] before consumption not found in {xlsx_path.name}")
        return None

    # after consumption
    key_cons_a = None
    for c in df_tgt.columns:
        nc = norm(c)
        if all(t in nc for t in ["building","average","kwh","m2"]):
            key_cons_a = c; break
    if key_cons_a is None:
        for c in df_tgt.columns:
            nc = norm(c)
            if ("kwh" in nc and "m2" in nc) and ("avg" in nc or "average" in nc):
                key_cons_a = c; break
    if key_cons_a is None:
        if verbose: print(f"[WARN] after consumption not found in {xlsx_path.name}")
        return None

    # select & rename
    ref = df_ref[[key_climate, key_btype, key_age, key_cons_b]].copy()
    ref.columns = ["Climate","Type of building","Age of construction","consumption_before"]
    tgt = df_tgt[[key_climate, key_btype, key_age, key_cons_a]].copy()
    tgt.columns = ["Climate","Type of building","Age of construction","consumption_after"]

    # numeric
    ref["consumption_before"] = pd.to_numeric(ref["consumption_before"], errors="coerce")
    tgt["consumption_after"]  = pd.to_numeric(tgt["consumption_after"],  errors="coerce")

    # merge
    merged = pd.merge(tgt, ref, on=["Climate","Type of building","Age of construction"], how="inner")
    merged = merged.dropna(subset=["consumption_before","consumption_after"])
    merged = merged[merged["consumption_before"] > 0]
    if len(merged) == 0:
        return None

    merged["energy_savings_percentage"] = (merged["consumption_before"] - merged["consumption_after"]) / merged["consumption_before"]
    merged = merged[merged["energy_savings_percentage"].between(0, 1, inclusive="both")]

    merged.insert(0, "source_file", xlsx_path.name)
    return merged.reset_index(drop=True)

# ---------- batch ----------
def main():
    ap = argparse.ArgumentParser(description="Batch-parse iNSPiRe Excel files to before/after + savings%, then merge to train.csv")
    ap.add_argument("--in_dir", required=True, help="Folder containing .xlsx files (searched recursively)")
    ap.add_argument("--out_csv", required=True, help="Output merged CSV path (e.g., train.csv)")
    ap.add_argument("--save_each_dir", default=None, help="Optional: folder to save per-file parsed CSVs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_csv = Path(args.out_csv)
    if not in_dir.exists():
        print(f"[ERROR] Input folder not found: {in_dir}")
        sys.exit(1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.save_each_dir:
        Path(args.save_each_dir).mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted([p for p in in_dir.rglob("*.xlsx") if not p.name.startswith("~")])
    if not xlsx_files:
        print(f"[ERROR] No .xlsx files under {in_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(xlsx_files)} xlsx files. Parsing...")
    all_dfs = []
    n_ok, n_fail = 0, 0
    for i, fp in enumerate(xlsx_files, 1):
        try:
            df = parse_one(fp, verbose=args.verbose)
            if df is None or df.empty:
                n_fail += 1
                if args.verbose: print(f"  [{i}/{len(xlsx_files)}] {fp.name}: no rows")
                continue
            n_ok += 1
            all_dfs.append(df)
            if args.save_each_dir:
                outp = Path(args.save_each_dir) / f"{fp.stem}_parsed.csv"
                df.to_csv(outp, index=False, encoding="utf-8-sig")
            if args.verbose:
                print(f"  [{i}/{len(xlsx_files)}] {fp.name}: {len(df)} rows")
        except Exception as e:
            n_fail += 1
            print(f"  [{i}/{len(xlsx_files)}] {fp.name}: ERROR {e}")

    if not all_dfs:
        print("[ERROR] No valid rows from any file. Check sheet names/headers/columns.")
        sys.exit(1)

    merged = pd.concat(all_dfs, ignore_index=True)

    # 去重（同一源文件重复行）
    merged.drop_duplicates(
        subset=["source_file","Climate","Type of building","Age of construction"],
        keep="first", inplace=True
    )

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_csv}  | rows={len(merged)}  | ok={n_ok}  fail={n_fail}")

if __name__ == "__main__":
    # Windows 控制台 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
