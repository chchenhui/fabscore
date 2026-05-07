
import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings("ignore")

EXPECTED_SHEETS = {
    "stock": ["Building stock statistics"],
    "ref":   ["Reference buildings simulations"],
    "tgt":   ["Target buildings simulations"],
}

def _try_read_sheet(filepath, sheet_candidates, header_candidates):
    """
    Try multiple sheet names and multiple header row indices robustly.
    Returns the first successfully read DataFrame (drop all-empty rows).
    """
    last_err = None
    for s in sheet_candidates:
        for h in header_candidates:
            try:
                df = pd.read_excel(filepath, sheet_name=s, header=h)
                if df is not None and len(df) > 0:
                    df = df.dropna(how="all")
                    if len(df) > 0:
                        return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Failed to read any of sheets {sheet_candidates} with headers {header_candidates}: {last_err}")

def _normalize_cols(df):
    """Lowercase + strip column names for robust selection."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def analyze_building_energy(filepath):
    """
    Parses a single iNSPiRe Excel file to analyze retrofit energy savings.

    - Reads 3 required sheets (with robust header handling)
    - Links before/after data
    - Computes savings_percentage
    - Joins stock average consumption for comparison
    """
    filename = os.path.basename(filepath)
    print(f"--- Analyzing File: {filename} ---")

    try:
        # Try a few likely header rows for robustness
        header_candidates_stock = [0, 1, 2, 3]
        header_candidates_ref   = [0, 1, 2, 3]
        header_candidates_tgt   = [0, 1, 2, 3]

        df_stock  = _try_read_sheet(filepath, EXPECTED_SHEETS["stock"], header_candidates_stock)
        df_ref    = _try_read_sheet(filepath, EXPECTED_SHEETS["ref"],   header_candidates_ref)
        df_target = _try_read_sheet(filepath, EXPECTED_SHEETS["tgt"],   header_candidates_tgt)

    except Exception as e:
        print(f"  [ERROR] Could not read required sheets from {filename}. Error: {e}")
        return None

    # Normalize column names
    df_stock  = _normalize_cols(df_stock)
    df_ref    = _normalize_cols(df_ref)
    df_target = _normalize_cols(df_target)

    # --- Prepare reference (before) ---
    # Expected columns (robust aliases)
    col_climate = next((c for c in ["climate", "climate zone"] if c in df_ref.columns), None)
    col_type    = next((c for c in ["type of building", "building type", "type"] if c in df_ref.columns), None)
    col_age     = next((c for c in ["age of construction", "construction age", "age"] if c in df_ref.columns), None)
    col_ref_cons= next((c for c in ["consumption (kwh/m2y)", "energy consumption (kwh/m2y)", "baseline consumption (kwh/m2y)"] if c in df_ref.columns), None)

    if not all([col_climate, col_type, col_age, col_ref_cons]):
        print(f"  [WARN] Missing expected columns in REF sheet for {filename}. Skipping.")
        return None

    df_ref_slim = df_ref[[col_climate, col_type, col_age, col_ref_cons]].copy()
    df_ref_slim.rename(columns={
        col_climate: "Climate",
        col_type: "Type of building",
        col_age: "Age of construction",
        col_ref_cons: "consumption_before",
    }, inplace=True)

    # --- Prepare target (after) ---
    # For target, consumption_after may appear as 'building average yearly' (with/without units)
    col_tgt_type = next((c for c in ["type of building", "building type", "type"] if c in df_target.columns), None)
    if col_tgt_type is None and "type of building" in df_ref_slim.columns:
        # We'll align later via merge-by columns from ref
        pass

    # Identify after-consumption column candidates
    tgt_after_candidates = [
        "building average yearly (kwh/m2y)",
        "building average yearly",
        "consumption (kwh/m2y)",
        "energy consumption (kwh/m2y)",
        "post-retrofit consumption (kwh/m2y)",
        "consumption after (kwh/m2y)"
    ]
    col_tgt_after = next((c for c in tgt_after_candidates if c in df_target.columns), None)
    if col_tgt_after is None:
        print(f"  [WARN] Could not locate 'consumption_after' in TGT sheet for {filename}. Skipping.")
        return None

    # Harmonize key columns in df_target for merge
    # Try to map climate/type/age column names
    col_tgt_climate = next((c for c in ["climate", "climate zone"] if c in df_target.columns), None)
    col_tgt_type    = col_tgt_type or next((c for c in ["type of building", "building type", "type"] if c in df_target.columns), None)
    col_tgt_age     = next((c for c in ["age of construction", "construction age", "age"] if c in df_target.columns), None)

    required_keys = [col_tgt_climate, col_tgt_type, col_tgt_age]
    if any(k is None for k in required_keys):
        print(f"  [WARN] Missing merge keys in TGT sheet for {filename}. Skipping.")
        return None

    df_tgt_slim = df_target[[col_tgt_climate, col_tgt_type, col_tgt_age, col_tgt_after]].copy()
    df_tgt_slim.rename(columns={
        col_tgt_climate: "Climate",
        col_tgt_type: "Type of building",
        col_tgt_age: "Age of construction",
        col_tgt_after: "consumption_after",
    }, inplace=True)

    # --- Merge before/after ---
    merge_keys = ["Climate", "Type of building", "Age of construction"]
    df_sim = pd.merge(df_tgt_slim, df_ref_slim, on=merge_keys, how="inner")

    if df_sim.empty:
        print(f"  [WARN] No matching records found between 'before' and 'after' in {filename}.")
        return None

    # --- Compute savings ---
    df_sim["consumption_before"] = pd.to_numeric(df_sim["consumption_before"], errors="coerce")
    df_sim["consumption_after"]  = pd.to_numeric(df_sim["consumption_after"],  errors="coerce")
    df_sim = df_sim.replace([np.inf, -np.inf], np.nan).dropna(subset=["consumption_before", "consumption_after"])
    df_sim = df_sim[df_sim["consumption_before"] > 0]

    if df_sim.empty:
        print(f"  [WARN] No valid rows after cleaning in {filename}.")
        return None

    df_sim["savings_percentage"] = (df_sim["consumption_before"] - df_sim["consumption_after"]) / df_sim["consumption_before"]

    # --- Stock average merge ---
    df_stock = df_stock.copy()
    # find stock columns
    stock_climate = next((c for c in ["climate", "climate zone"] if c in df_stock.columns), None)
    stock_type    = next((c for c in ["type of building", "building type", "type"] if c in df_stock.columns), None)
    stock_avg     = next((c for c in ["average consumption (kwh/m2y)", "avg consumption (kwh/m2y)", "average (kwh/m2y)"] if c in df_stock.columns), None)

    if not all([stock_climate, stock_type, stock_avg]):
        print(f"  [WARN] Missing expected columns in STOCK sheet for {filename}. Proceeding without stock average.")
        df_sim["stock_average_consumption"] = np.nan
    else:
        df_stock_slim = df_stock[[stock_climate, stock_type, stock_avg]].copy()
        df_stock_slim.rename(columns={
            stock_climate: "Climate",
            stock_type: "Type of building",
            stock_avg: "stock_average_consumption",
        }, inplace=True)
        df_stock_avg = df_stock_slim.groupby(["Climate", "Type of building"], as_index=False)["stock_average_consumption"].mean()
        df_sim = pd.merge(df_sim, df_stock_avg, on=["Climate", "Type of building"], how="left")

    # Ensure tidy dtypes
    num_cols = ["consumption_before", "consumption_after", "savings_percentage", "stock_average_consumption"]
    for c in num_cols:
        if c in df_sim.columns:
            df_sim[c] = pd.to_numeric(df_sim[c], errors="coerce")

    return df_sim

def process_folder(root_dir, out_csv, preview_n=5):
    """
    Scan a folder for .xlsx files, run analyze_building_energy on each,
    concatenate, and save a consolidated train.csv
    """
    xlsx_files = sorted(
        f for f in glob.glob(os.path.join(root_dir, "**", "*.xlsx"), recursive=True)
        if not os.path.basename(f).startswith("~$")  # ignore temp files
    )
    print(f"Found {len(xlsx_files)} .xlsx files under: {root_dir}")

    results = []
    failed  = []

    for f in xlsx_files:
        try:
            df = analyze_building_energy(f)
            if df is not None and len(df) > 0:
                df["source_file"] = os.path.basename(f)
                results.append(df)
        except Exception as e:
            print(f"  [ERROR] Fatal processing error for {os.path.basename(f)}: {e}")
            failed.append((f, str(e)))

    if not results:
        print("No valid rows collected. Nothing to save.")
        return None

    df_all = pd.concat(results, ignore_index=True)
    # Drop perfect duplicates just in case
    df_all = df_all.drop_duplicates()

    # Save
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n✅ Saved consolidated dataset to: {out_csv}")
    print(f"   Total rows: {len(df_all):,}")
    print(f"   From files: {df_all['source_file'].nunique()}")

    # Simple preview
    keep_cols = [c for c in [
        "source_file", "Type of building", "Climate", "Age of construction",
        "consumption_before", "consumption_after", "savings_percentage", "stock_average_consumption"
    ] if c in df_all.columns]

    print("\n--- Preview ---")
    print(df_all[keep_cols].head(preview_n))

    # Return df for interactive sessions
    return df_all

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch-process iNSPiRe FP7 Excel files into a consolidated train.csv")
    parser.add_argument("--root", required=True, help="Root folder of 'iNSPiRe FP7 Retrofit Database'")
    parser.add_argument("--out_csv", required=True, help="Path to output CSV, e.g., ./train.csv")
    parser.add_argument("--preview_n", type=int, default=5, help="Rows to preview in stdout")
    args = parser.parse_args()

    process_folder(args.root, args.out_csv, args.preview_n)
