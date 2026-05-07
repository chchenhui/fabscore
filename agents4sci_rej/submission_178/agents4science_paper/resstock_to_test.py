# resstock_to_test.py  -- ResStock → iNSPiRe-style test CSV (auto-discovery version)
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def to_num(s): return pd.to_numeric(s, errors="coerce")

def pick_first(df, candidates, required=False, default=None):
    """Fuzzy match over provided candidates (case-insensitive)."""
    lowers = {c.lower(): c for c in df.columns}
    for q in candidates:
        ql = q.lower()
        for name_lower, orig in lowers.items():
            if ql in name_lower:
                return orig
    if required:
        raise SystemExit(f"[ERROR] Required columns not found. Tried: {candidates}")
    return default

def find_col_regex(df, patterns, prefer=None):
    """Search any column whose name matches any regex in patterns and is not all-NaN.
       prefer: optional regex to prefer among matches."""
    cols = []
    for c in df.columns:
        name = c.lower()
        for pat in patterns:
            if re.search(pat, name):
                if not to_num(df[c]).isna().all() if df[c].dtype != object else df[c].notna().any():
                    cols.append(c)
                    break
    if not cols:
        return None
    if prefer:
        pref = [c for c in cols if re.search(prefer, c.lower())]
        if pref:
            return pref[0]
    return cols[0]

# ---- iNSPiRe-style mappers ----
def map_system_inspire(s: str) -> str:
    s = str(s).lower()
    if "heat pump" in s or "hp" in s or "mini-split" in s or "split" in s or "packaged" in s:
        return "air_to_water_heat_pump"
    if "boiler" in s or "furnace" in s or "condens" in s:
        if "pellet" in s or "wood" in s or "biomass" in s:
            return "boiler_pellet"
        return "boiler_condensing"
    return "other"

def map_terminal_inspire(t: str) -> str:
    t = str(t).lower()
    if "fan" in t or "coil" in t: return "fan_coil"
    if "forced air" in t or "forced_air" in t or re.search(r"\bair\b", t): return "fan_coil"
    if "baseboard" in t or "rad" in t or "radiator" in t: return "radiator"
    if "cei" in t or "ceiling" in t or "panel" in t: return "ceiling_panel"
    return "other"

def build_test_df(df, args):
    IDX = df.index  # unified index
    lowers = {c.lower(): c for c in df.columns}

    # ---------- 1) Basic categorical fields (用 in./直觉 名称 + 自动兜底) ----------
    c_btype   = pick_first(df, ["in.geometry_building_type_recs","in.geometry_building_type",
                                "in.residential_building_type","building type","bldg"])
    c_vintage = pick_first(df, ["in.vintage","year built bin","year built","period"])
    c_clim    = pick_first(df, ["in.ashrae_iecc_climate_zone_2004","in.iecc_climate_zone",
                                "iecc climate zone","ashrae climate","climate zone","climate"])
    c_stories = pick_first(df, ["in.geometry_stories","number of stories","num stories",
                                "stories in building","stories"])

    def S_str(col, default=""):
        if col:
            return df[col].astype(str).str.strip().reindex(IDX)
        return pd.Series([default]*len(IDX), index=IDX, dtype="object")

    def S_num(col, factor=1.0):
        if col:
            return pd.to_numeric(df[col], errors="coerce").reindex(IDX) * factor
        return pd.Series([np.nan]*len(IDX), index=IDX, dtype="float64")

    btype   = S_str(c_btype,   default="")
    period  = S_str(c_vintage, default="")
    climate = S_str(c_clim,    default="")
    floors  = S_num(c_stories)

    # ---------- 2) Heating / Fuel / Terminal (自动发现 + 备选) ----------
    # try explicit picks first
    c_heat = pick_first(df, ["in.heating_system_type","in.heating_system","in.primary_heating_system",
                             "in.space_heating_system","heating system","heating equipment","heating type"])
    c_fuel = pick_first(df, ["in.heating_fuel","in.primary_heating_fuel","in.heating_fuel_type",
                             "heating fuel","fuel type"])
    c_term = pick_first(df, ["in.heating_distribution","in.heating_distribution_system","in.hvac_distribution",
                             "in.hvac_system_type","distribution system","distribution","terminal","hvac terminal"])

    # if not found, auto regex
    if not c_heat:
        c_heat = find_col_regex(df, [r"\bin\.?space_?heating_?system\b", r"\bin\.?heating_?system(_type)?\b",
                                     r"\bheating (system|equipment|type)\b"])
    if not c_fuel:
        c_fuel = find_col_regex(df, [r"\bin\.?heating_?fuel(_type)?\b", r"\b(primary )?heating fuel\b", r"\bfuel type\b"])
    if not c_term:
        c_term = find_col_regex(df, [r"\bin\.?heating_?distribution(_system)?\b", r"\bhvac (distribution|terminal)\b",
                                     r"\bdistribution system\b", r"\bterminal\b"])
        if not c_term:
            # as last resort, some datasets put hot water distribution; it's not ideal but better than nothing
            c_term = pick_first(df, ["in.hot_water_distribution"])

    heat_s = S_str(c_heat, default="")
    fuel_s = S_str(c_fuel, default="")
    term_s = S_str(c_term, default="unknown")

    system_raw = (heat_s.where(heat_s.ne(""), other="") +
                  pd.Series([" | "]*len(IDX), index=IDX) +
                  fuel_s.where(fuel_s.ne(""), other="")).str.strip(" |")

    # ---------- 3) Floor area (auto-detect) ----------
    # Prefer any column with m2; else ft^2 → m2; else pick most plausible "area" numeric with reasonable median
    c_area_m2 = find_col_regex(df, [r"m2", r"\b\(m2\)\b", r"area_m2", r"floor.?area.*m2", r"conditioned.*m2"], prefer=r"conditioned|geometry")
    c_area_ft = find_col_regex(df, [r"ft2|\(ft\^?2\)|floor.?area.*ft", r"conditioned.*ft"], prefer=r"conditioned|geometry")

    area_m2 = S_num(c_area_m2)
    if area_m2.isna().all():
        area_m2 = S_num(c_area_ft, factor=0.092903)

    # ---------- 4) Annual site energy in kWh (auto-detect) ----------
    # Priority 1: a clear total site energy in kWh
    c_kwh_total = find_col_regex(df, [r"total.*site.*kwh", r"\bsite.*energy.*kwh\b", r"\bannual_?kwh\b"])
    kwh = S_num(c_kwh_total)

    # Priority 2: if total kWh not found, try sum of per-fuel end uses in kWh
    if kwh.isna().all():
        kwh_cols = [c for c in df.columns if re.search(r"kwh", c.lower()) and re.search(r"(total|site|electricity|gas|fuel|end ?use)", c.lower())]
        if kwh_cols:
            tmp = pd.DataFrame({c: to_num(df[c]).reindex(IDX) for c in kwh_cols})
            kwh = tmp.sum(axis=1, min_count=1)

    # Priority 3: try MMBtu/kBtu/therm → kWh
    if kwh.isna().all():
        # total MMBtu / kBtu / therm
        c_mmbtu = find_col_regex(df, [r"total.*site.*mmbtu", r"\bsite.*energy.*mmbtu\b", r"annual.*mmbtu"])
        c_kbtu  = find_col_regex(df, [r"total.*site.*kbtu", r"\bsite.*energy.*kbtu\b", r"annual.*kbtu"])
        c_therm = find_col_regex(df, [r"therm"], prefer=r"total|site|annual")

        if c_mmbtu:
            kwh = S_num(c_mmbtu) * 293.071
        elif c_kbtu:
            kwh = S_num(c_kbtu) * 0.293071
        elif c_therm:
            kwh = S_num(c_therm) * 29.3001
        else:
            # try per-fuel MMBtu/kBtu columns
            fuel_cols_mmbtu = [c for c in df.columns if re.search(r"(electric|gas|propane|fuel.?oil|wood|biomass).*mmbtu", c.lower())]
            fuel_cols_kbtu  = [c for c in df.columns if re.search(r"(electric|gas|propane|fuel.?oil|wood|biomass).*kbtu",  c.lower())]
            if fuel_cols_mmbtu:
                tmp = pd.DataFrame({c: to_num(df[c]).reindex(IDX) for c in fuel_cols_mmbtu})
                kwh = tmp.sum(axis=1, min_count=1) * 293.071
            elif fuel_cols_kbtu:
                tmp = pd.DataFrame({c: to_num(df[c]).reindex(IDX) for c in fuel_cols_kbtu})
                kwh = tmp.sum(axis=1, min_count=1) * 0.293071

    out = pd.DataFrame({
        "building_id": np.arange(1, len(IDX)+1),
        "btype":   btype,
        "period":  period,
        "system":  system_raw,
        "terminal":term_s,
        "floors":  floors,
        "climate": climate,
        "area_m2": area_m2,
        "annual_kwh": kwh,
    }, index=IDX).reset_index(drop=True)

    # ---------- 5) Target ----------
    out["annual_eui_kwh_m2"] = np.where(
        (out["area_m2"] > 0) & np.isfinite(out["annual_kwh"]),
        out["annual_kwh"] / out["area_m2"],
        np.nan
    )

    # ---------- 6) Optional filters ----------
    if args.filter_btype:   out = out[out["btype"].str.lower().str.contains(args.filter_btype.lower(), na=False)]
    if args.filter_period:  out = out[out["period"].str.contains(args.filter_period, na=False)]
    if args.filter_climate: out = out[out["climate"].str.contains(args.filter_climate, na=False)]
    if args.filter_fuel:    out = out[out["system"].str.lower().str.contains(args.filter_fuel.lower(), na=False)]

    # ---------- 7) Harmonize enums ----------
    out["system"]   = out["system"].map(map_system_inspire)
    out["terminal"] = out["terminal"].map(map_terminal_inspire)

    # ---------- 8) Sampling ----------
    if args.sample and 0 < args.sample < len(out):
        out = out.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    # ---------- 9) Degree-days merge ----------
    if args.deg_days_csv:
        ddp = Path(args.deg_days_csv)
        if ddp.exists():
            dd = pd.read_csv(ddp)
            need = {"climate","HDD_18","CDD_22"}
            if not need.issubset(dd.columns):
                raise SystemExit(f"[ERROR] Degree-days CSV must contain: {need}")
            out["climate"] = out["climate"].astype(str).str.strip()
            dd["climate"]  = dd["climate"].astype(str).str.strip()
            out = out.merge(dd[["climate","HDD_18","CDD_22"]], on="climate", how="left")
        else:
            print(f"[WARN] Degree-days CSV not found: {ddp}")

    for c in ["btype","period","system","terminal","climate"]:
        out[c] = out[c].astype(str).str.strip()

    return out.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Convert ResStock outputs to iNSPiRe-style test CSV (auto-discovery)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--deg_days_csv", default=None)
    ap.add_argument("--filter_btype", default=None)
    ap.add_argument("--filter_period", default=None)
    ap.add_argument("--filter_climate", default=None)
    ap.add_argument("--filter_fuel", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists(): raise SystemExit(f"[ERROR] Input file not found: {p}")
    df = pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)

    out = build_test_df(df, args)
    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote {outp} | rows={len(out)} cols={out.shape[1]}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
