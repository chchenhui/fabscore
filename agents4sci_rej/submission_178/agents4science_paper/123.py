# -*- coding: utf-8 -*-
"""
Compute per-building energy savings from ResStock baseline/upgrade CSVs.

Requirements:
  pip install pandas

Author: you + ChatGPT
"""

import os
import pandas as pd
from typing import Optional

# ----------- CONFIG -----------
BASE_DIR = r"C:\Users\liang\Desktop\Agent4Science2025\dataset\ResStock"
BASELINE_FILE = "baseline_metadata_and_annual_results.csv"
UPGRADE_FILE  = "upgrade01_metadata_and_annual_results.csv"
OUT_FILE = "_savings_by_building.csv"

# 主指标列（最小可用公式）
COL_BASE_TOTAL = "out.site_energy.total.energy_consumption.kwh"
COL_UPG_SAV    = "out.site_energy.total.energy_consumption.kwh.savings"  # = baseline - upgrade

# 可能用到的“来源列名”候选（不同版本/导出可能差异）
CANDIDATE_COLS = {
    # 气候区：优先 ASHRAE/IECC，其次 Building America
    "climate": [
        "in.ashrae_iecc_climate_zone_2004",
        "in.energystar_climate_zone_2023",
        "in.building_america_climate_zone",
        "in.cec_climate_zone",
    ],
    # 建筑类型
    "btype": [
        "in.geometry_building_type_recs",
        "in.geometry_building_type_acs",
        "in.geometry_space_combination",
    ],
    # 建成年代
    "period": [
        "in.vintage",
        "in.vintage_acs",
    ],
    # 面积（sqft）
    "area_sqft": [
        "in.geometry_floor_area",
        "in.sqft",
        "in.area_floor",  # 偶见
    ],
    # 供暖/制冷设备（用于生成 system）
    "heat_type": [
        "in.hvac_heating_type_and_fuel",
        "in.hvac_heating_type",
    ],
    "heat_eff": [
        "in.hvac_heating_efficiency",
    ],
    "cool_type": [
        "in.hvac_cooling_type",
    ],
    "cool_eff": [
        "in.hvac_cooling_efficiency",
    ],
}


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_system_col(df: pd.DataFrame) -> pd.Series:
    """Combine heating/cooling type/eff into a compact 'system' string."""
    ht_col = pick_col(df, CANDIDATE_COLS["heat_type"])
    he_col = pick_col(df, CANDIDATE_COLS["heat_eff"])
    ct_col = pick_col(df, CANDIDATE_COLS["cool_type"])
    ce_col = pick_col(df, CANDIDATE_COLS["cool_eff"])

    def fmt(row):
        parts = []
        if ht_col and pd.notna(row.get(ht_col, None)) and str(row.get(ht_col)).strip():
            s = f"H:{row[ht_col]}"
            if he_col and pd.notna(row.get(he_col, None)):
                s += f" (η={row[he_col]})"
            parts.append(s)
        if ct_col and pd.notna(row.get(ct_col, None)) and str(row.get(ct_col)).strip():
            s = f"C:{row[ct_col]}"
            if ce_col and pd.notna(row.get(ce_col, None)):
                s += f" (SEER/EER={row[ce_col]})"
            parts.append(s)
        return " | ".join(parts) if parts else None

    return df.apply(fmt, axis=1)


def main():
    baseline_path = os.path.join(BASE_DIR, BASELINE_FILE)
    upgrade_path  = os.path.join(BASE_DIR, UPGRADE_FILE)
    out_path = os.path.join(BASE_DIR, OUT_FILE)

    # 读文件：dtype=str 可以避免科学计数/丢精度；随后对需要的列再转数值
    baseline = pd.read_csv(baseline_path, low_memory=False)
    upgrade  = pd.read_csv(upgrade_path,  low_memory=False)

    # 基本存在性检查
    req_cols_baseline = {"bldg_id", COL_BASE_TOTAL}
    req_cols_upgrade  = {"bldg_id", COL_UPG_SAV}
    missing_b = req_cols_baseline - set(baseline.columns)
    missing_u = req_cols_upgrade  - set(upgrade.columns)
    if missing_b:
        raise ValueError(f"Baseline CSV 缺少列: {missing_b}")
    if missing_u:
        raise ValueError(f"Upgrade CSV 缺少列: {missing_u}")

    # 选择参数列来源
    climate_col = pick_col(baseline, CANDIDATE_COLS["climate"])
    btype_col   = pick_col(baseline, CANDIDATE_COLS["btype"])
    period_col  = pick_col(baseline, CANDIDATE_COLS["period"])
    area_col    = pick_col(baseline, CANDIDATE_COLS["area_sqft"])

    # 准备 baseline 子集
    keep_cols_base = ["bldg_id", COL_BASE_TOTAL]
    for c in [climate_col, btype_col, period_col, area_col]:
        if c and c not in keep_cols_base:
            keep_cols_base.append(c)
    # 如果需要，额外用于 system 的列也从 baseline 提取
    for key in ("heat_type", "heat_eff", "cool_type", "cool_eff"):
        c = pick_col(baseline, CANDIDATE_COLS[key])
        if c and c not in keep_cols_base:
            keep_cols_base.append(c)

    base_sub = baseline[keep_cols_base].copy()

    # 数值化总能耗列
    base_sub[COL_BASE_TOTAL] = pd.to_numeric(base_sub[COL_BASE_TOTAL], errors="coerce")

    # 生成 system 列
    base_sub["system"] = build_system_col(base_sub)

    # 英制面积转 m²（如无面积列则置空）
    if area_col:
        base_sub["area_m2"] = pd.to_numeric(base_sub[area_col], errors="coerce") * 0.092903
    else:
        base_sub["area_m2"] = pd.NA

    # 统一参数列命名
    rename_map = {}
    if climate_col: rename_map[climate_col] = "climate"
    if btype_col:   rename_map[btype_col]   = "btype"
    if period_col:  rename_map[period_col]  = "period"
    base_sub = base_sub.rename(columns=rename_map)

    # 升级表：保留 upgrade id/name（可选），以及 savings 列
    keep_cols_upg = ["bldg_id", "upgrade", "upgrade_name"] if "upgrade_name" in upgrade.columns else ["bldg_id", "upgrade"]
    keep_cols_upg.append(COL_UPG_SAV)
    upg_sub = upgrade[keep_cols_upg].copy()
    upg_sub[COL_UPG_SAV] = pd.to_numeric(upg_sub[COL_UPG_SAV], errors="coerce")

    # 合并（按 bldg_id；如同一 bldg 有多种 upgrade，可后续按 upgrade 过滤/分组）
    df = pd.merge(upg_sub, base_sub, on="bldg_id", how="left", validate="m:1")

    # 计算节能百分比：pct_sav = savings / baseline_total * 100
    df["baseline_total_kwh"] = df[COL_BASE_TOTAL]
    df["kwh_sav"] = df[COL_UPG_SAV]
    df["pct_sav"] = (df["kwh_sav"] / df["baseline_total_kwh"]) * 100.0

    # 清理与选择输出列
    out_cols = [
        "bldg_id",
        "upgrade",
    ]
    if "upgrade_name" in df.columns:
        out_cols.append("upgrade_name")

    # 目标参数列（如不存在则自动跳过）
    for c in ["climate", "btype", "period", "area_m2", "system"]:
        if c in df.columns:
            out_cols.append(c)

    out_cols += [
        "baseline_total_kwh",   # 改造前总能耗
        "kwh_sav",              # 绝对节能量（kWh）
        "pct_sav",              # 节能百分比（%）
    ]

    df_out = df[out_cols].copy()

    # 可选：处理无穷/NaN
    df_out["pct_sav"] = df_out["pct_sav"].replace([float("inf"), -float("inf")], pd.NA)

    # 导出
    out_full_path = os.path.join(BASE_DIR, OUT_FILE)
    df_out.to_csv(out_full_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {out_full_path}")
    print(df_out.head(10))


if __name__ == "__main__":
    main()
