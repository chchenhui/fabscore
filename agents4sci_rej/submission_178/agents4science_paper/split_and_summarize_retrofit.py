# -*- coding: utf-8 -*-
"""
自动分割 + 汇总：
给定一个包含三张逻辑表（按行顺序拼接到同一 CSV）的文件，
自动识别三个表的表头行并切分为 DataFrame，随后计算改造前后能耗对比、节能率，
并与 stock 的平均能耗对比，输出 retrofit_savings_summary.csv。

表 1：Building stock statistics
  必备列（模糊匹配）：Climate, Type of building, Type of energy use, Average consumption (kWh/m2y)

表 2：Reference buildings simulations
  必备列（模糊匹配）：Climate, Type of building, Age of construction, Type of load, Consumption (kWh/m2y)

表 3：Retrofit package simulations
  优先使用（模糊匹配）：
    Reference building heating/cooling consumption
    Building average heating/cooling consumption
    Building average yearly / Yearly / Total
"""

import argparse
import os
import re
import pandas as pd
import numpy as np

# ----------------------------- 基础工具 -----------------------------
def read_big_csv(path, encoding="utf-8", sep=","):
    """
    更健壮的读取器：
    1) 优先按参数 sep/encoding 读取；
    2) 失败则自动尝试多种编码与分隔符；
    3) 最后使用 pandas 的 sep=None 自动嗅探（python 引擎）。
    """
    import csv

    # 1) 先尝试用户给定的 sep/encoding
    try:
        return pd.read_csv(path, header=None, dtype=str, encoding=encoding,
                           sep=sep, na_filter=False, engine="c")
    except Exception:
        pass

    # 2) 自动尝试常见编码和分隔符组合
    encodings = [encoding, "utf-8-sig", "cp1252"]
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sp in seps:
            try:
                return pd.read_csv(path, header=None, dtype=str, encoding=enc,
                                   sep=sp, na_filter=False, engine="c")
            except Exception:
                continue

    # 3) 最后兜底：pandas 自动嗅探分隔符（python 引擎更宽松）
    try:
        return pd.read_csv(path, header=None, dtype=str,
                           sep=None, engine="python", na_filter=False)
    except Exception as e:
        raise SystemExit(f"❌ 仍无法读取 CSV：{e}")

def norm_text(s):
    s = (s or "").strip()
    s = s.replace("°C", "C").replace("°c", "C")
    s = re.sub(r'[\s\u3000]+', ' ', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^A-Za-z0-9_/\- ]', '', s)
    return s.lower().strip()

def normalize_cols(cols):
    return [norm_text(c) for c in cols]

def to_num(v):
    if v is None:
        return np.nan
    vs = str(v).strip().lower().replace('%','')
    vs = vs.replace(',', '.')
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', vs)
    return float(m.group(0)) if m else np.nan

def num_series(s):
    return s.apply(to_num)

def pick_col(df, candidates):
    cols = list(df.columns)
    # 完全匹配
    for cand in candidates:
        for c in cols:
            if c == cand:
                return c
    # 子串匹配
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    return None

def safe_ratio(n, d):
    n = pd.to_numeric(n, errors='coerce')
    d = pd.to_numeric(d, errors='coerce')
    out = (n / d).replace([np.inf, -np.inf], np.nan)
    return out

# ----------------------- 表头识别 & 切分 -----------------------
def find_header_rows(df_raw):
    """
    扫描每一行，把该行拼成一个逗号连接的大字符串，做列名归一化后判断是否匹配某张表的特征。
    返回：{'stock': idx, 'reference': idx, 'retrofit': idx}（行为索引）
    """
    signatures = {
      'stock': {
          'all': ['climate', 'type of building', 'type of energy use', 'average consumption'],
      },
      'reference': {
          'all': ['climate', 'type of building', 'age of construction', 'type of load', 'consumption'],
      },
      'retrofit': {
          'any': [
              'reference building heating consumption',
              'building average heating consumption',
              'building average yearly',
              'reference building cooling consumption',
              'building average cooling consumption',
              # 有些数据集合并/翻译不同，也尝试更短的关键词
              'reference building heating',
              'reference building cooling',
              'building average cooling',
              'building average heating',
              'average yearly',
              'yearly',
              'total'
          ]
      }
    }

    found = {}
    for i in range(len(df_raw)):
        row_vals = [str(x) for x in df_raw.iloc[i].tolist()]
        header_line = ','.join(row_vals)
        header_norm = norm_text(header_line)

        # 简单判定：列分隔要足够多
        if header_norm.count(',') < 3 and header_norm.count(' ') < 3:
            continue

        def has_all(keys):
            return all(k in header_norm for k in keys)

        def has_any(keys):
            return any(k in header_norm for k in keys)

        if 'stock' not in found and has_all(signatures['stock']['all']):
            found['stock'] = i
            continue
        if 'reference' not in found and has_all(signatures['reference']['all']):
            found['reference'] = i
            continue
        if 'retrofit' not in found and has_any(signatures['retrofit']['any']):
            found['retrofit'] = i
            continue

        if len(found) == 3:
            break

    return found

def slice_table(df_raw, start_idx, end_idx=None):
    """
    把 [start_idx, end_idx) 行切出来，第一行当作表头。
    """
    if end_idx is None:
        sub = df_raw.iloc[start_idx:].reset_index(drop=True)
    else:
        sub = df_raw.iloc[start_idx:end_idx].reset_index(drop=True)
    # 第一行做表头
    header = sub.iloc[0].tolist()
    data = sub.iloc[1:].reset_index(drop=True)
    data.columns = header
    return data

# ----------------------- 三表解析 -----------------------
def parse_stock(df_stock_raw):
    df = df_stock_raw.copy()
    df.columns = normalize_cols(df.columns)

    col_climate   = pick_col(df, ["climate"])
    col_type_bld  = pick_col(df, ["type of building","building type","type building","type"])
    col_type_use  = pick_col(df, ["type of energy use","energy use","use"])
    col_avg_cons  = pick_col(df, ["average consumption","avg consumption","consumption"])

    for c in [col_climate, col_type_bld, col_type_use, col_avg_cons]:
        if c is None:
            raise ValueError("Stock 表缺关键列（climate / type of building / type of energy use / average consumption）")

    out = df[[col_climate, col_type_bld, col_type_use, col_avg_cons]].copy()
    out.columns = ["climate","building_type","energy_use","stock_avg_consumption_kwh_m2y"]
    out["stock_avg_consumption_kwh_m2y"] = num_series(out["stock_avg_consumption_kwh_m2y"])
    out["energy_use"] = out["energy_use"].astype(str).str.strip().str.title()
    out["climate"] = out["climate"].astype(str).str.strip()
    out["building_type"] = out["building_type"].astype(str).str.strip()
    return out.dropna(subset=["stock_avg_consumption_kwh_m2y"])

def parse_reference(df_ref_raw):
    df = df_ref_raw.copy()
    df.columns = normalize_cols(df.columns)

    col_climate   = pick_col(df, ["climate"])
    col_type_bld  = pick_col(df, ["type of building","building type","type building"])
    col_age       = pick_col(df, ["age of construction","age"])
    col_load      = pick_col(df, ["type of load","load"])
    col_cons      = pick_col(df, ["consumption"])

    if any(v is None for v in [col_climate, col_type_bld, col_age, col_load, col_cons]):
        raise ValueError("Reference 表缺关键列（climate / type of building / age of construction / type of load / consumption）")

    out = df[[col_climate, col_type_bld, col_age, col_load, col_cons]].copy()
    out.columns = ["climate","building_type","age","load","ref_consumption_kwh_m2y"]
    out["ref_consumption_kwh_m2y"] = num_series(out["ref_consumption_kwh_m2y"])
    out["load"] = out["load"].astype(str).str.strip().str.title()  # Heating / Cooling / Dhw / Yearly
    out["climate"] = out["climate"].astype(str).str.strip()
    out["building_type"] = out["building_type"].astype(str).str.strip()
    out["age"] = out["age"].astype(str).str.strip()
    return out

def parse_retrofit(df_ret_raw):
    df = df_ret_raw.copy()
    df.columns = normalize_cols(df.columns)

    col_climate = pick_col(df, ["climate"])
    col_type_bld = pick_col(df, ["type of building","building type","type building"])
    col_age = pick_col(df, ["age of construction","age"])

    # 参考（改造前）
    col_ref_heat = pick_col(df, ["reference building heating consumption","ref heating consumption"])
    col_ref_cool = pick_col(df, ["reference building cooling consumption","ref cooling consumption"])
    col_ref_year = pick_col(df, ["reference building yearly","reference building total","reference yearly","ref yearly"])
    # 改造后
    col_after_heat = pick_col(df, ["building average heating consumption","heating consumption"])
    col_after_cool = pick_col(df, ["building average cooling consumption","cooling consumption"])
    col_after_year = pick_col(df, ["building average yearly","yearly","total"])

    base_cols = [col_climate, col_type_bld, col_age]
    if all(x is None for x in base_cols):
        raise ValueError("Retrofit 表缺少基本键（climate / type of building / age）中的至少一个")

    keep = []
    for c in base_cols + [col_ref_heat, col_ref_cool, col_ref_year, col_after_heat, col_after_cool, col_after_year]:
        if c is not None:
            keep.append(c)
    out = df[keep].copy()

    rename_map = {}
    if col_climate: rename_map[col_climate] = "climate"
    if col_type_bld: rename_map[col_type_bld] = "building_type"
    if col_age: rename_map[col_age] = "age"
    if col_ref_heat: rename_map[col_ref_heat] = "ref_heating"
    if col_ref_cool: rename_map[col_ref_cool] = "ref_cooling"
    if col_ref_year: rename_map[col_ref_year] = "ref_yearly"
    if col_after_heat: rename_map[col_after_heat] = "after_heating"
    if col_after_cool: rename_map[col_after_cool] = "after_cooling"
    if col_after_year: rename_map[col_after_year] = "after_yearly"
    out = out.rename(columns=rename_map)

    for c in ["ref_heating","ref_cooling","ref_yearly","after_heating","after_cooling","after_yearly"]:
        if c in out.columns:
            out[c] = num_series(out[c])

    for c in ["climate","building_type","age"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    return out

# ----------------------- 汇总计算 -----------------------
def build_summary(stock_df, ref_df, ret_df):
    rows = []
    key_cols = [c for c in ["climate","building_type","age"] if c in ret_df.columns]
    base = ret_df.copy()

    for load, ref_col, aft_col in [
        ("Heating","ref_heating","after_heating"),
        ("Cooling","ref_cooling","after_cooling"),
        ("Yearly","ref_yearly","after_yearly")
    ]:
        if (ref_col in base.columns) or (aft_col in base.columns):
            tmp = base[key_cols].copy()
            tmp["load"] = load
            tmp["ref_consumption_kwh_m2y"] = base.get(ref_col, np.nan)
            tmp["after_consumption_kwh_m2y"] = base.get(aft_col, np.nan)
            rows.append(tmp)

    part = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=key_cols+["load","ref_consumption_kwh_m2y","after_consumption_kwh_m2y"])

    # 用 reference 表补缺 ref_consumption
    if not ref_df.empty and not part.empty:
        join_keys = [k for k in ["climate","building_type","age","load"] if (k in part.columns) and (k in ref_df.columns)]
        if join_keys:
            part = part.merge(
                ref_df[["climate","building_type","age","load","ref_consumption_kwh_m2y"]],
                how="left",
                on=join_keys,
                suffixes=("", "_ref2")
            )
            if "ref_consumption_kwh_m2y_ref2" in part.columns:
                mask = part["ref_consumption_kwh_m2y"].isna() & part["ref_consumption_kwh_m2y_ref2"].notna()
                part.loc[mask, "ref_consumption_kwh_m2y"] = part.loc[mask, "ref_consumption_kwh_m2y_ref2"]
                part = part.drop(columns=[c for c in part.columns if c.endswith("_ref2")], errors="ignore")

    # 节能率
    part["savings_ratio"] = (part["ref_consumption_kwh_m2y"] - part["after_consumption_kwh_m2y"]) / part["ref_consumption_kwh_m2y"]

    # 合并 stock 平均
    if not stock_df.empty and not part.empty:
        part["energy_use"] = part["load"].str.title()
        for c in ["climate","building_type"]:
            if c in part.columns:
                part[c] = part[c].astype(str).str.strip()
        join_keys_stock = [c for c in ["climate","building_type","energy_use"] if c in stock_df.columns and c in part.columns]
        part = part.merge(stock_df, how="left", on=join_keys_stock)
        part["pct_below_stock"] = 1.0 - safe_ratio(part["after_consumption_kwh_m2y"], part["stock_avg_consumption_kwh_m2y"])
        part["delta_vs_stock_kwh_m2y"] = part["after_consumption_kwh_m2y"] - part["stock_avg_consumption_kwh_m2y"]

    out_cols = [c for c in [
        "climate","building_type","age","load",
        "ref_consumption_kwh_m2y","after_consumption_kwh_m2y","savings_ratio",
        "stock_avg_consumption_kwh_m2y","pct_below_stock","delta_vs_stock_kwh_m2y"
    ] if c in part.columns]
    summary = part[out_cols].copy()

    sort_cols = [c for c in ["climate","building_type","age","load"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    return summary

# ----------------------- 主流程 -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="包含三张逻辑表的单个大CSV路径")
    ap.add_argument("--encoding", default="utf-8", help="CSV 编码（默认 utf-8）")
    ap.add_argument("--sep", default=",", help="分隔符（默认 ,）")
    ap.add_argument("--out", default="retrofit_savings_summary.csv", help="输出CSV路径")
    args = ap.parse_args()

    df_raw = read_big_csv(args.input, encoding=args.encoding, sep=args.sep)

    # 识别表头
    heads = find_header_rows(df_raw)
    if not {'stock','reference','retrofit'}.issubset(set(heads.keys())):
        raise SystemExit(f"❌ 未能识别到全部表头，识别到：{heads}")

    # 计算每段范围
    idx_stock = heads['stock']
    idx_ref   = heads['reference']
    idx_ret   = heads['retrofit']
    starts = sorted([('stock', idx_stock), ('reference', idx_ref), ('retrofit', idx_ret)], key=lambda x: x[1])

    segments = {}
    for i, (name, start) in enumerate(starts):
        end = starts[i+1][1] if i+1 < len(starts) else None
        segments[name] = slice_table(df_raw, start, end)

    df_stock_raw = segments['stock']
    df_ref_raw   = segments['reference']
    df_ret_raw   = segments['retrofit']

    # 解析三表
    stock_df = parse_stock(df_stock_raw)
    ref_df   = parse_reference(df_ref_raw)
    ret_df   = parse_retrofit(df_ret_raw)

    # 汇总计算
    summary = build_summary(stock_df, ref_df, ret_df)

    # 存盘
    summary.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出: {args.out}")
    print(summary.head(10))

if __name__ == "__main__":
    main()
