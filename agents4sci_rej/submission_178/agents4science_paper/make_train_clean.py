# make_train_clean.py
import argparse, re
import pandas as pd, numpy as np
from pathlib import Path

def to_num(s): 
    return pd.to_numeric(s, errors="coerce")

def find_col(df, *cands_regex):
    cols = [c for c in df.columns]
    lowers = {c.lower(): c for c in cols}
    for pat in cands_regex:
        rp = re.compile(pat, flags=re.I)
        for name_lower, orig in lowers.items():
            if rp.search(name_lower):
                return orig
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
n0 = len(df)

# -------- 目标列优先级 --------
y_col = "Average demand (kWh/m2y)"
y = to_num(df[y_col]) if y_col in df.columns else pd.Series([np.nan]*n0)

if y.notna().sum() == 0:
    alt = [
        "Average consumption (kWh/m2y)",
        r"(specific|site).*energy.*(kwh/?m2|eui)",
        r"\beui\b",
    ]
    used = None
    for a in alt:
        if a.startswith("(") or a.startswith(r"\b"):   # 正则
            c = find_col(df, a)
        else:
            c = a if a in df.columns else None
        if c is not None:
            tmp = to_num(df[c])
            if tmp.notna().sum() > 0:
                y, used = tmp, c
                break
    if args.verbose:
        print(f"[INFO] fallback y from: {used}")

# -------- 再兜底：kWh / 面积 --------
if y.notna().sum() == 0:
    # 能耗列
    c_kwh = None
    for pat in [r"annual.*kwh", r"total.*site.*kwh", r"site.*energy.*kwh", r"electricity: total \(kwh\)"]:
        c_kwh = find_col(df, pat)
        if c_kwh: break
    # 面积列（m2 优先，否则 ft^2→m2）
    c_m2 = None
    for pat in [r"(conditioned )?floor.*area.*\(m2\)", r"area_m2", r"floor.*m2"]:
        c_m2 = find_col(df, pat)
        if c_m2: break
    c_ft = None
    if not c_m2:
        for pat in [r"(conditioned )?floor.*area.*\(ft\^?2\)", r"floor.*ft"]:
            c_ft = find_col(df, pat)
            if c_ft: break

    if c_kwh is not None and (c_m2 or c_ft):
        area = to_num(df[c_m2]) if c_m2 else to_num(df[c_ft]) * 0.092903
        kwh  = to_num(df[c_kwh])
        y = np.where((area > 0) & np.isfinite(kwh), kwh/area, np.nan)

# 写回目标列名为原名（训练脚本会找它）
df[y_col] = pd.to_numeric(y, errors="coerce")

# -------- 同步保证特征列 --------
# 若只有 'climate' 列而没有 'Climate'，复制一份
if "Climate" not in df.columns and "climate" in df.columns:
    df["Climate"] = df["climate"].astype(str)
# 基本类别列转字符串
for c in ["period","system","terminal","Climate","btype"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# -------- 丢掉目标为非有限数的行 --------
bad = ~np.isfinite(df[y_col])
if "source_file" in df.columns:
    print("[INFO] bad target rows by source_file (top):")
    print(df.loc[bad, "source_file"].value_counts().head(15).to_string())

df = df.loc[~bad].copy()
n1 = len(df)

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
print(f"[OK] cleaned: {args.in_csv} -> {args.out_csv} | kept {n1}/{n0} rows")
