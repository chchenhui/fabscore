# merge_inspire_to_train.py  (CLI + recursive glob + verbose)
import re, sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", default=None, help="包含 iNSPiRe xlsx 的目录（可含子目录）")
ap.add_argument("--files",  default=None, nargs="*", help="显式列出若干 xlsx 文件路径（可与 --in_dir 二选一）")
ap.add_argument("--deg_days_csv", default=None, help="可选：度日表 CSV（列：climate,HDD_18,CDD_22）")
ap.add_argument("--out_csv", default="train.csv", help="输出 train.csv 路径")
args = ap.parse_args()

# ---------- 1) 收集文件 ----------
xlsx_files = []
if args.files:
    xlsx_files = [str(Path(f)) for f in args.files if str(f).lower().endswith(".xlsx")]
elif args.in_dir:
    p = Path(args.in_dir)
    if not p.exists():
        sys.exit(f"目录不存在：{p}")
    # 递归搜 *.xlsx（兼容大小写）
    xlsx_files = [str(x) for x in p.rglob("*.xlsx")]
else:
    sys.exit("请提供 --in_dir 或 --files。")

if not xlsx_files:
    sys.exit("未找到任何 xlsx，请检查 --in_dir 或 --files 指定的路径/通配。")

print("[INFO] 找到 xlsx 文件：")
for fp in xlsx_files:
    print("  -", fp)

# ---------- 2) 文件名解析 ----------
re_fname = re.compile(
    r"(?P<btype>SFH|sMFH)_(?P<period>I|III)_?(?P<floors>\d+FL)?_?"
    r"(?P<system>AWHP|GWHP|COND|PELL)_?(?P<terminal>CEI|FC|RAD)?",
    re.I
)
MAP_BTYPE = {"SFH":"single_family_house", "sMFH":"small_multi_family_house"}
MAP_PERIOD= {"I":"I","III":"III"}
MAP_SYSTEM= {"AWHP":"air_to_water_heat_pump","GWHP":"ground_water_heat_pump","COND":"boiler_condensing","PELL":"boiler_pellet"}
MAP_TERMINAL={"CEI":"ceiling_panel","FC":"fan_coil","RAD":"radiator"}

def parse_from_name(path_str: str):
    name = Path(path_str).stem
    m = re_fname.search(name)
    if not m:
        return {"btype": np.nan, "period": np.nan, "system": np.nan,
                "terminal": np.nan, "floors": np.nan}
    gd = m.groupdict()
    btype   = MAP_BTYPE.get(gd.get("btype","").upper(), np.nan)
    period  = MAP_PERIOD.get(gd.get("period","").upper(), np.nan)
    system  = MAP_SYSTEM.get(gd.get("system","").upper(), np.nan)
    terminal= MAP_TERMINAL.get(gd.get("terminal","").upper(), np.nan)
    floors  = gd.get("floors")
    if floors and floors.upper().endswith("FL"):
        try: floors = int(floors[:-2])
        except: floors = np.nan
    else:
        floors = np.nan
    return {"btype": btype, "period": period, "system": system,
            "terminal": terminal, "floors": floors}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- 3) 读取与合并 ----------
rows = []
for fp in xlsx_files:
    try:
        raw = pd.read_excel(fp, engine="openpyxl")
    except Exception:
        raw = pd.read_excel(fp)
    raw = normalize_columns(raw)

    meta = parse_from_name(fp)
    meta_df = pd.DataFrame({k:[v] for k,v in meta.items()})
    meta_df["source_file"] = Path(fp).name
    # 将 meta 复制到每行
    meta_expanded = pd.concat([meta_df]*len(raw), ignore_index=True)

    out = pd.concat([meta_expanded, raw.reset_index(drop=True)], axis=1)
    rows.append(out)

df_all = pd.concat(rows, ignore_index=True)

# ---------- 4) 轻度规范 ----------
for c in ["btype","period","system","terminal","source_file"]:
    if c in df_all.columns:
        df_all[c] = df_all[c].astype(str).str.strip()
if "climate" not in df_all.columns:
    df_all["climate"] = np.nan
if "floors" in df_all.columns:
    df_all["floors"] = pd.to_numeric(df_all["floors"], errors="coerce")

# ---------- 5) 可选：合并度日 ----------
if args.deg_days_csv:
    ddp = Path(args.deg_days_csv)
    if ddp.exists():
        dd = pd.read_csv(ddp)
        need = {"climate","HDD_18","CDD_22"}
        if not need.issubset(set(dd.columns)):
            sys.exit(f"度日表缺少这些列：{need - set(dd.columns)}")
        df_all = df_all.merge(dd[["climate","HDD_18","CDD_22"]], on="climate", how="left")
    else:
        print(f"[WARN] 找不到度日表：{ddp}，跳过合并。")

# ---------- 6) 丢掉全空列并导出 ----------
empty_cols = [c for c in df_all.columns if df_all[c].isna().all()]
if empty_cols:
    print("[INFO] 丢掉全空列：", empty_cols)
df_all = df_all.drop(columns=empty_cols)

outp = Path(args.out_csv)
outp.parent.mkdir(parents=True, exist_ok=True)
df_all.to_csv(outp, index=False, encoding="utf-8-sig")
print(f"[OK] Wrote {outp} | rows={len(df_all)} cols={len(df_all.columns)}")
