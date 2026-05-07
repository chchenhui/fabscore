# inspect_resstock.py
import argparse, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--n", type=int, default=20, help="print first N columns")
args = ap.parse_args()

p = Path(args.input)
if p.suffix.lower() == ".parquet":
    df = pd.read_parquet(p)
else:
    df = pd.read_csv(p)

print(f"[INFO] rows={len(df)}, cols={len(df.columns)}")
print("\n=== All columns ===")
for i, c in enumerate(df.columns):
    print(f"{i:>3} | {c}")
    if i+1 >= args.n: break

# 关键字段的模糊搜索
keys = {
    "btype": ["bldg","building type","resstock building type","geometry building type","residential building type"],
    "vintage": ["vintage","year built","period","year built bin"],
    "heat": ["heating system","heating type","primary heating","heating equipment"],
    "fuel": ["heating fuel","fuel type","primary heating fuel"],
    "terminal": ["distribution","terminal","hvac terminal","heat distribution"],
    "stories": ["stories","num stories","number of stories","stories in building"],
    "climate": ["climate","iecc climate","ashrae climate"],
    "area_m2": ["conditioned floor area (m2)","floor area m2","conditioned floor area","floor area"],
    "kwh_total": ["total site energy (kwh)","site energy kwh","annual_kwh","total electricity (kwh)","electricity: total (kwh)","gas: total (kwh)","total site energy (mmbtu)","site energy (mmbtu)"],
}
print("\n=== Candidate columns (fuzzy match) ===")
lowers = {c.lower(): c for c in df.columns}
def find_one(cands):
    for q in cands:
        for name_lower, orig in lowers.items():
            if q in name_lower:
                return orig
    return None
for k, cands in keys.items():
    col = find_one([q.lower() for q in cands])
    print(f"{k:>10}: {col}")

print("\n=== Head (5) ===")
print(df.head(5).to_string(index=False))
