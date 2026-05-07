# clean_target.py
import pandas as pd, numpy as np

p_in  = r"C:\Users\liang\Desktop\Agent4Science2025\dataset\train.csv"
p_out = r"C:\Users\liang\Desktop\Agent4Science2025\dataset\train_clean.csv"
target = "Average demand (kWh/m2y)"

df = pd.read_csv(p_in)
y  = pd.to_numeric(df.get(target), errors="coerce")
bad = ~np.isfinite(y)

print(f"[INFO] rows={len(df)}  drop_bad_target={int(bad.sum())}")
df = df.loc[~bad].copy()

df.to_csv(p_out, index=False, encoding="utf-8-sig")
print(f"[OK] wrote {p_out}  rows={len(df)}")
