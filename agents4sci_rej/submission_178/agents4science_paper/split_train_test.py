# split_train_test.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser(description="Clean dataset and split into train/test (80/20).")
    ap.add_argument("--in_csv", required=True, help="Input CSV file (raw merged dataset)")
    ap.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    ap.add_argument("--target", default="Average demand (kWh/m2y)", help="Target column name")
    ap.add_argument("--test_size", type=float, default=0.2, help="Fraction for test set (default 0.2)")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    n0 = len(df)

    # ---- 1. 目标列清洗 ----
    if args.target not in df.columns:
        raise SystemExit(f"[ERROR] Target column '{args.target}' not found in {args.in_csv}")

    y = pd.to_numeric(df[args.target], errors="coerce")
    m = np.isfinite(y) & (y > 0)
    df = df.loc[m].copy()
    n1 = len(df)
    print(f"[CLEAN] Removed {n0-n1} invalid rows (keep {n1}/{n0}).")

    # ---- 2. 特征清洗（转字符串、去空格） ----
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # ---- 3. Train/Test 划分 ----
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed
    )

    # ---- 4. 保存结果 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_split.csv"
    test_path = out_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Train set: {train_path} | rows={len(train_df)}")
    print(f"[OK] Test set : {test_path} | rows={len(test_df)}")

if __name__ == "__main__":
    main()
