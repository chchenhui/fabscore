# split_inspire_holdout.py
import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--train_out", required=True)
ap.add_argument("--test_out", required=True)
ap.add_argument("--test_size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=2025)
# 按“域语义”分组，避免泄漏；可按需增减
ap.add_argument("--group_cols", default="btype,period,system,terminal,source_file")
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
# 保障这些列存在（不存在就填空）
gcols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
for c in gcols:
    if c not in df.columns: df[c] = ""

# 组键（字符串拼接）
grp_key = df[gcols].astype(str).agg("||".join, axis=1)

# GroupShuffleSplit：按组随机切分，避免同一组样本泄漏到两边
gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
train_idx, test_idx = next(gss.split(df, groups=grp_key))

train_df = df.iloc[train_idx].reset_index(drop=True)
testA_df = df.iloc[test_idx].reset_index(drop=True)

Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(args.train_out, index=False, encoding="utf-8-sig")
testA_df.to_csv(args.test_out, index=False, encoding="utf-8-sig")

print(f"[OK] TrainA rows={len(train_df)} | TestA rows={len(testA_df)}")
print(f"[INFO] group_cols={gcols}")
