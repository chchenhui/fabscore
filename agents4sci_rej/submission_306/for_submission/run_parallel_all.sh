#!/usr/bin/env bash
set -euo pipefail

INPUT="cohorts/evidence_ready.csv"   
JOBS=48                              
SHARD_SIZE=200                       
N_PERM=1024                          
OTHER_ARGS="--null-mode rotation --k auto --min-block-len 30 --topN 100 --alpha 0.05 --sym-mode mean --cath"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "$INPUT" ]; then
  echo "ERROR: $INPUT not found" >&2; exit 1
fi

REPO_ROOT="$(pwd)"
SHARDS_DIR="cohorts/shards"
RUNS_DIR="runs_parallel"
mkdir -p "$SHARDS_DIR" "$RUNS_DIR"

python - <<'PY'
import pandas as pd, os, math, sys
INPUT = os.environ.get("INPUT")
SHARDS_DIR = os.environ.get("SHARDS_DIR")
SZ = int(os.environ.get("SHARD_SIZE","200"))

df = pd.read_csv(INPUT)
id_col = None
for c in ["id","uniprot_id","uid","entry","accession"]:
    if c in df.columns:
        id_col = c; break
if id_col is None:
    
    for c in df.columns:
        if "id" in c.lower():
            id_col = c; break
if id_col is None:
    print("ERROR: ID column not found in INPUT (expected one of id/uniprot_id/uid/entry/accession)", file=sys.stderr)
    sys.exit(2)

n = len(df)
if n == 0:
    print("ERROR: INPUT is empty", file=sys.stderr); sys.exit(2)

os.makedirs(SHARDS_DIR, exist_ok=True)
num_shards = math.ceil(n / SZ)
for i in range(num_shards):
    df.iloc[i*SZ:(i+1)*SZ].to_csv(os.path.join(SHARDS_DIR, f"ids_{i:03d}.csv"), index=False)
print(f"[split] rows={n} shards={num_shards} id_col={id_col}")
PY

find "$SHARDS_DIR" -name 'ids_*.csv' | sort | \
xargs -n1 -P "$JOBS" -I{} bash -lc '
set -euo pipefail
shard=$(basename {} .csv)
run_dir="'$RUNS_DIR'"/$shard
mkdir -p "$run_dir"
cd "$run_dir"
echo "[start] $shard -> $run_dir"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHONPATH="'$REPO_ROOT'" python -m bcrparts all \
  --ids-file "'$REPO_ROOT'"/{} \
  --n-perm "'$N_PERM'" \
  '"$OTHER_ARGS"' \
  1> stdout.log 2> stderr.log

if [ -f results/bcrparts/mechspec.csv ]; then cp -f results/bcrparts/mechspec.csv mechspec.csv; fi
if [ -f results/bcrparts/topN.csv ]; then cp -f results/bcrparts/topN.csv topN.csv; fi

echo "[done]  $shard"
'

echo "[all parallel jobs finished]"
