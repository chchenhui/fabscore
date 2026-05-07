#!/bin/bash
# Task 7: Lambda sweep for {0.1, 0.5} with N=2 releases.
# Attack (template matching) + STS12 utility evaluation.
# Parameters: G=50, M=200, N=2, seeds=[42,123,456], eps=10.0, K=5, mask_seed=42.
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

SWEEP_DIR="$PROJ_DIR/concept_leakage/results/lambda_sweep"
mkdir -p "$SWEEP_DIR"

echo "============================================="
echo "=== Task 7: Lambda Sweep (N=2) ==="
echo "=== Lambdas: 0.1 0.5 ==="
echo "============================================="

echo "--- Phase 1: Generate smoothed covariance (lambda=0.1) ---"
python concept_leakage/noise/smoothed.py --lam 0.1

echo ""
echo "--- Phase 2a: Attack lambda=0.1, N=2 ---"
python concept_leakage/attack/run_smoothed_attack.py \
    --lam 0.1 \
    --n_releases 2 \
    --results_dir "$SWEEP_DIR" \
    --fp_subdir "smoothed_lam0.10_n2"

echo ""
echo "--- Phase 2b: Attack lambda=0.5, N=2 ---"
python concept_leakage/attack/run_smoothed_attack.py \
    --lam 0.5 \
    --n_releases 2 \
    --results_dir "$SWEEP_DIR" \
    --fp_subdir "smoothed_lam0.50_n2"

echo ""
echo "--- Phase 3a: STS12 eval lambda=0.1 ---"
python concept_leakage/evaluation/sts12_smoothed_eval.py \
    --lam 0.1 \
    --results_dir "$SWEEP_DIR"

echo ""
echo "--- Phase 3b: STS12 eval lambda=0.5 ---"
python concept_leakage/evaluation/sts12_smoothed_eval.py \
    --lam 0.5 \
    --results_dir "$SWEEP_DIR"

echo ""
echo "============================================="
echo "=== Verification: Check all result files ==="
echo "============================================="

python3 -c "
import json, os, sys

sweep_dir = '$SWEEP_DIR'
files = [
    'smoothed_attack_lam0.10_results.json',
    'smoothed_attack_lam0.50_results.json',
    'smoothed_sts12_lam0.10_results.json',
    'smoothed_sts12_lam0.50_results.json',
]
all_ok = True
for f in files:
    p = os.path.join(sweep_dir, f)
    if not os.path.exists(p):
        print(f'MISSING: {p}')
        all_ok = False
        continue
    with open(p) as fh:
        d = json.load(fh)
    if 'attack' in f:
        print(f'{f}: acc={d[\"accuracy_mean\"]:.4f}+/-{d[\"accuracy_std\"]:.4f}, f1={d[\"macro_f1_mean\"]:.4f}+/-{d[\"macro_f1_std\"]:.4f}')
    else:
        print(f'{f}: pearson={d[\"noisy_pearson_mean\"]:.4f}+/-{d[\"noisy_pearson_std\"]:.4f}')

if not all_ok:
    print('ERROR: Some result files are missing!')
    sys.exit(1)
print('\nAll 4 result files verified.')
"

echo ""
echo "=== Task 7 Lambda Sweep Complete ==="
