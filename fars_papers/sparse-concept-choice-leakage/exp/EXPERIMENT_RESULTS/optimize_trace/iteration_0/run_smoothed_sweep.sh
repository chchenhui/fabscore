#!/bin/bash
# Lambda sweep for smoothed covariance mitigation (Condition C).
# Tests lambda in {0.2, 0.5, 0.7, 0.9, 0.95, 0.99} with N=10 releases.
# STS12 eval for representative lambdas (0.2, 0.9, 0.99).
set -e

PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp"
cd "$PROJ_DIR"
source .venv/bin/activate

export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
export WANDB_MODE=offline

SWEEP_RESULTS_DIR="$PROJ_DIR/concept_leakage/results/smoothed_sweep"
mkdir -p "$SWEEP_RESULTS_DIR"

LAMBDAS="0.2 0.5 0.7 0.9 0.95 0.99"
N_RELEASES=10

echo "============================================="
echo "=== Smoothed Covariance Lambda Sweep ==="
echo "=== Lambdas: $LAMBDAS ==="
echo "=== N_releases: $N_RELEASES ==="
echo "============================================="

for LAM in $LAMBDAS; do
    echo ""
    echo "============================================="
    echo "=== Lambda = $LAM ==="
    echo "============================================="

    echo "--- Phase 1: Generate Smoothed Covariances (lambda=$LAM) ---"
    python concept_leakage/noise/smoothed.py --lam $LAM

    echo "--- Phase 2: Smoothed Attack (lambda=$LAM, N=$N_RELEASES) ---"
    python concept_leakage/attack/run_smoothed_attack.py \
        --lam $LAM \
        --n_releases $N_RELEASES \
        --results_dir "$SWEEP_RESULTS_DIR" \
        --fp_subdir "smoothed_lam${LAM}"
done

echo ""
echo "============================================="
echo "=== Phase 3: STS12 Utility for Key Lambdas ==="
echo "============================================="

for LAM in 0.2 0.9 0.99; do
    echo "--- STS12 eval (lambda=$LAM) ---"
    python concept_leakage/evaluation/sts12_smoothed_eval.py \
        --lam $LAM \
        --results_dir "$SWEEP_RESULTS_DIR"
done

echo ""
echo "============================================="
echo "=== Phase 4: Compile Sweep Summary ==="
echo "============================================="

python3 -c "
import json, glob, os
results_dir = '$SWEEP_RESULTS_DIR'
sweep = {'lambdas': {}, 'n_releases': $N_RELEASES, 'experiment': 'smoothed_lambda_sweep'}

for f in sorted(glob.glob(os.path.join(results_dir, 'smoothed_attack_lam*_results.json'))):
    with open(f) as fh:
        d = json.load(fh)
    lam = str(d['smoothing_lambda'])
    sweep['lambdas'][lam] = {
        'accuracy_mean': d['accuracy_mean'],
        'accuracy_std': d['accuracy_std'],
        'macro_f1_mean': d['macro_f1_mean'],
        'macro_f1_std': d['macro_f1_std'],
    }

for f in sorted(glob.glob(os.path.join(results_dir, 'smoothed_sts12_lam*_results.json'))):
    with open(f) as fh:
        d = json.load(fh)
    lam = str(d['smoothing_lambda'])
    if lam in sweep['lambdas']:
        sweep['lambdas'][lam]['sts12_noisy_pearson_mean'] = d['noisy_pearson_mean']
        sweep['lambdas'][lam]['sts12_noisy_pearson_std'] = d['noisy_pearson_std']
        sweep['lambdas'][lam]['sts12_clean_pearson'] = d['clean_pearson']

out = os.path.join(results_dir, 'sweep_summary.json')
with open(out, 'w') as fh:
    json.dump(sweep, fh, indent=2)
print(json.dumps(sweep, indent=2))
print(f'\nSweep summary saved to {out}')
"

echo ""
echo "============================================="
echo "=== Smoothed Sweep Complete ==="
echo "============================================="
