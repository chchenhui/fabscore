#!/bin/bash
# Evaluate VOCASET checkpoints and aggregate results
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

DATA_DIR=scaffoldswap/data/vocaset/processed
RESULTS_DIR=scaffoldswap/results/vocaset
mkdir -p ${RESULTS_DIR}

CONDITION=${1:-A}

for SEED in 42 123 456; do
    CKPT="scaffoldswap/outputs/vocaset/cond${CONDITION}/seed${SEED}/best_model.pt"
    OUT="${RESULTS_DIR}/cond${CONDITION}_seed${SEED}.json"
    echo "Evaluating Condition ${CONDITION}, seed=${SEED}..."
    python3 -m scaffoldswap.evaluate \
        --checkpoint "${CKPT}" \
        --data_dir "${DATA_DIR}" \
        --output "${OUT}"
    echo ""
done

echo ""
echo "=== Aggregating results ==="
python3 -c "
import json, os, numpy as np

results_dir = '${RESULTS_DIR}'
cond = '${CONDITION}'
seed_results = []
for seed in [42, 123, 456]:
    path = os.path.join(results_dir, f'cond{cond}_seed{seed}.json')
    with open(path) as f:
        seed_results.append(json.load(f))

metrics = {}
for m in ['LVE', 'MVE', 'UFVE', 'FDD']:
    vals = [r[m] for r in seed_results]
    metrics[f'{m}_mean'] = float(np.mean(vals))
    metrics[f'{m}_std'] = float(np.std(vals))
    metrics[f'{m}_per_seed'] = {str(s): v for s, v in zip([42,123,456], vals)}

agg = {
    'condition': cond,
    'dataset': 'vocaset',
    'description': '100 epochs + cosine LR',
    'seeds': [42, 123, 456],
    **metrics,
    'best_epochs': {str(s): r['best_epoch'] for s, r in zip([42,123,456], seed_results)},
    'best_val_losses': {str(s): r['best_val_loss'] for s, r in zip([42,123,456], seed_results)},
}

agg_path = os.path.join(results_dir, f'cond{cond}_results.json')
with open(agg_path, 'w') as f:
    json.dump(agg, f, indent=2)

print(f'Condition {cond}: LVE={metrics[\"LVE_mean\"]:.6f} +/- {metrics[\"LVE_std\"]:.6f}')
print(f'  MVE={metrics[\"MVE_mean\"]:.6f}, UFVE={metrics[\"UFVE_mean\"]:.6f}, FDD={metrics[\"FDD_mean\"]:.6f}')
print()
"

echo "Evaluation complete."
