#!/bin/bash
# Evaluate all 3 seeds + aggregate results
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate

export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export PYTHONPATH="${PYTHONPATH}:."
export NUMBA_CACHE_DIR=/tmp/numba_cache

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

mkdir -p scaffoldswap/results/biwi

for SEED in 42 123 456; do
    echo "=== Evaluating seed=${SEED} ==="
    python3 scaffoldswap/evaluate.py \
        --checkpoint scaffoldswap/outputs/biwi/condA/seed${SEED}/best_model.pt \
        --data_dir scaffoldswap/data/biwi/processed \
        --output scaffoldswap/results/biwi/condA_seed${SEED}.json \
        --device cuda
done

echo "=== Aggregating results ==="
python3 -c "
import json, numpy as np

results = []
for seed in [42, 123, 456]:
    with open(f'scaffoldswap/results/biwi/condA_seed{seed}.json') as f:
        results.append(json.load(f))

metrics = ['LVE', 'MVE', 'UFVE', 'FDD']
agg = {'condition': 'A', 'dataset': 'biwi', 'seeds': [42, 123, 456]}
for m in metrics:
    vals = [r[m] for r in results]
    agg[f'{m}_mean'] = float(np.mean(vals))
    agg[f'{m}_std'] = float(np.std(vals))
    agg[f'{m}_per_seed'] = {str(r['seed']): r[m] for r in results}

agg['best_epochs'] = {str(r['seed']): r['best_epoch'] for r in results}
agg['best_val_losses'] = {str(r['seed']): r['best_val_loss'] for r in results}

with open('scaffoldswap/results/biwi/condA_results.json', 'w') as f:
    json.dump(agg, f, indent=2)

print('\\n=== Condition A (SSL + Prosody) on BIWI ===')
for m in metrics:
    print(f'  {m}: {agg[f\"{m}_mean\"]:.6f} +/- {agg[f\"{m}_std\"]:.6f}')
print(f'  Best epochs: {agg[\"best_epochs\"]}')
"

echo "=== Done ==="
