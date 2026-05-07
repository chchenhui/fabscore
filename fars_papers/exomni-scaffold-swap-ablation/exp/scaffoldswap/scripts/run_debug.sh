#!/bin/bash
# Debug training: 5 epochs to verify GPU pipeline end-to-end
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

echo "=== Debug training: 5 epochs, seed=42 ==="

# Create a temporary debug config with 5 epochs
python3 -c "
import yaml
with open('scaffoldswap/configs/biwi_condA.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['training']['epochs'] = 5
cfg['training']['eval_every_epochs'] = 1
cfg['output_dir'] = 'scaffoldswap/outputs/biwi/condA_debug'
with open('scaffoldswap/configs/biwi_condA_debug.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Debug config created')
"

python3 scaffoldswap/train.py \
    --config scaffoldswap/configs/biwi_condA_debug.yaml \
    --seed 42 \
    --device cuda

echo "=== Debug evaluation ==="
python3 scaffoldswap/evaluate.py \
    --checkpoint scaffoldswap/outputs/biwi/condA_debug/seed42/best_model.pt \
    --data_dir scaffoldswap/data/biwi/processed \
    --output scaffoldswap/results/biwi/condA_debug.json \
    --device cuda

echo "=== Debug complete ==="
cat scaffoldswap/results/biwi/condA_debug.json
