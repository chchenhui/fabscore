#!/bin/bash
# Debug run for VOCASET: 5 epochs to verify pipeline
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

echo "=== Debug run: Condition A on VOCASET ==="
python3 -m scaffoldswap.train --config scaffoldswap/configs/vocaset_condA_debug.yaml --seed 42
echo "Debug training done"

echo ""
echo "=== Debug eval ==="
python3 -m scaffoldswap.evaluate \
    --checkpoint scaffoldswap/outputs/vocaset/condA_debug/seed42/best_model.pt \
    --data_dir scaffoldswap/data/vocaset/processed \
    --output scaffoldswap/outputs/vocaset/condA_debug/eval_result.json

echo "Debug complete!"
