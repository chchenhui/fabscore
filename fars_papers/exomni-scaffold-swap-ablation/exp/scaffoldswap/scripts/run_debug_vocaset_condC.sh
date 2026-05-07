#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

echo "=== Debug: VOCASET Condition C (10 epochs) ==="
python3 -m scaffoldswap.train --config scaffoldswap/configs/vocaset_condC_debug.yaml --seed 42
echo "=== Debug complete ==="
