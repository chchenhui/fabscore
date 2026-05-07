#!/bin/bash
# Debug run for Condition B: 2 epochs, 1 seed, verify pipeline
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

echo "Debug run: Condition B on BIWI, 2 epochs, seed=42"

python3 scaffoldswap/train.py \
    --config scaffoldswap/configs/biwi_condB_debug.yaml \
    --seed 42 \
    --device cuda

echo "Debug run complete!"
