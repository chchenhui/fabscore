#!/bin/bash
# Debug training: Condition A with temporal shuffle, 5 epochs, seed=42
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export WANDB_RUN_NAME=condA_shuffled_biwi_debug_seed42
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

echo "=== Debug: Condition A Shuffled on BIWI, 5 epochs, seed=42 ==="
python3 -m scaffoldswap.train \
    --config scaffoldswap/configs/biwi_condA_shuffled_debug_v3.yaml \
    --seed 42 \
    --shuffle_temporal

echo "=== Debug complete ==="
