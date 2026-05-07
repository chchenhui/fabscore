#!/bin/bash
# Full training: Condition A with temporal shuffle, 600 epochs, seed=42
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export WANDB_RUN_NAME=condA_shuffled_biwi_seed42
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

echo "=== Training: Condition A Shuffled on BIWI, 600 epochs, seed=42 ==="
python3 -m scaffoldswap.train \
    --config scaffoldswap/configs/biwi_condA_shuffled_v3.yaml \
    --seed 42 \
    --shuffle_temporal

echo "=== Evaluation ==="
python3 -m scaffoldswap.evaluate \
    --checkpoint scaffoldswap/outputs/biwi/condA_shuffled/seed42/best_model.pt \
    --data_dir scaffoldswap/data/biwi/processed \
    --output scaffoldswap/results/biwi/condA_shuffled_seed42.json

echo "=== Training and evaluation complete ==="
