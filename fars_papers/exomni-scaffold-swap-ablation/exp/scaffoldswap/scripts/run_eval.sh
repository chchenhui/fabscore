#!/bin/bash
# Evaluation script for ScaffoldSwap
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate

export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export PYTHONPATH="${PYTHONPATH}:."
export NUMBA_CACHE_DIR=/tmp/numba_cache

SEED=${1:-42}
CKPT="scaffoldswap/outputs/biwi/condA/seed${SEED}/best_model.pt"
DATA_DIR="scaffoldswap/data/biwi/processed"
OUT="scaffoldswap/results/biwi/condA_seed${SEED}.json"

echo "Evaluating seed=${SEED}, checkpoint=${CKPT}"

python3 scaffoldswap/evaluate.py \
    --checkpoint ${CKPT} \
    --data_dir ${DATA_DIR} \
    --output ${OUT} \
    --device cuda
