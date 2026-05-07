#!/bin/bash
# Training script for ScaffoldSwap Condition B on BIWI
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

SEED=${1:-42}
echo "Training Condition B on BIWI with seed=${SEED}"

python3 scaffoldswap/train.py \
    --config scaffoldswap/configs/biwi_condB.yaml \
    --seed ${SEED} \
    --device cuda
