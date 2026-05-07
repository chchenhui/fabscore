#!/bin/bash
# Train v3 (600 epochs, AdamW, warmup): pass DATASET, CONDITION, SEED as args
# Usage: bash run_train_v3.sh <biwi|vocaset> <A|B|C> <seed>
set -e
DATASET=${1:?Usage: run_train_v3.sh <biwi|vocaset> <A|B|C> <seed>}
CONDITION=${2:?Usage: run_train_v3.sh <biwi|vocaset> <A|B|C> <seed>}
SEED=${3:?Usage: run_train_v3.sh <biwi|vocaset> <A|B|C> <seed>}
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.
CONFIG="scaffoldswap/configs/${DATASET}_cond${CONDITION}_v3.yaml"
echo "Training v3: ${DATASET} Condition ${CONDITION} seed=${SEED} (600 epochs, AdamW, warmup)"
python3 -m scaffoldswap.train --config "${CONFIG}" --seed "${SEED}"
echo "Training complete: ${DATASET} Condition ${CONDITION}, seed=${SEED}"
