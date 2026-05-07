#!/bin/bash
# Train on VOCASET v2 (300 epochs): pass CONDITION and SEED as args
# Usage: bash run_train_vocaset_v2.sh <A|B|C> <seed>
set -e
CONDITION=${1:?Usage: run_train_vocaset_v2.sh <A|B|C> <seed>}
SEED=${2:?Usage: run_train_vocaset_v2.sh <A|B|C> <seed>}

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.

CONFIG="scaffoldswap/configs/vocaset_cond${CONDITION}.yaml"
echo "Training Condition ${CONDITION} on VOCASET v2 with seed=${SEED} (300 epochs + cosine LR)"
python3 -m scaffoldswap.train --config "${CONFIG}" --seed "${SEED}"
echo "Training complete: Condition ${CONDITION}, seed=${SEED}"
