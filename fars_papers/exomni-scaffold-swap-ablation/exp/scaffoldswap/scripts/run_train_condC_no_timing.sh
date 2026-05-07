#!/bin/bash
# Train Condition C w/o timing ablation on BIWI. Usage: bash run_train_condC_no_timing.sh <seed> [debug]
set -e
SEED=${1:?Usage: run_train_condC_no_timing.sh <seed> [debug]}
DEBUG=${2:-""}
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export WANDB_RUN_NAME="condC_no_timing_biwi_seed${SEED}"
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.
if [ "$DEBUG" = "debug" ]; then
    CONFIG="scaffoldswap/configs/biwi_condC_no_timing_debug_v3.yaml"
    echo "DEBUG: Training C w/o timing (10 epochs) seed=${SEED}"
else
    CONFIG="scaffoldswap/configs/biwi_condC_no_timing_v3.yaml"
    echo "Training C w/o timing (600 epochs) seed=${SEED}"
fi
python3 -m scaffoldswap.train --config "${CONFIG}" --seed "${SEED}"
echo "Training complete: C w/o timing, seed=${SEED}"
