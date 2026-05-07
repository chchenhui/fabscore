#!/bin/bash
# Train encryptor with NO diversity loss (lambda2=0) for ablation study.
# Uses task-specified hyperparams: lambda1=1.0, lr=1e-4, max_steps=3000.
# Usage: bash run_train_no_div.sh <seed>
set -e
PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/key-search-bypasses-encrypted-activation-monitors/exp"
source "${PROJ_DIR}/.venv/bin/activate"
set -a && source "${PROJ_DIR}/.env" && set +a
export WANDB_MODE=offline
cd "${PROJ_DIR}"

SEED=${1:-42}
MAX_STEPS=${2:-3000}

export WANDB_NAME="encryptor_no_div_s${SEED}"

python -u -m key_search_bypass.encryptor.train \
    --seed ${SEED} \
    --output_dir "key_search_bypass/outputs/encryptor_no_div/seed_${SEED}" \
    --max_steps ${MAX_STEPS} \
    --lr 1e-4 \
    --batch_size 8 \
    --grad_accum 4 \
    --warmup_steps 500 \
    --lambda1 1.0 \
    --lambda2 0.0 \
    --tau_low 0.005 \
    --tau_high 0.05 \
    --curriculum_warmup 1000 \
    --train_size 20000 \
    --eval_samples 500 \
    --eval_steps 100 \
    2>&1
