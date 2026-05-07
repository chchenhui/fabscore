#!/bin/bash
# Train encryptor with high diversity for stronger key-search attacks.
# Usage: bash run_train_opt_diverse.sh <seed>
set -e
PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/key-search-bypasses-encrypted-activation-monitors/exp"
source "${PROJ_DIR}/.venv/bin/activate"
set -a && source "${PROJ_DIR}/.env" && set +a
export WANDB_MODE=offline
cd "${PROJ_DIR}"

SEED=${1:-42}

python -u -m key_search_bypass.encryptor.train \
    --seed ${SEED} \
    --output_dir "key_search_bypass/outputs/encryptor/seed_${SEED}_diverse" \
    --max_steps 5000 \
    --lr 3e-5 \
    --batch_size 8 \
    --grad_accum 4 \
    --warmup_steps 1000 \
    --lambda1 0.2 \
    --lambda2 0.5 \
    --tau_low 0.005 \
    --tau_high 0.05 \
    --curriculum_warmup 1500 \
    --train_size 20000 \
    --eval_samples 500 \
    --eval_steps 100 \
    2>&1
