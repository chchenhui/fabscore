#!/bin/bash
# Optimized encryptor v2: balanced privacy/utility with eps=0.1, margin=0.5,
# moderate lambdas (0.3/0.1), lower LR (3e-5), longer training (5000 steps).
set -e
PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/key-search-bypasses-encrypted-activation-monitors/exp"
source "${PROJ_DIR}/.venv/bin/activate"
set -a && source "${PROJ_DIR}/.env" && set +a
cd "${PROJ_DIR}"

SEED=${1:-42}

python -u -m key_search_bypass.encryptor.train \
    --seed ${SEED} \
    --max_steps 5000 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --max_length 128 \
    --train_size 20000 \
    --val_size 1000 \
    --eval_steps 100 \
    --eval_samples 500 \
    --key_dim 128 \
    --lambda1 0.3 \
    --lambda2 0.1 \
    --tau_low 0.003 \
    --tau_high 0.03 \
    --curriculum_warmup 1500 \
    --output_dir "${PROJ_DIR}/key_search_bypass/outputs/encryptor/seed_${SEED}_opt2" \
    2>&1
