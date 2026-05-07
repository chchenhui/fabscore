#!/bin/bash
# Sanity check for optimized encryptor training (300 steps, seed 42).
set -e
PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/key-search-bypasses-encrypted-activation-monitors/exp"
source "${PROJ_DIR}/.venv/bin/activate"
set -a && source "${PROJ_DIR}/.env" && set +a
cd "${PROJ_DIR}"
python -u -m key_search_bypass.encryptor.train \
    --seed 42 \
    --max_steps 300 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --max_length 128 \
    --train_size 20000 \
    --val_size 1000 \
    --eval_steps 50 \
    --eval_samples 500 \
    --key_dim 128 \
    --lambda1 0.05 \
    --lambda2 0.02 \
    --tau_low 0.003 \
    --tau_high 0.03 \
    --curriculum_warmup 200 \
    --output_dir "${PROJ_DIR}/key_search_bypass/outputs/encryptor/seed_42_opt_sanity" \
    2>&1
