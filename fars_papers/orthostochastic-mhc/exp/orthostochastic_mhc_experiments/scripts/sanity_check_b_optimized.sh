#!/bin/bash
# Sanity check for optimized Setting B orthostochastic config (1 GPU, few iters).

set -euo pipefail

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp"
VENV="${EXP_ROOT}/.venv/bin/activate"
ENV_FILE="${EXP_ROOT}/.env"
TRAIN_DIR="${EXP_ROOT}/mHC-manifold-constrained-hyper-connections/examples/nanogpt"
CONFIG="${EXP_ROOT}/orthostochastic_mhc_experiments/configs/setting_b_orthostochastic.py"
OUT_DIR="${EXP_ROOT}/orthostochastic_mhc_experiments/logs/sanity_check_b_opt"

source "${VENV}"
set -a; source "${ENV_FILE}"; set +a

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "${TRAIN_DIR}"

python train.py \
    "${CONFIG}" \
    seed=42 \
    out_dir="${OUT_DIR}" \
    max_iters=30 \
    eval_interval=10 \
    log_interval=5 \
    gradient_accumulation_steps=1 \
    wandb_log=False \
    "$@"
