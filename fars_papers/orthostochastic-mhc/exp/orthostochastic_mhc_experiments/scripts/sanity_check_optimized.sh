#!/bin/bash
# Sanity check for optimized orthostochastic config (fixed init + identity_mix + ns_steps=15)
set -euo pipefail

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp"
VENV="${EXP_ROOT}/.venv/bin/activate"
ENV_FILE="${EXP_ROOT}/.env"
TRAIN_DIR="${EXP_ROOT}/mHC-manifold-constrained-hyper-connections/examples/nanogpt"
CONFIG="${EXP_ROOT}/orthostochastic_mhc_experiments/configs/setting_a_orthostochastic.py"
OUT_DIR="${EXP_ROOT}/orthostochastic_mhc_experiments/logs/sanity_check_optimized"

source "${VENV}"
set -a; source "${ENV_FILE}"; set +a

cd "${TRAIN_DIR}"

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

python train.py \
    "${CONFIG}" \
    seed=42 \
    out_dir="${OUT_DIR}" \
    max_iters=50 \
    eval_interval=25 \
    wandb_log=False \
    wandb_run_name="sanity-check-optimized"
