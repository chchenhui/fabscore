#!/bin/bash
# Sanity check for unconstrained HC Setting A: 50 iters, 1 GPU
set -euo pipefail

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp"
VENV="${EXP_ROOT}/.venv/bin/activate"
ENV_FILE="${EXP_ROOT}/.env"
TRAIN_DIR="${EXP_ROOT}/mHC-manifold-constrained-hyper-connections/examples/nanogpt"
CONFIG="${EXP_ROOT}/orthostochastic_mhc_experiments/configs/setting_a_hc_unconstrained.py"
OUT_DIR="${EXP_ROOT}/orthostochastic_mhc_experiments/logs/sanity_hc_unconstrained"

source "${VENV}"
set -a; source "${ENV_FILE}"; set +a

cd "${TRAIN_DIR}"

python -m torch.distributed.run --standalone --nproc_per_node=1 train.py \
    "${CONFIG}" \
    seed=1 \
    out_dir="${OUT_DIR}" \
    max_iters=50 \
    eval_interval=25 \
    wandb_log=False \
    max_train_shards=5
