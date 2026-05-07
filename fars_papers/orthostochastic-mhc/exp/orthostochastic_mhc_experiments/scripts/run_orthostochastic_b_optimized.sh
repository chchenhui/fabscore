#!/bin/bash
# Launch optimized mHC-Orthostochastic training for Setting B (6-layer, hc_num_streams=8).
# Fixes: identity-like init (code), ns_steps=20, mhc_residual_identity_mix=True (alpha=0.1),
#        eval_interval=250 for better checkpoint selection.
# Usage: bash run_orthostochastic_b_optimized.sh <SEED> [extra overrides...]

set -euo pipefail

SEED=${1:?"Usage: run_orthostochastic_b_optimized.sh <SEED>"}
shift || true

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp"
VENV="${EXP_ROOT}/.venv/bin/activate"
ENV_FILE="${EXP_ROOT}/.env"
TRAIN_DIR="${EXP_ROOT}/mHC-manifold-constrained-hyper-connections/examples/nanogpt"
CONFIG="${EXP_ROOT}/orthostochastic_mhc_experiments/configs/setting_b_orthostochastic.py"
OUT_DIR="${EXP_ROOT}/orthostochastic_mhc_experiments/logs/setting_b_orthostochastic_optimized_seed${SEED}"

source "${VENV}"
set -a; source "${ENV_FILE}"; set +a

cd "${TRAIN_DIR}"

python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    "${CONFIG}" \
    seed="${SEED}" \
    out_dir="${OUT_DIR}" \
    wandb_run_name="setting-b-orthostochastic-opt-seed${SEED}" \
    "$@"
