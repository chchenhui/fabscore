#!/bin/bash
# Run optimized RRCS with configurable r_cap.
# Usage: bash scripts/run_rrcs_opt.sh <SEED> <R_CAP> [MAX_ITERS]
set -euo pipefail

SEED=${1:?Usage: run_rrcs_opt.sh <SEED> <R_CAP> [MAX_ITERS]}
R_CAP=${2:?Usage: run_rrcs_opt.sh <SEED> <R_CAP> [MAX_ITERS]}
MAX_ITERS=${3:-5000}

EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NANOGPT_DIR="$(cd "$(dirname "$0")/../examples/nanogpt" && pwd)"

source "${EXP_ROOT}/.venv/bin/activate"

set -a
source "${EXP_ROOT}/.env"
set +a

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "${NANOGPT_DIR}"

R_CAP_TAG=$(echo "${R_CAP}" | tr '.' 'p')

exec python train.py config/train_fineweb10B_mhc_48l_rrcs_opt.py \
    seed="${SEED}" \
    mhc_r_cap="${R_CAP}" \
    out_dir="results/logs/rrcs_opt_rcap${R_CAP_TAG}_seed${SEED}" \
    wandb_run_name="rrcs_opt_rcap${R_CAP_TAG}_seed${SEED}" \
    wandb_project="${WANDB_PROJECT}" \
    max_iters="${MAX_ITERS}" \
    diag_interval=10
