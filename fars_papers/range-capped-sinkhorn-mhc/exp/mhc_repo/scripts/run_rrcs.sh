#!/bin/bash
# Run RRCS (Range-Capped Sinkhorn) training for a single seed.
# Usage: bash scripts/run_rrcs.sh <SEED> [MAX_ITERS]
set -euo pipefail

SEED=${1:?Usage: run_rrcs.sh <SEED> [MAX_ITERS]}
MAX_ITERS=${2:-5000}

EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NANOGPT_DIR="$(cd "$(dirname "$0")/../examples/nanogpt" && pwd)"

source "${EXP_ROOT}/.venv/bin/activate"

set -a
source "${EXP_ROOT}/.env"
set +a

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "${NANOGPT_DIR}"

exec python train.py config/train_fineweb10B_mhc_48l_rrcs.py \
    seed="${SEED}" \
    out_dir="results/logs/rrcs_seed${SEED}" \
    wandb_run_name="rrcs_seed${SEED}" \
    wandb_project="${WANDB_PROJECT}" \
    max_iters="${MAX_ITERS}" \
    diag_interval=10
