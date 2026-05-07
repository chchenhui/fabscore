#!/bin/bash
# Run r_cap ablation for RRCS. Uses dedicated config per r_cap value.
# Usage: bash scripts/run_rcap_ablation.sh <SEED> <R_CAP> [MAX_ITERS]
# R_CAP must be 20 or 40 (r_cap=30 uses run_rrcs.sh).
set -euo pipefail

SEED=${1:?Usage: run_rcap_ablation.sh <SEED> <R_CAP> [MAX_ITERS]}
R_CAP=${2:?Usage: run_rcap_ablation.sh <SEED> <R_CAP> [MAX_ITERS]}
MAX_ITERS=${3:-5000}

EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NANOGPT_DIR="$(cd "$(dirname "$0")/../examples/nanogpt" && pwd)"

source "${EXP_ROOT}/.venv/bin/activate"

set -a
source "${EXP_ROOT}/.env"
set +a

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "${NANOGPT_DIR}"

R_CAP_INT=$(printf "%.0f" "${R_CAP}")
CONFIG="config/train_fineweb10B_mhc_48l_rrcs_r${R_CAP_INT}.py"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

exec python train.py "${CONFIG}" \
    seed="${SEED}" \
    out_dir="results/logs/rrcs_r${R_CAP_INT}_seed${SEED}" \
    wandb_run_name="rrcs_r${R_CAP_INT}_seed${SEED}" \
    wandb_project="${WANDB_PROJECT}" \
    max_iters="${MAX_ITERS}" \
    diag_interval=10
