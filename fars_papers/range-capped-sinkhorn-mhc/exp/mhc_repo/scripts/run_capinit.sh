#!/bin/bash
# Run fixed tau_cap-init control training for a single seed.
# Usage: bash scripts/run_capinit.sh <SEED> [MAX_ITERS]
set -euo pipefail

SEED=${1:?Usage: run_capinit.sh <SEED> [MAX_ITERS]}
MAX_ITERS=${2:-5000}

EXP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NANOGPT_DIR="$(cd "$(dirname "$0")/../examples/nanogpt" && pwd)"

source "${EXP_ROOT}/.venv/bin/activate"

set -a
source "${EXP_ROOT}/.env"
set +a

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "${NANOGPT_DIR}"

exec python train.py config/train_fineweb10B_mhc_48l_capinit.py \
    seed="${SEED}" \
    out_dir="results/logs/capinit_seed${SEED}" \
    wandb_run_name="capinit_seed${SEED}" \
    wandb_project="${WANDB_PROJECT}" \
    max_iters="${MAX_ITERS}" \
    diag_interval=10
