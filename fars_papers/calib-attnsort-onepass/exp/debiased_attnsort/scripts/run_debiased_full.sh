#!/bin/bash
# Run debiased_k1 for a single seed with 200 examples (full run).
# Usage: bash run_debiased_full.sh <seed>
set -euo pipefail

SEED="${1:?Usage: bash run_debiased_full.sh <seed>}"
PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

export WANDB_MODE=offline

cd "${PROJ_ROOT}"

python debiased_attnsort/src/eval_pipeline.py --mode debiased_k1 --seed "${SEED}" --num_examples 200
