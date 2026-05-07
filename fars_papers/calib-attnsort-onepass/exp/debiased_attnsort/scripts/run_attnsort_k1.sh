#!/bin/bash
# Run attention sorting k=1 (uncalibrated) for a single seed.
# Usage: bash run_attnsort_k1.sh --seed 42 [--num_examples 200] [--sanity_check]
set -euo pipefail

PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

cd "${PROJ_ROOT}"

python debiased_attnsort/src/eval_pipeline.py --mode attnsort_k1 "$@"
