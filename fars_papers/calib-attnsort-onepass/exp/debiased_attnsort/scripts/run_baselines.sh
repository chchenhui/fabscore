#!/bin/bash
# Run baseline experiments: vanilla generation (no_sorting) and random_reorder sanity check.
# Usage: bash run_baselines.sh --seed 42 --mode no_sorting [--num_examples 200] [--sanity_check]
set -euo pipefail

PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

cd "${PROJ_ROOT}"

python debiased_attnsort/src/eval_pipeline.py "$@"
