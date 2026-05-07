#!/bin/bash
# Run optimized debiased k=1 for a single seed.
# Changes from original: alpha=0.01, num_bins=30, p75 quantile aggregation,
# quantile-based binning, additive debiasing.
# Usage: bash run_optimized_debiased.sh <seed>
set -euo pipefail

SEED="${1:?Usage: bash run_optimized_debiased.sh <seed>}"
PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

cd "${PROJ_ROOT}"

python debiased_attnsort/src/eval_pipeline.py --mode debiased_k1 --seed "${SEED}" --num_examples 200
