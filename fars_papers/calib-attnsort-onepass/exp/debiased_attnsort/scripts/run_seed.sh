#!/bin/bash
# Run both no_sorting and random_reorder for a single seed.
# Usage: bash run_seed.sh <seed> [num_examples]
set -euo pipefail

SEED=${1:?Usage: run_seed.sh <seed> [num_examples]}
NUM_EXAMPLES=${2:-200}

PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"

source "${PROJ_ROOT}/.venv/bin/activate"

set -a
source "${PROJ_ROOT}/.env"
set +a

cd "${PROJ_ROOT}"

echo "=== Running no_sorting seed=${SEED} n=${NUM_EXAMPLES} ==="
python debiased_attnsort/src/eval_pipeline.py --mode no_sorting --seed "${SEED}" --num_examples "${NUM_EXAMPLES}"

echo "=== Running random_reorder seed=${SEED} n=${NUM_EXAMPLES} ==="
python debiased_attnsort/src/eval_pipeline.py --mode random_reorder --seed "${SEED}" --num_examples "${NUM_EXAMPLES}"

echo "=== All done for seed=${SEED} ==="
