#!/bin/bash
# Run YaRN evaluation for all 3 seeds of a given mode.
# Usage: bash run_yarn_3seeds.sh <mode>
# Example: bash run_yarn_3seeds.sh no_sorting
set -euo pipefail
MODE="${1:?Usage: run_yarn_3seeds.sh <mode>}"
PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"
source "${PROJ_ROOT}/.venv/bin/activate"
set -a
source "${PROJ_ROOT}/.env"
set +a
cd "${PROJ_ROOT}"
export PYTHONUNBUFFERED=1
for SEED in 42 123 456; do
    echo "========== Running mode=${MODE} seed=${SEED} =========="
    python debiased_attnsort/src/eval_yarn.py --mode "${MODE}" --seed "${SEED}" --num_examples 200
    echo "========== Done mode=${MODE} seed=${SEED} =========="
done
echo "All 3 seeds complete for mode=${MODE}"
