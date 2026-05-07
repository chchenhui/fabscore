#!/bin/bash
# Launch ice-shelf inverse problem experiments.
# Usage: bash overlap_lbfgs_pinn/scripts/run_ice_shelf.sh [budget] [seeds]
# Run from the project root (exp/).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

BUDGET="${1:-30000}"
SEEDS="${2:-0,1,2}"

echo "Running Adam+Resampling baseline on ice-shelf"
echo "Budget: ${BUDGET}, Seeds: ${SEEDS}"

python -m overlap_lbfgs_pinn.scripts.run_adam_resampling_ice_shelf \
    --budget "${BUDGET}" \
    --seeds "${SEEDS}"
