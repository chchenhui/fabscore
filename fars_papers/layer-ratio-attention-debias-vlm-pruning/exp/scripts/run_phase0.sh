#!/bin/bash
# Phase-0 diagnostic: extract attention from InternVL2.5-8B and compute metrics.
# Usage: bash scripts/run_phase0.sh [--debug]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASE_DIR"
source .venv/bin/activate

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

python -u analysis/phase0_diagnostic.py "$@" 2>&1
