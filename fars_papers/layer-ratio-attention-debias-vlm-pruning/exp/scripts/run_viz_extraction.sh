#!/bin/bash
# Extract attention data for visualization examples.
# Usage: run_viz_extraction.sh [max_examples]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

source "${BASE_DIR}/.venv/bin/activate"
cd "${BASE_DIR}"

MAX_EXAMPLES="${1:--1}"

python analysis/extract_attention_data.py \
    --max-examples "${MAX_EXAMPLES}" \
    2>&1
