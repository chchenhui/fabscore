#!/bin/bash
# Run condition A or B on 1,500-example ConFiQA-MC subset.
# Usage: bash run_subset_ab.sh --condition A|B

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

source .venv/bin/activate
set -a; source .env; set +a

python eacp/scripts/run_inference.py \
    --subset_indices eacp/data/confiqa_mc_1500_subset_indices.json \
    "$@"
