#!/bin/bash
# LAVE v3: Grammar fix (bare identifiers) + irrelevance bypass + reduced retry.
# Usage: bash run_lave_cfg_v3.sh <seed> [--limit N]

set -e

SEED=${1:?Usage: bash run_lave_cfg_v3.sh <seed> [--limit N]}
shift

EXP_ROOT=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/lave-tool-calling-bfcl/exp
source ${EXP_ROOT}/.venv/bin/activate

set -a
source ${EXP_ROOT}/.env
set +a

export PYTHONPATH=${EXP_ROOT}/CD4dLLM:${EXP_ROOT}:${PYTHONPATH}

echo "=== LAVE v3 CFG-Constrained Inference ==="
echo "Seed: ${SEED}"
echo "Fixes: extended grammar (bare identifiers), irrelevance bypass, max_retry=10"
echo "max_tokens=512, steps=256, top_k=10, top_n_beam=10, random_n_beam=10"
echo "Extra args: $@"
echo "==============================="

python ${EXP_ROOT}/bfcl_cfg_diffusion/inference/run_lave_cfg.py \
    --seed ${SEED} \
    --max_tokens 512 \
    --steps 256 \
    --top_k_per_mask 10 \
    --top_n_beam 10 \
    --random_n_beam 10 \
    --max_retry_num_total 10 \
    --skip_lave_categories irrelevance \
    --output_dir ${EXP_ROOT}/bfcl_cfg_diffusion/results/lave_cfg_v3 \
    "$@"
