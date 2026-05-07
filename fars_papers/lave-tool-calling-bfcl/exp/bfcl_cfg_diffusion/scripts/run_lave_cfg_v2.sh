#!/bin/bash
# LAVE v2: original beam width, medium retry budget (15), original gen_length,
# with truncation post-processing in run_lave_cfg.py.
# Usage: bash run_lave_cfg_v2.sh <seed> [--limit N]

set -e

SEED=${1:?Usage: bash run_lave_cfg_v2.sh <seed> [--limit N]}
shift

EXP_ROOT=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/lave-tool-calling-bfcl/exp
source ${EXP_ROOT}/.venv/bin/activate

set -a
source ${EXP_ROOT}/.env
set +a

export PYTHONPATH=${EXP_ROOT}/CD4dLLM:${EXP_ROOT}:${PYTHONPATH}

echo "=== LAVE v2 Inference ==="
echo "Seed: ${SEED}"
echo "max_tokens=256, steps=128, top_k=5, top_n_beam=5, random_n_beam=5, max_retry=15"
echo "Extra args: $@"
echo "==============================="

python ${EXP_ROOT}/bfcl_cfg_diffusion/inference/run_lave_cfg.py \
    --seed ${SEED} \
    --max_tokens 256 \
    --steps 128 \
    --top_k_per_mask 5 \
    --top_n_beam 5 \
    --random_n_beam 5 \
    --max_retry_num_total 15 \
    --output_dir ${EXP_ROOT}/bfcl_cfg_diffusion/results/lave_cfg_v2 \
    "$@"
