#!/bin/bash
# Optimized LAVE CFG-constrained diffusion inference with increased retry budget,
# wider beam search, and longer generation length.
# Usage: bash run_lave_cfg_optimized.sh <seed> [--limit N]

set -e

SEED=${1:?Usage: bash run_lave_cfg_optimized.sh <seed> [--limit N]}
shift

EXP_ROOT=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/lave-tool-calling-bfcl/exp
source ${EXP_ROOT}/.venv/bin/activate

set -a
source ${EXP_ROOT}/.env
set +a

export PYTHONPATH=${EXP_ROOT}/CD4dLLM:${EXP_ROOT}:${PYTHONPATH}

echo "=== Optimized LAVE CFG-Constrained Inference ==="
echo "Seed: ${SEED}"
echo "max_tokens=512, steps=256, top_k=10, top_n_beam=10, random_n_beam=10, max_retry=30"
echo "Extra args: $@"
echo "==============================="

python ${EXP_ROOT}/bfcl_cfg_diffusion/inference/run_lave_cfg.py \
    --seed ${SEED} \
    --max_tokens 512 \
    --steps 256 \
    --top_k_per_mask 10 \
    --top_n_beam 10 \
    --random_n_beam 10 \
    --max_retry_num_total 30 \
    --output_dir ${EXP_ROOT}/bfcl_cfg_diffusion/results/lave_cfg_optimized \
    "$@"
