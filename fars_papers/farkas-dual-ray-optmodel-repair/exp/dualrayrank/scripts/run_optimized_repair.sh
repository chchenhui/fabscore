#!/bin/bash
# Optimized iterative best-of-N repair with configurable model.
# Usage: bash run_optimized_repair.sh <model> <tp_size> <gpu_mem> <n> <rounds> <tag>
# Example: bash run_optimized_repair.sh Qwen/Qwen2.5-72B-Instruct 4 0.92 16 2 72b
set -e

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
TP_SIZE="${2:-1}"
GPU_MEM="${3:-0.92}"
N_SAMPLES="${4:-16}"
ROUNDS="${5:-2}"
TAG="${6:-}"

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/farkas-dual-ray-optmodel-repair/exp
cd "$PROJ_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"
set -a
source .env
set +a
export WANDB_MODE=offline

echo "Model: $MODEL"
echo "TP size: $TP_SIZE"
echo "GPU mem: $GPU_MEM"
echo "N samples: $N_SAMPLES"
echo "Rounds: $ROUNDS"
echo "Tag: $TAG"

python -m dualrayrank.scripts.run_optimized_repair \
    --model "$MODEL" \
    --attempt0-dir dualrayrank/outputs/attempt0 \
    --output-dir dualrayrank/outputs/repair_optimized \
    --results-dir dualrayrank/results \
    --iis-results dualrayrank/results/iis_topk_results.json \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_MEM" \
    --k 10 \
    --n "$N_SAMPLES" \
    --rounds "$ROUNDS" \
    --temperature 0.7 \
    --seed 42 \
    --tag "$TAG"

echo ""
echo "=== Optimized repair experiment complete ==="
