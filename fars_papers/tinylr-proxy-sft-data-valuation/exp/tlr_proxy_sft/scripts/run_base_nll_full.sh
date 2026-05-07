#!/bin/bash
# Launch NLL computation for all 12 datasets across 4 GPUs.
# Each GPU runs a worker that processes 3 datasets sequentially.
set -euo pipefail

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp
SCRIPT=${BASE}/tlr_proxy_sft/analysis/compute_base_nll.py
DATA_DIR=${BASE}/tlr_proxy_sft/data/processed
OUT_DIR=${BASE}/tlr_proxy_sft/results/base_nll_parts

DATASETS=(
    "AM-Thinking-v1-Distilled-math"
    "DeepMath-309K"
    "Magpie-Reasoning-V2-250K-CoT-QwQ-math"
    "Maths-College"
    "OpenR1-Math"
    "QwQ-LongCoT-130K-math"
    "R1-Distill-SFT-math"
    "hkust-nlp__dart-math-hard"
    "mathplus"
    "numinamath-cot"
    "numinamath1_5"
    "openmathinstruct-2"
)

NUM_GPUS=4
mkdir -p ${OUT_DIR}

run_worker() {
    local GPU_ID=$1
    shift
    local DS_LIST=("$@")
    for DS in "${DS_LIST[@]}"; do
        echo "[GPU ${GPU_ID}] Starting ${DS}..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} run \
            --model Qwen/Qwen2.5-1.5B \
            --data-dir ${DATA_DIR} \
            --output-dir ${OUT_DIR} \
            --dataset "${DS}" \
            --gpu-id 0 \
            --cutoff-len 4096 \
            --batch-size 8
        echo "[GPU ${GPU_ID}] Finished ${DS}"
    done
}

PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    worker_datasets=()
    for i in "${!DATASETS[@]}"; do
        if [ $((i % NUM_GPUS)) -eq ${gpu_id} ]; then
            worker_datasets+=("${DATASETS[$i]}")
        fi
    done
    echo "GPU ${gpu_id} assigned: ${worker_datasets[*]}"
    run_worker ${gpu_id} "${worker_datasets[@]}" &
    PIDS+=($!)
done

echo "Waiting for all ${#PIDS[@]} GPU workers..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait ${pid} || FAIL=$((FAIL+1))
done

if [ ${FAIL} -gt 0 ]; then
    echo "ERROR: ${FAIL} workers failed"
    exit 1
fi

echo "All workers completed. Merging results..."
python ${SCRIPT} merge \
    --output-dir ${OUT_DIR} \
    --output-csv ${BASE}/tlr_proxy_sft/results/base_nll_scores.csv

echo "Done!"
