#!/bin/bash
# Job script: launches vLLM server, runs R_safe deduction for C2 defense, then stops.
# Parses canary + test sets with R_safe rules for all 9 (dataset, seed) combos.
set -e

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/logrules-induction-poisoning/exp"
source "${EXP_ROOT}/.venv/bin/activate"

echo "Starting vLLM server..."
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo "Running R_safe deduction for C2 defense..."
python "${EXP_ROOT}/logrules_poisoning/scripts/run_c2_rsafe_deduction.py" \
    --api-base http://localhost:8001/v1 \
    --api-key EMPTY \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-concurrency 64 \
    --datasets BGL,Linux,HDFS \
    --seeds 42,123,456

echo "C2 R_safe deduction job complete."
