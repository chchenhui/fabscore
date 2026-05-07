#!/bin/bash
# Full job: best-of-5 parsing on all datasets and seeds.
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

echo "Running best-of-5 full experiment..."
python "${EXP_ROOT}/logrules_poisoning/scripts/run_best_of_5.py" \
    --api-base http://localhost:8001/v1 \
    --api-key EMPTY \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-concurrency 64

echo "Full experiment complete."
