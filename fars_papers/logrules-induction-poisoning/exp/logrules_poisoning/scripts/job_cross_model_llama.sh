#!/bin/bash
# Job script: launches LLaMA-3-8B-Instruct vLLM server, runs C0/C1/C2 deduction on BGL.
set -e

EXP_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/logrules-induction-poisoning/exp"
source "${EXP_ROOT}/.venv/bin/activate"

echo "Starting vLLM server for LLaMA-3-8B-Instruct..."
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
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

echo "Running cross-model deduction (C0/C1/C2 on BGL with LLaMA-3-8B-Instruct)..."
python "${EXP_ROOT}/logrules_poisoning/scripts/run_cross_model_deduction.py" \
    --api-base http://localhost:8001/v1 \
    --api-key EMPTY \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-concurrency 64

echo "Cross-model experiment complete."
