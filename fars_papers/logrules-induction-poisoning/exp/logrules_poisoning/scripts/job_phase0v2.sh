#!/bin/bash
# Job script: launches vLLM server, runs Phase-0v2 deduction + evaluation, then stops.
# Evaluates new anti-wildcard payloads (D/E/F) at k=1,3,5,7 across all datasets and seeds.
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

echo "Running Phase-0v2 deduction + evaluation (all datasets, all seeds, with test set)..."
python "${EXP_ROOT}/logrules_poisoning/scripts/run_phase0v2_deduction.py" \
    --api-base http://localhost:8001/v1 \
    --api-key EMPTY \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-concurrency 64 \
    --datasets BGL,Linux,HDFS \
    --seeds 42,123,456 \
    --eval-test

echo "Phase-0v2 deduction job complete."
