#!/bin/bash
# Run ORI (base model, no training) evaluation on all 8 TRACE tasks.
# Starts vLLM server locally, runs evaluation, then stops server.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"

MODEL_PATH="$AUDIT_DIR/checkpoints/base_models/Qwen2-1.5B"
DATA_PATH="$AUDIT_DIR/data/trace_tasks/TRACE-Benchmark/LLM-CL-Benchmark_5000"
OUTPUT_DIR="$AUDIT_DIR/results/ori_eval"
PORT=8001

mkdir -p "$OUTPUT_DIR"

echo "Starting vLLM server for ORI evaluation..."
vllm serve "$MODEL_PATH" \
    --port $PORT \
    --data-parallel-size ${VLLM_DP:-1} \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo "Running TRACE evaluation..."
python "$AUDIT_DIR/evaluation/trace_eval.py" \
    --base_url "http://localhost:$PORT/v1" \
    --api_key "EMPTY" \
    --model_name "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --tasks "C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten" \
    --output_dir "$OUTPUT_DIR" \
    --temperature 0.1 \
    --max_tokens 512 \
    --tokenizer_path "$MODEL_PATH" \
    --max_prompt_len 1024

echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true

echo "ORI evaluation complete. Results in $OUTPUT_DIR"
