#!/bin/bash
# Quick test: deploy checkpoint 1 (FOMC), run test_token_ids.py to verify fix.
set -e

PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp"
AUDIT_DIR="$PROJECT_DIR/audit"

source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

CKPT_DIR="$AUDIT_DIR/results/sft_default_seed42/checkpoints/1"
PORT=8001

echo "Starting vLLM server for checkpoint 1 (FOMC)..."
vllm serve "$CKPT_DIR" \
    --port $PORT \
    --data-parallel-size 2 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests \
    --gpu-memory-utilization 0.85 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID=$VLLM_PID)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server process died!"
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "vLLM server failed to start!"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo ""
echo "=== Running token_ids test ==="
python "$AUDIT_DIR/scripts/test_token_ids.py" "http://localhost:$PORT/v1" "$CKPT_DIR"

echo ""
echo "=== Running full FOMC evaluation with token_ids ==="
python "$AUDIT_DIR/evaluation/trace_eval.py" \
    --base_url "http://localhost:$PORT/v1" \
    --api_key "EMPTY" \
    --model_name "$CKPT_DIR" \
    --data_path "$AUDIT_DIR/data/trace_tasks/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
    --tasks "C-STANCE,FOMC" \
    --output_dir "$AUDIT_DIR/results/sft_default_seed42/eval_test" \
    --temperature 0.1 \
    --max_tokens 512 \
    --tokenizer_path "$AUDIT_DIR/checkpoints/base_models/Qwen2-1.5B" \
    --max_prompt_len 1024

echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "Done!"
