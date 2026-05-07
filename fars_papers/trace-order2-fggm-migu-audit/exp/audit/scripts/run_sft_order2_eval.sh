#!/bin/bash
# Full evaluation for SFT Order 2: starts vLLM per checkpoint, evaluates, builds matrix.
# Usage: bash audit/scripts/run_sft_order2_eval.sh <seed> [num_gpus]
# Example: bash audit/scripts/run_sft_order2_eval.sh 42 8
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

SEED="${1:?Usage: $0 <seed> [num_gpus]}"
NUM_GPUS="${2:-8}"

CHECKPOINT_BASE="$AUDIT_DIR/results/sft_order2_seed${SEED}/checkpoints"
OUTPUT_BASE="$AUDIT_DIR/results/sft_order2_seed${SEED}/eval"
DATA_PATH="$AUDIT_DIR/data/trace_tasks/TRACE-Benchmark/LLM-CL-Benchmark_5000"
TOKENIZER_PATH="$AUDIT_DIR/checkpoints/base_models/Qwen2-1.5B"
PORT=8001

TASKS=("NumGLUE-cm" "NumGLUE-ds" "FOMC" "20Minuten" "C-STANCE" "Py150" "MeetingBank" "ScienceQA")

for task_idx in $(seq 0 7); do
    CKPT_DIR="$CHECKPOINT_BASE/$task_idx"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "Checkpoint not found for task $task_idx, skipping."
        continue
    fi

    OUTPUT_DIR="$OUTPUT_BASE/checkpoint_$task_idx"
    RESULT_FILE="$OUTPUT_DIR/eval_results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "Results already exist for checkpoint $task_idx at $RESULT_FILE, skipping."
        continue
    fi

    EVAL_TASKS=""
    for eval_idx in $(seq 0 $task_idx); do
        if [ -n "$EVAL_TASKS" ]; then
            EVAL_TASKS="$EVAL_TASKS,"
        fi
        EVAL_TASKS="$EVAL_TASKS${TASKS[$eval_idx]}"
    done

    mkdir -p "$OUTPUT_DIR"

    echo "============================================"
    echo "Evaluating checkpoint $task_idx: $CKPT_DIR"
    echo "Tasks: $EVAL_TASKS"
    echo "Output: $OUTPUT_DIR"
    echo "============================================"

    echo "Starting vLLM server with DP=$NUM_GPUS..."
    vllm serve "$CKPT_DIR" \
        --port $PORT \
        --data-parallel-size $NUM_GPUS \
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
            echo "vLLM server failed to start in 10 minutes!"
            kill $VLLM_PID 2>/dev/null
            exit 1
        fi
        sleep 5
    done

    python "$AUDIT_DIR/evaluation/trace_eval.py" \
        --base_url "http://localhost:$PORT/v1" \
        --api_key "EMPTY" \
        --model_name "$CKPT_DIR" \
        --data_path "$DATA_PATH" \
        --tasks "$EVAL_TASKS" \
        --output_dir "$OUTPUT_DIR" \
        --temperature 0.1 \
        --max_tokens 512 \
        --tokenizer_path "$TOKENIZER_PATH" \
        --max_prompt_len 1024

    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null || true
    sleep 5

    echo "Checkpoint $task_idx evaluation complete."
done

echo ""
echo "All evaluations complete. Building performance matrix..."

python -c "
import json, os
tasks = ['NumGLUE-cm','NumGLUE-ds','FOMC','20Minuten','C-STANCE','Py150','MeetingBank','ScienceQA']
base = '$OUTPUT_BASE'
matrix = {}
for t_idx in range(8):
    edir = os.path.join(base, f'checkpoint_{t_idx}')
    rfile = os.path.join(edir, 'eval_results.json')
    if os.path.exists(rfile):
        with open(rfile) as f:
            results = json.load(f)
        matrix[t_idx] = {tn: results[tn]['primary'] for tn in results}

# TRACE-OP: average of OP_t for t=1..T, where OP_t = (1/t) * sum_{i=1}^{t} R_{t,i}
T = len(matrix)
op_per_step = []
for t in range(T):
    row = matrix.get(t, {})
    if row:
        op_t = sum(row.values()) / len(row)
        op_per_step.append(op_t)
    else:
        op_per_step.append(0.0)
trace_op = sum(op_per_step) / T if T > 0 else 0.0

# BWT
bwt_sum = 0
for i in range(T - 1):
    tn = tasks[i]
    final = matrix.get(T-1, {}).get(tn, 0)
    diag = matrix.get(i, {}).get(tn, 0)
    bwt_sum += (final - diag)
bwt_T = bwt_sum / (T - 1) if T > 1 else 0.0

print(f'TRACE-OP: {trace_op:.2f}')
print(f'BWT: {bwt_T:.2f}')
print()
print('OP per step:')
for i, op in enumerate(op_per_step):
    print(f'  OP_{i+1}: {op:.2f}')
print()
print('Performance matrix (primary metric per task):')
for t_idx in sorted(matrix.keys()):
    row = matrix[t_idx]
    vals = [f'{row.get(tn, \"-\"):>7.2f}' if isinstance(row.get(tn, '-'), (int,float)) else f'{\"-\":>7s}' for tn in tasks[:t_idx+1]]
    print(f'  After task {t_idx} ({tasks[t_idx]:>12s}): {\" \".join(vals)}')

summary = {
    'TRACE_OP': trace_op,
    'BWT': bwt_T,
    'OP_per_step': op_per_step,
    'matrix': {str(k): v for k, v in matrix.items()},
    'task_order': tasks,
}
with open(os.path.join(base, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {base}/summary.json')
"
