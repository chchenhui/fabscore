#!/bin/bash
# Evaluate all 8 task checkpoints to build the 8x8 performance matrix.
# For checkpoint t, evaluate on tasks 0..t (all tasks seen so far).
# Starts vLLM server per checkpoint, runs eval, then stops.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"

CHECKPOINT_BASE="${1:?Usage: $0 <checkpoint_base_dir> [output_base_dir]}"
OUTPUT_BASE="${2:-$AUDIT_DIR/results/sft_default_seed42/eval}"
DATA_PATH="$AUDIT_DIR/data/trace_tasks/TRACE-Benchmark/LLM-CL-Benchmark_5000"
PORT=8001
TOKENIZER_PATH="$AUDIT_DIR/checkpoints/base_models/Qwen2-1.5B"
NUM_GPUS="${VLLM_DP:-8}"

TASKS=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")

for task_idx in $(seq 0 7); do
    CKPT_DIR="$CHECKPOINT_BASE/${task_idx}_best"
    if [ ! -d "$CKPT_DIR" ]; then
        CKPT_DIR="$CHECKPOINT_BASE/$task_idx"
    fi

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

    echo "Waiting for vLLM server..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "vLLM server failed to start!"
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
tasks = ['C-STANCE','FOMC','MeetingBank','Py150','ScienceQA','NumGLUE-cm','NumGLUE-ds','20Minuten']
matrix = {}
for t_idx in range(8):
    edir = os.path.join('$OUTPUT_BASE', f'checkpoint_{t_idx}')
    rfile = os.path.join(edir, 'eval_results.json')
    if os.path.exists(rfile):
        with open(rfile) as f:
            results = json.load(f)
        matrix[t_idx] = {tn: results[tn]['primary'] for tn in results}

op_T = sum(matrix.get(7,{}).values()) / max(len(matrix.get(7,{})), 1)
bwt_sum = 0
for i in range(7):
    tn = tasks[i]
    final = matrix.get(7,{}).get(tn, 0)
    diag = matrix.get(i,{}).get(tn, 0)
    bwt_sum += (final - diag)
bwt_T = bwt_sum / 7

print(f'TRACE-OP: {op_T:.2f}')
print(f'BWT: {bwt_T:.2f}')
print()
print('Performance matrix (primary metric per task):')
for t_idx in sorted(matrix.keys()):
    row = matrix[t_idx]
    vals = [f'{row.get(tn, \"-\"):>7.2f}' if isinstance(row.get(tn, '-'), (int,float)) else f'{\"-\":>7s}' for tn in tasks[:t_idx+1]]
    print(f'  After task {t_idx} ({tasks[t_idx]:>12s}): {\" \".join(vals)}')

summary = {'TRACE_OP': op_T, 'BWT': bwt_T, 'matrix': matrix}
with open(os.path.join('$OUTPUT_BASE', 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to $OUTPUT_BASE/summary.json')
"
