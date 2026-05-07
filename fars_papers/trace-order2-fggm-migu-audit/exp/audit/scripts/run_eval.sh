#!/bin/bash
# Launch TRACE evaluation against a vLLM endpoint.
# Usage: bash audit/scripts/run_eval.sh <base_url> <api_key> <model_name> <tasks> <output_dir>

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDIT_DIR="$SCRIPT_DIR/.."

source "$PROJECT_DIR/.venv/bin/activate"

BASE_URL="${1:?Usage: $0 <base_url> <api_key> <model_name> <tasks> <output_dir>}"
API_KEY="${2:-EMPTY}"
MODEL_NAME="${3:?model name required}"
TASKS="${4:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}"
OUTPUT_DIR="${5:?output dir required}"
DATA_PATH="$AUDIT_DIR/data/trace_tasks/TRACE-Benchmark/LLM-CL-Benchmark_5000"

python "$AUDIT_DIR/evaluation/trace_eval.py" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --tasks "$TASKS" \
    --output_dir "$OUTPUT_DIR" \
    --temperature 0.1 \
    --max_tokens 512
