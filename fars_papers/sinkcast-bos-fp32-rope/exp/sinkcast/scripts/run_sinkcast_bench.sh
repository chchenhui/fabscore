#!/bin/bash
# Run SinkCast K=1 shift microbenchmark on GPU via TrainService.
# Usage: MODEL=llama-3.1-8b bash sinkcast/scripts/run_sinkcast_bench.sh
#   or:  MODEL=mistral-7b-v0.3 DEBUG=1 bash sinkcast/scripts/run_sinkcast_bench.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

MODEL="${MODEL:-llama-3.1-8b}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/sinkcast/results/microbench/sinkcast_k1}"

DEBUG_FLAG=""
if [ "${DEBUG}" = "1" ]; then
    DEBUG_FLAG="--debug"
fi

echo "Running SinkCast K=1 microbenchmark: model=$MODEL"
python "$PROJECT_ROOT/sinkcast/benchmarks/sinkcast_microbench.py" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $DEBUG_FLAG

echo "Finished SinkCast benchmark for $MODEL"
