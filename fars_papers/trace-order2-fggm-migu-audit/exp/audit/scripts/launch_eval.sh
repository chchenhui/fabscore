#!/bin/bash
# Wrapper to launch full eval from TrainService.
set -e

PROJECT_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/trace-order2-fggm-migu-audit/exp"
source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

CHECKPOINT_BASE="${1:-$PROJECT_DIR/audit/results/sft_default_seed42/checkpoints}"
OUTPUT_BASE="${2:-$PROJECT_DIR/audit/results/sft_default_seed42/eval_v2}"
NUM_GPUS="${EVAL_NUM_GPUS:-${VLLM_DP:-8}}"

echo "Python: $(which python)"
echo "vLLM: $(which vllm)"
echo "NUM_GPUS: $NUM_GPUS"

export VLLM_DP=$NUM_GPUS
bash "$PROJECT_DIR/audit/scripts/run_full_eval.sh" "$CHECKPOINT_BASE" "$OUTPUT_BASE"
