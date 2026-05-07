#!/bin/bash
# Optimized Llama-3.1-8B-Instruct run: all 3 conditions (A, B, C) with frequency_penalty
# to prevent repetition loops, and increased max_tokens for longer reasoning.
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

export WANDB_MODE=offline

MODEL="meta-llama/Llama-3.1-8B-Instruct"
INPUT="dut_project/data/erdos_conventions_bench.jsonl"
OUTDIR="dut_project/outputs/llama31_8b_opt3"
FREQ_PENALTY=0.3
MAX_TOKENS=4096

mkdir -p "$OUTDIR"

for COND in A B C; do
    echo "=========================================="
    echo "Running Condition $COND - Llama-3.1-8B-Instruct (optimized)"
    echo "  frequency_penalty=$FREQ_PENALTY, max_tokens=$MAX_TOKENS"
    echo "=========================================="
    python -m dut_project.inference.run_inference \
        --model "$MODEL" \
        --condition "$COND" \
        --input "$INPUT" \
        --output "$OUTDIR/condition_${COND,,}.jsonl" \
        --temperature 0.0 \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size 1 \
        --k 3 \
        --frequency-penalty "$FREQ_PENALTY"
    echo "Condition $COND done."
    echo
done

echo "All conditions complete. Outputs in $OUTDIR/"
