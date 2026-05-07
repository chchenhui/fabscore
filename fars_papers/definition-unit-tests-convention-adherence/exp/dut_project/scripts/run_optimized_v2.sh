#!/bin/bash
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

export WANDB_MODE=offline

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
INPUT="dut_project/data/erdos_conventions_bench.jsonl"
OUTDIR="dut_project/outputs/qwen25_math_7b_v2"

mkdir -p "$OUTDIR"

for COND in A B C; do
    echo "=========================================="
    echo "Running Condition $COND (v2)"
    echo "=========================================="
    python -m dut_project.inference.run_inference \
        --model "$MODEL" \
        --condition "$COND" \
        --input "$INPUT" \
        --output "$OUTDIR/condition_$(echo $COND | tr 'A-Z' 'a-z').jsonl" \
        --temperature 0.0 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --k 3
    echo "Condition $COND done."
done

echo "All v2 conditions completed. Outputs in $OUTDIR/"
