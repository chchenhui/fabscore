#!/bin/bash
# Majority-vote inference (5 samples) for Qwen2.5-Math-7B-Instruct, conditions A and B.
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

set -a; source "$PROJ/.env"; set +a
export WANDB_MODE=offline

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
INPUT="dut_project/data/erdos_conventions_bench.jsonl"
OUTDIR="dut_project/outputs/qwen25_math_7b"

mkdir -p "$OUTDIR"

for COND in A B; do
    echo "=========================================="
    echo "Qwen Maj@5 - Condition $COND"
    echo "=========================================="
    python -m dut_project.inference.run_inference \
        --model "$MODEL" \
        --condition "$COND" \
        --input "$INPUT" \
        --output "$OUTDIR/condition_$(echo $COND | tr 'A-Z' 'a-z')_maj5.jsonl" \
        --temperature 0.7 \
        --top-p 0.95 \
        --max-tokens 512 \
        --tensor-parallel-size 1 \
        --k 3 \
        --num-samples 5
    echo "Condition $COND done."
done

echo "Qwen maj@5 inference complete."
