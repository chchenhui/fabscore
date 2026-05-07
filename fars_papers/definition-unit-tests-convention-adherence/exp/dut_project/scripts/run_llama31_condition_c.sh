#!/bin/bash
# Run Condition C (discriminative DUT checks, k=3) inference for Llama-3.1-8B-Instruct on ErdosConventionsBench.
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

export WANDB_MODE=offline

MODEL="meta-llama/Llama-3.1-8B-Instruct"
INPUT="dut_project/data/erdos_conventions_bench.jsonl"
OUTDIR="dut_project/outputs/llama31_8b"

mkdir -p "$OUTDIR"

echo "=========================================="
echo "Running Condition C (discriminative DUT) - Llama-3.1-8B-Instruct"
echo "=========================================="
python -m dut_project.inference.run_inference \
    --model "$MODEL" \
    --condition C \
    --input "$INPUT" \
    --output "$OUTDIR/condition_c.jsonl" \
    --temperature 0.0 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --k 3

echo "Condition C done. Output in $OUTDIR/condition_c.jsonl"
