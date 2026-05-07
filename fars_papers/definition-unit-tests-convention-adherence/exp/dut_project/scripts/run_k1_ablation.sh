#!/bin/bash
# Run Condition C inference with k=1 (single discriminative check) on both models.
# This is the k-ablation experiment comparing k=1 vs default k=3.
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

export WANDB_MODE=offline

INPUT="dut_project/data/erdos_conventions_bench.jsonl"

echo "=========================================="
echo "K-ABLATION: Condition C with k=1"
echo "=========================================="

echo "--- Qwen2.5-Math-7B-Instruct (k=1) ---"
python -m dut_project.inference.run_inference \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --condition C \
    --input "$INPUT" \
    --output dut_project/outputs/qwen25_math_7b/condition_c_k1.jsonl \
    --temperature 0.0 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --k 1
echo "Qwen k=1 done."

echo "--- Llama-3.1-8B-Instruct (k=1) ---"
python -m dut_project.inference.run_inference \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --condition C \
    --input "$INPUT" \
    --output dut_project/outputs/llama31_8b/condition_c_k1.jsonl \
    --temperature 0.0 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --k 1
echo "Llama k=1 done."

echo "=========================================="
echo "K-ablation inference complete."
echo "=========================================="
