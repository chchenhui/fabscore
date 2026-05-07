#!/bin/bash
# Inference: Qwen2.5-Math-7B-Instruct, Condition A (glossary-only) on ErdosConventionsBench
set -euo pipefail

PROJ=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/definition-unit-tests-convention-adherence/exp
source "$PROJ/.venv/bin/activate"
cd "$PROJ"

python -m dut_project.inference.run_inference \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --condition A \
    --input dut_project/data/erdos_conventions_bench.jsonl \
    --output dut_project/outputs/qwen25_math_7b/condition_a.jsonl \
    --temperature 0.0 \
    --max-tokens 2048 \
    --tensor-parallel-size 1
