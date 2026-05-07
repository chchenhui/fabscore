#!/bin/bash
# Batch evaluate all merged proxy checkpoints on GSM8K and MATH-500.
# Qwen2.5-1.5B has 12 attention heads -> tp=1 works, use dp=8 for max throughput.
# Usage: bash run_proxy_eval.sh <regime>
# Example: bash run_proxy_eval.sh proxy_std
set -euo pipefail

REGIME="${1:-proxy_std}"

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

OUTPUT_BASE="${PROJECT_ROOT}/tlr_proxy_sft/outputs/${REGIME}"
RESULTS_BASE="${PROJECT_ROOT}/tlr_proxy_sft/results/${REGIME}"
mkdir -p "$RESULTS_BASE"

NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected: $NPROC"

TP_SIZE=1
DP_SIZE=1
echo "tensor_parallel_size: $TP_SIZE, data_parallel_size: $DP_SIZE (sequential, 1.5B model evals are fast)"

DATASETS=(
    "AM-Thinking-v1-Distilled-math"
    "DeepMath-309K"
    "Maths-College"
    "OpenR1-Math"
    "QwQ-LongCoT-130K-math"
    "R1-Distill-SFT-math"
    "hkust-nlp__dart-math-hard"
    "mathplus"
    "numinamath-cot"
    "numinamath1_5"
    "openmathinstruct-2"
    "Magpie-Reasoning-V2-250K-CoT-QwQ-math"
)
SEEDS=(42 123 456)

total=0
skipped=0
success=0
failed=0

for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))
        merged_path="${OUTPUT_BASE}/${ds}/seed_${seed}/merged"
        result_dir="${RESULTS_BASE}/${ds}/seed_${seed}"

        if [ ! -d "$merged_path" ] || [ ! -f "${merged_path}/config.json" ]; then
            echo "SKIP: ${ds}/seed_${seed} (no merged model)"
            skipped=$((skipped + 1))
            continue
        fi

        existing_results=$(find "$result_dir" -name "results_*.json" 2>/dev/null | head -1 || true)
        if [ -n "$existing_results" ]; then
            echo "SKIP: ${ds}/seed_${seed} (results exist)"
            skipped=$((skipped + 1))
            continue
        fi

        echo "EVAL: ${ds}/seed_${seed}"
        mkdir -p "$result_dir"

        if lm_eval \
            --model vllm \
            --model_args "pretrained=${merged_path},dtype=bfloat16,tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},gpu_memory_utilization=0.9,max_model_len=4096" \
            --tasks gsm8k,minerva_math500 \
            --num_fewshot 0 \
            --batch_size auto \
            --output_path "$result_dir" \
            --log_samples; then
            echo "DONE: ${ds}/seed_${seed}"
            success=$((success + 1))
        else
            echo "FAILED: ${ds}/seed_${seed}"
            failed=$((failed + 1))
        fi
    done
done

echo "=== Evaluation Complete ==="
echo "Total: $total, Skipped: $skipped, Success: $success, Failed: $failed"
