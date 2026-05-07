#!/bin/bash
# Batch launcher for 36 optimized tiny-LR proxy runs (LR=1e-5, 1000 steps).
# Runs 8 jobs in parallel on 8 GPUs, cycling through all configs.
# Skips configs whose output directories already contain adapter_config.json.
set -euo pipefail

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-tinylr-proxy-sft-data-valuation}"
export PYTHONPATH="${PROJECT_ROOT}/LlamaFactory/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export TRITON_CACHE_DIR="/tmp/triton_cache"
export TRITON_HOME="/tmp/triton_home"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME"
export PYTHONUNBUFFERED=1

LAUNCHER="${PROJECT_ROOT}/LlamaFactory/src/llamafactory/launcher.py"
CONFIG_DIR="${PROJECT_ROOT}/tlr_proxy_sft/configs/proxy_tiny_v2"
OUTPUT_BASE="${PROJECT_ROOT}/tlr_proxy_sft/outputs/proxy_tiny_v2"
NUM_GPUS=8

CONFIGS=()
for cfg in "$CONFIG_DIR"/*.yaml; do
    basename_cfg=$(basename "$cfg" .yaml)
    ds_name=$(echo "$basename_cfg" | sed 's/_seed[0-9]*$//')
    seed=$(echo "$basename_cfg" | grep -oP 'seed\K[0-9]+')
    output_dir="${OUTPUT_BASE}/${ds_name}/seed_${seed}"

    if [ -f "${output_dir}/adapter_config.json" ]; then
        echo "SKIP: ${basename_cfg} (already completed)"
        continue
    fi
    CONFIGS+=("$cfg")
done

TOTAL=${#CONFIGS[@]}
echo "=== Proxy Tiny-v2 (LR=1e-5, 1000 steps) Batch Training ==="
echo "Total configs to run: $TOTAL"
echo "GPUs available: $NUM_GPUS"
echo "============================================================"

run_one() {
    local gpu_id=$1
    local config_path=$2
    local config_name=$(basename "$config_path" .yaml)
    echo "[GPU $gpu_id] START: $config_name"
    CUDA_VISIBLE_DEVICES=$gpu_id python "$LAUNCHER" "$config_path" 2>&1 | while IFS= read -r line; do
        echo "[GPU $gpu_id] $line"
    done
    echo "[GPU $gpu_id] DONE: $config_name"
}

idx=0
while [ $idx -lt $TOTAL ]; do
    PIDS=()
    wave_start=$idx
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $idx -ge $TOTAL ]; then
            break
        fi
        run_one "$gpu_id" "${CONFIGS[$idx]}" &
        PIDS+=($!)
        idx=$((idx + 1))
    done

    wave_end=$((idx - 1))
    echo "=== Wave: configs $wave_start-$wave_end (${#PIDS[@]} jobs) ==="

    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "WARNING: job PID $pid failed"
            FAILED=$((FAILED + 1))
        fi
    done
    echo "=== Wave complete. Failed: $FAILED ==="
done

echo "=== All waves complete ==="

completed=0
for cfg in "$CONFIG_DIR"/*.yaml; do
    basename_cfg=$(basename "$cfg" .yaml)
    ds_name=$(echo "$basename_cfg" | sed 's/_seed[0-9]*$//')
    seed=$(echo "$basename_cfg" | grep -oP 'seed\K[0-9]+')
    output_dir="${OUTPUT_BASE}/${ds_name}/seed_${seed}"
    if [ -f "${output_dir}/adapter_config.json" ]; then
        completed=$((completed + 1))
    fi
done
echo "Completed: $completed / 36"
