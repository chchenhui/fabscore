#!/bin/bash
# Merge all proxy LoRA adapters with base model (Qwen2.5-1.5B).
# Usage: bash run_proxy_merge.sh <regime>
set -euo pipefail

REGIME="${1:-proxy_std}"

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate
set -a
source .env
set +a

export PYTHONUNBUFFERED=1

echo "=== Merging LoRA adapters for regime: $REGIME ==="
python tlr_proxy_sft/scripts/merge_lora_proxy.py --regime "$REGIME"
echo "=== Merge Complete ==="
