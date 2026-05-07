#!/bin/bash
# Quick sanity check for v3 training changes (AdamW + warmup)
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp
source .venv/bin/activate
source .env 2>/dev/null || true
export WANDB_MODE=offline
export HF_HOME=pretrained_models
export TRANSFORMERS_CACHE=pretrained_models
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH=.
CONFIG="scaffoldswap/configs/biwi_condC_debug_v3.yaml"
echo "Sanity check: BIWI condC v3 debug (20 epochs, AdamW + warmup)"
python3 -m scaffoldswap.train --config "${CONFIG}" --seed 42
echo "Sanity check complete"
