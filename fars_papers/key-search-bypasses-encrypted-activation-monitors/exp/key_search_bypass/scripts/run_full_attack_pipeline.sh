#!/bin/bash
# Full attack pipeline: validate encryptor, extract activations, train monitor,
# run random + gradient key-search attacks.
set -e
PROJ_DIR="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/key-search-bypasses-encrypted-activation-monitors/exp"
source "${PROJ_DIR}/.venv/bin/activate"
set -a && source "${PROJ_DIR}/.env" && set +a
export WANDB_MODE=offline
cd "${PROJ_DIR}"
python -u key_search_bypass/scripts/run_full_attack_pipeline.py "$@" 2>&1
