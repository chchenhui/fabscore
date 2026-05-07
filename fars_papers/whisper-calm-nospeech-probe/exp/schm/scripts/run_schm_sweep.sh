#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/whisper-calm-nospeech-probe/exp"
cd "$PROJECT_ROOT"

source .venv/bin/activate

set -a
source .env
set +a

export PYTHONUNBUFFERED=1

EXTRA_ARGS=""
if [ "${SANITY_CHECK:-0}" = "1" ]; then
    EXTRA_ARGS="--sanity-check"
fi

python -m schm.inference.run_schm_sweep \
    --batch-size 16 \
    --tau-list 0.3 0.4 0.5 0.6 \
    --modes suppress mask \
    $EXTRA_ARGS 2>&1
