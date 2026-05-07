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

python -m schm.inference.validate_p_nospeech --batch-size 16 $EXTRA_ARGS 2>&1
