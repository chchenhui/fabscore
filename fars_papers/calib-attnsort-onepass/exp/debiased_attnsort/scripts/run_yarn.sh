#!/bin/bash
# Run YaRN-Llama-2-7b-64k evaluation on SynthWiki.
# Usage: bash run_yarn.sh --mode no_sorting --seed 42 [--num_examples 200] [--sanity_check]
set -euo pipefail
PROJ_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/calib-attnsort-onepass/exp"
source "${PROJ_ROOT}/.venv/bin/activate"
set -a
source "${PROJ_ROOT}/.env"
set +a
cd "${PROJ_ROOT}"
export PYTHONUNBUFFERED=1
python debiased_attnsort/src/eval_yarn.py "$@"
