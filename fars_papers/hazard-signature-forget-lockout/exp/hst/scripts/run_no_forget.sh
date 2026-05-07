#!/bin/bash
set -e

PROJECT_ROOT="/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/hazard-signature-forget-lockout/exp"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"
python -m hst.scripts.run_no_forget
