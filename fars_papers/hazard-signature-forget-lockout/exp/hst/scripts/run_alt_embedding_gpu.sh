#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/hazard-signature-forget-lockout/exp
source .venv/bin/activate
python -u hst/scripts/run_alt_embedding_ablation.py 2>&1
echo "=== DONE ==="
