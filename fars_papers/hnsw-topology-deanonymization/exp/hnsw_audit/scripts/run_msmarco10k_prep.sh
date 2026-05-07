#!/bin/bash
set -e
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/hnsw-topology-deanonymization/exp/.venv/bin/activate
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/hnsw-topology-deanonymization/exp/hnsw_audit

echo "Python: $(which python)"
echo "Starting MSMARCO-10K preparation..."
python scripts/run_msmarco10k_prep.py
echo "MSMARCO-10K preparation complete."
