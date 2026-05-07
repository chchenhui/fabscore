#!/bin/bash
# Download all 103 training shards + 1 val shard of FineWeb10B (pre-tokenized GPT-2)
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp/.venv/bin/activate

export HF_TOKEN=$(grep HF_TOKEN /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp/.env | cut -d= -f2)

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/orthostochastic-mhc/exp/mHC-manifold-constrained-hyper-connections/examples/nanogpt/data/fineweb10B

python download.py 103

echo "Download complete. Listing files:"
ls -lh *.bin | wc -l
ls -lh *.bin | head -5
ls -lh *.bin | tail -5
