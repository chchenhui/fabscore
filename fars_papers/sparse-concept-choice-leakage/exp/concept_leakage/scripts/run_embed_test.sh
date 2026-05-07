#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp
source .venv/bin/activate
python concept_leakage/data/embed_test_data.py
echo "=== Test embedding done ==="
