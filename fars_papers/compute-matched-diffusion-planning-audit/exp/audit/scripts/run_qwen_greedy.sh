#!/bin/bash
set -e
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/compute-matched-diffusion-planning-audit/exp
source .venv/bin/activate
python audit/inference/qwen_greedy.py --tasks countdown sudoku
echo "=== Qwen greedy inference completed ==="
