#!/bin/bash
# Sanity check: verify SinkCast d=0 matches BF16 d=0 by running both on
# 5 RULER niah_single samples at seq_len 4096.
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sinkcast-bos-fp32-rope/exp
source .venv/bin/activate

export WANDB_MODE=offline
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/shared/huggingface
export HF_HUB_CACHE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/shared/huggingface/hub

set -a
source .env
set +a

MODEL="llama-3.1-8b"

echo "=== Running BF16 baseline ==="
python3 sinkcast/benchmarks/ruler_shift.py \
    --model "$MODEL" \
    --seq_lengths 4096 \
    --tasks niah_single \
    --n_samples 50 \
    --debug_n 5 \
    --shift_M 4096 \
    --output_dir "sinkcast/results/downstream/sanity_bf16" \
    --seed 42

echo ""
echo "=== Running SinkCast K=4 ==="
python3 sinkcast/benchmarks/ruler_shift_sinkcast.py \
    --model "$MODEL" \
    --seq_lengths 4096 \
    --tasks niah_single \
    --n_samples 50 \
    --debug_n 5 \
    --shift_M 4096 \
    --K 4 \
    --output_dir "sinkcast/results/downstream/sanity_sc" \
    --seed 42

echo ""
echo "=== Comparing results ==="
python3 -c "
import json
with open('sinkcast/results/downstream/sanity_bf16/ruler_llama-3.1-8b.json') as f:
    bf16 = json.load(f)
with open('sinkcast/results/downstream/sanity_sc/ruler_llama-3.1-8b.json') as f:
    sc = json.load(f)

for seq_len in bf16['results']:
    for task in bf16['results'][seq_len]:
        if task == 'overall':
            continue
        bf16_d0 = bf16['results'][seq_len][task]['accuracy_delta0']
        sc_d0 = sc['results'][seq_len][task]['accuracy_delta0']
        bf16_dM = bf16['results'][seq_len][task]['accuracy_deltaM']
        sc_dM = sc['results'][seq_len][task]['accuracy_deltaM']
        bf16_drop = bf16['results'][seq_len][task]['drop']
        sc_drop = sc['results'][seq_len][task]['drop']
        match = 'MATCH' if abs(bf16_d0 - sc_d0) < 0.01 else 'MISMATCH'
        print(f'{task} seq={seq_len}: BF16 d0={bf16_d0:.1f} dM={bf16_dM:.1f} drop={bf16_drop:.1f} | SC d0={sc_d0:.1f} dM={sc_dM:.1f} drop={sc_drop:.1f} | d0: {match}')
"

echo "=== Sanity check complete ==="
