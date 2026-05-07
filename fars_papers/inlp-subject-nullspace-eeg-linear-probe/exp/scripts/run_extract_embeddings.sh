#!/bin/bash
# Run frozen CBraMod embedding extraction on GPU
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/inlp-subject-nullspace-eeg-linear-probe/exp
source .venv/bin/activate
python project/data/extract_embeddings.py
