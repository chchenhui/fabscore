#!/bin/bash
source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sparse-concept-choice-leakage/exp/.venv/bin/activate

python -c "
import sys
print('Python:', sys.version)

import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    x = torch.randn(10, 768, device='cuda')
    print('GPU tensor test passed, shape:', x.shape)

import sentence_transformers
print('sentence-transformers:', sentence_transformers.__version__)

import mteb
print('mteb:', mteb.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)

import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import seaborn
print('seaborn:', seaborn.__version__)

import pandas
print('pandas:', pandas.__version__)

import datasets
print('datasets:', datasets.__version__)

import transformers
print('transformers:', transformers.__version__)

print()
print('--- Functional Tests ---')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
emb = model.encode(['Hello world test sentence.'])
print(f'gtr-t5-base embedding dim: {emb.shape[1]} (expected 768)')
assert emb.shape[1] == 768, f'Wrong dim: {emb.shape[1]}'

from datasets import load_dataset
ds = load_dataset('ai4privacy/pii-masking-300k', split='train[:10]')
print(f'PII-Masking-300K sample loaded: {len(ds)} rows')
print(f'Columns: {ds.column_names}')

print()
print('=== ALL CHECKS PASSED ===')
"
