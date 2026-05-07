#!/bin/bash
# Download all required models and verify environment setup on GPU node.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate

export HF_TOKEN=$(grep HF_TOKEN /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.env | cut -d= -f2)

echo "=== Step 1: Download models ==="
MODELS=(
    "bcywinski/gemma-2-9b-it-taboo-gold"
    "bcywinski/gemma-2-9b-it-taboo-moon"
    "bcywinski/gemma-2-9b-it-taboo-flag"
    "bcywinski/gemma-2-9b-it-user-female"
    "bcywinski/gemma-2-9b-it-user-male"
    "google/gemma-2-9b-it"
    "google/gemma-3-4b-it"
)

for model in "${MODELS[@]}"; do
    echo "--- Downloading $model ---"
    huggingface-cli download "$model" --token "$HF_TOKEN"
    echo "--- Done: $model ---"
done

echo "=== Step 2: Verify GPU environment ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'Transformers version: {transformers.__version__}')

import accelerate
print(f'Accelerate version: {accelerate.__version__}')

import flash_attn
print(f'Flash-Attention version: {flash_attn.__version__}')

import seaborn
print(f'Seaborn version: {seaborn.__version__}')

import bitsandbytes
print(f'Bitsandbytes version: {bitsandbytes.__version__}')

import sae_lens
print(f'SAE-Lens version: {sae_lens.__version__}')

import scipy, numpy, pandas, matplotlib, tqdm, dotenv, safetensors
print('All imports successful.')
"

echo "=== Step 3: Verify model loading with GPU ==="
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'bcywinski/gemma-2-9b-it-taboo-gold'
print(f'Loading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Tokenizer loaded. Vocab size: {tokenizer.vocab_size}')

print(f'Loading model {model_name} in bf16...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='flash_attention_2',
)
print(f'Model loaded on device: {model.device}')
print(f'Model layers: {model.config.num_hidden_layers}')

inputs = tokenizer('Hello, world!', return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
print(f'Forward pass successful. Hidden states count: {len(outputs.hidden_states)}')
print(f'Hidden state shape: {outputs.hidden_states[0].shape}')
print(f'Logits shape: {outputs.logits.shape}')

del model
torch.cuda.empty_cache()
print('GPU memory freed.')
"

echo "=== All verification steps passed ==="
