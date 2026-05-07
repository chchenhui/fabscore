#!/bin/bash
# Verify environment: imports, GPU, and model loading with forward pass.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/suppression-contrast-secret-elicitation/exp/.venv/bin/activate

echo "=== Step 1: Verify package imports and GPU ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers; print(f'Transformers version: {transformers.__version__}')
import accelerate; print(f'Accelerate version: {accelerate.__version__}')
import flash_attn; print(f'Flash-Attention version: {flash_attn.__version__}')
import seaborn; print(f'Seaborn version: {seaborn.__version__}')
import bitsandbytes; print(f'Bitsandbytes version: {bitsandbytes.__version__}')
import sae_lens; print(f'SAE-Lens version: {sae_lens.__version__}')
import scipy, numpy, pandas, matplotlib, tqdm, dotenv, safetensors
print('All imports successful.')
"

echo "=== Step 2: Verify model loading and forward pass ==="
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
print(f'Model loaded. Num layers: {model.config.num_hidden_layers}')

inputs = tokenizer('Hello, world!', return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
print(f'Forward pass successful.')
print(f'Hidden states count: {len(outputs.hidden_states)}')
print(f'Hidden state shape: {outputs.hidden_states[0].shape}')
print(f'Logits shape: {outputs.logits.shape}')

del model
torch.cuda.empty_cache()
print('Cleanup done.')
"

echo "=== All verification steps passed ==="
