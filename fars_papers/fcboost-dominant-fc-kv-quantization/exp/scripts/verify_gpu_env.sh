#!/bin/bash
# GPU environment verification script for FCBoost.
# Verifies: PyTorch+CUDA, model loading, kitty_sim import, lm_eval import, trivial forward pass.
set -e

WORKDIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/fcboost-dominant-fc-kv-quantization/exp
source ${WORKDIR}/.venv/bin/activate

export TORCH_CUDA_ARCH_LIST="8.0"

echo "=== GPU Info ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

echo "=== Import Checks ==="
python -c "
import kitty_sim
print('kitty_sim OK')
from kitty_sim.kitty_simulate import KittyKVCacheConfig, KittyKVCache
print('KittyKVCache OK')
import lm_eval
print(f'lm_eval {lm_eval.__version__} OK')
import transformers
print(f'transformers {transformers.__version__} OK')
import flash_attn
print(f'flash_attn {flash_attn.__version__} OK')
"

echo "=== Model Load + Forward Pass ==="
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen3-8B'
print(f'Loading tokenizer from {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f'Loading model from {model_name} to GPU...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
)
print(f'Model loaded. dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}')

config = model.config
print(f'Layers: {config.num_hidden_layers}, QHeads: {config.num_attention_heads}, KVHeads: {config.num_key_value_heads}, HeadDim: {config.head_dim}')

inputs = tokenizer('Hello, world!', return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
print(f'Forward pass OK. Logits shape: {outputs.logits.shape}')

gen = model.generate(**inputs, max_new_tokens=16)
decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
print(f'Generation OK: {decoded[:100]}')
print('ALL CHECKS PASSED')
"
