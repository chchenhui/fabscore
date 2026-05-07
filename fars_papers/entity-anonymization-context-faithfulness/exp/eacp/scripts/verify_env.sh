#!/bin/bash
# Verify the GPU environment: torch CUDA, transformers model loading, vllm init.
set -e

PROJ_DIR=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/entity-anonymization-context-faithfulness/exp
source ${PROJ_DIR}/.venv/bin/activate
source ${PROJ_DIR}/.env

export HF_TOKEN=${HF_TOKEN}

echo "=== Step 1: Verify torch + CUDA ==="
python -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
assert torch.cuda.is_available(), 'CUDA not available!'
print('PASSED')
"

echo "=== Step 2: Download and test Llama-3.1-8B-Instruct ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
print(f'Loading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('Tokenizer loaded.')

print(f'Loading model {model_name}...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
print('Model loaded.')

prompt = 'What is the capital of France?'
messages = [{'role': 'user', 'content': prompt}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(model.device)
print(f'Input tokens: {input_ids.shape}')

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f'Model response: {response}')
print('PASSED')

del model
torch.cuda.empty_cache()
"

echo "=== Step 3: Verify vllm ==="
python -c "
import vllm
print(f'vllm version: {vllm.__version__}')
print('vllm import PASSED')
"

echo "=== ALL CHECKS PASSED ==="
