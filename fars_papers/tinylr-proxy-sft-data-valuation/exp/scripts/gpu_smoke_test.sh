#!/bin/bash
# GPU smoke test: verify torch+CUDA, model loading, lm-eval, and LLaMA-Factory CLI
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp/.venv/bin/activate

echo "=== Step 1: Verify torch CUDA ==="
python3 -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print(f'GPU matmul OK, result shape: {y.shape}')
"

echo "=== Step 2: Load Qwen2.5-1.5B on GPU ==="
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'Qwen/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
inputs = tokenizer('1+1=', return_tensors='pt').to(model.device)
out = model.generate(**inputs, max_new_tokens=10)
print(f'Model output: {tokenizer.decode(out[0], skip_special_tokens=True)}')
print('Qwen2.5-1.5B loaded and generated OK.')
del model
torch.cuda.empty_cache()
"

echo "=== Step 3: lm-eval dry run (gsm8k, 5 samples) ==="
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B,dtype=float16 --tasks gsm8k --limit 5 --num_fewshot 0 --batch_size 4

echo "=== Step 4: LLaMA-Factory CLI ==="
llamafactory-cli version

echo "=== ALL GPU SMOKE TESTS PASSED ==="
