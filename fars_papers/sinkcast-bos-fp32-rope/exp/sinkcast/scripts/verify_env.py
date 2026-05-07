# Verification script: checks torch+CUDA, flash_attn with softmax_lse, transformers, and model loader.
# Run on GPU to confirm full environment is functional.

import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)

import transformers
print(f"Transformers: {transformers.__version__}")

import flash_attn
print(f"FlashAttention: {flash_attn.__version__}")

from flash_attn import flash_attn_func
print("flash_attn_func imported OK")

print("\n--- FlashAttention forward pass test ---")
batch, seqlen, nheads, d = 1, 128, 8, 64
q = torch.randn(batch, seqlen, nheads, d, dtype=torch.bfloat16, device="cuda")
k = torch.randn(batch, seqlen, nheads, d, dtype=torch.bfloat16, device="cuda")
v = torch.randn(batch, seqlen, nheads, d, dtype=torch.bfloat16, device="cuda")

out, softmax_lse, _ = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
print(f"Output shape: {out.shape}")
print(f"softmax_lse shape: {softmax_lse.shape}")
print(f"softmax_lse dtype: {softmax_lse.dtype}")
assert out.shape == (batch, seqlen, nheads, d), f"Unexpected output shape: {out.shape}"
assert softmax_lse.shape == (batch, nheads, seqlen), f"Unexpected lse shape: {softmax_lse.shape}"
print("FlashAttention test PASSED")

print("\n--- BF16/FP32 RoPE test ---")
cos = torch.randn(seqlen, d, dtype=torch.float32, device="cuda")
sin = torch.randn(seqlen, d, dtype=torch.float32, device="cuda")
x_fp32 = torch.randn(1, seqlen, nheads, d, dtype=torch.float32, device="cuda")
x_bf16 = x_fp32.to(torch.bfloat16)
x1, x2 = x_fp32[..., :d//2], x_fp32[..., d//2:]
rotated_fp32 = torch.cat([-x2, x1], dim=-1)
rope_fp32 = x_fp32 * cos.unsqueeze(0).unsqueeze(2) + rotated_fp32 * sin.unsqueeze(0).unsqueeze(2)
print(f"FP32 RoPE output dtype: {rope_fp32.dtype}, shape: {rope_fp32.shape}")
print("RoPE test PASSED")

print("\n--- Model loader import test ---")
sys.path.insert(0, "/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/sinkcast-bos-fp32-rope/exp")
from sinkcast.models.loader import load_model_and_tokenizer, SUPPORTED_MODELS
print(f"Supported models: {list(SUPPORTED_MODELS.keys())}")
print("Model loader import PASSED")

print("\n--- Additional packages ---")
import rouge_score; print(f"rouge_score OK")
import jieba; print(f"jieba OK")
import seaborn; print(f"seaborn OK")
import datasets; print(f"datasets OK")
import accelerate; print(f"accelerate OK")

print("\n=== ALL ENVIRONMENT CHECKS PASSED ===")
