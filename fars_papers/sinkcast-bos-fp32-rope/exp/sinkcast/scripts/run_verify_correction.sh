#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python -c "
import torch
import math
from flash_attn import flash_attn_func
from sinkcast.core.rope_utils import compute_fp32_cos_sin, fp32_rope_rotate

torch.manual_seed(42)
B, S, H, H_kv, D = 1, 64, 32, 8, 128
groups = H // H_kv
device = 'cuda'

q_raw = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
k_raw = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float32)
v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float32)

pos_ids = torch.arange(S, device=device).unsqueeze(0) + 256
inv_freq = 1.0 / (500000.0 ** (torch.arange(0, D, 2, device=device).float() / D))
rope_config = {'inv_freq': inv_freq, 'head_dim': D, 'attention_scaling': 1.0}

cos, sin = compute_fp32_cos_sin(inv_freq, pos_ids, 1.0)
cos_bf16 = cos.bfloat16().float()
sin_bf16 = sin.bfloat16().float()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

cos_u = cos_bf16.unsqueeze(1)
sin_u = sin_bf16.unsqueeze(1)
q_rot_bf16 = (q_raw * cos_u + rotate_half(q_raw) * sin_u).bfloat16()
k_rot_bf16 = (k_raw * cos_u + rotate_half(k_raw) * sin_u).bfloat16()

k_rot_bf16_exp = k_rot_bf16.repeat_interleave(groups, dim=1)
v_exp = v.repeat_interleave(groups, dim=1)
k_raw_exp = k_raw.repeat_interleave(groups, dim=1)

q_rot_fp32 = fp32_rope_rotate(q_raw, cos, sin)
k_rot_fp32_exp = fp32_rope_rotate(k_raw, cos, sin).repeat_interleave(groups, dim=1)

q_fa = q_rot_bf16.transpose(1,2)
k_fa = k_rot_bf16.transpose(1,2)
v_fa = v.bfloat16().transpose(1,2)
fa_out, fa_lse, _ = flash_attn_func(q_fa, k_fa, v_fa, causal=True, return_attn_probs=True)

logits_fp32_full = torch.einsum('bhsd,bhkd->bhsk', q_rot_fp32.float(), k_rot_fp32_exp.float()) / math.sqrt(D)
causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
logits_fp32_full.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
probs_fp32 = torch.softmax(logits_fp32_full, dim=-1)
o_fp32 = torch.einsum('bhsk,bhkd->bhsd', probs_fp32, v_exp.float())

o_flash = fa_out.transpose(1,2).float()
lse = fa_lse.float()
bf16_err = (o_flash - o_fp32).abs()
print(f'BF16 Flash vs FP32 oracle: max={bf16_err.max():.6f}, mean={bf16_err.mean():.6f}')
print()

# Batch correction with causal mask
def sinkcast_batch_causal(fa_out, lse, q_bf16_rot, k_bf16_rot_exp, q_raw, k_raw_exp, v_exp, pos_ids, rope_config, K):
    o = fa_out.transpose(1,2).float()
    lse_cur = lse.clone()
    B, H, S, D = o.shape
    scale = 1.0 / math.sqrt(D)
    inv_freq = rope_config['inv_freq']
    cos_q, sin_q = compute_fp32_cos_sin(inv_freq, pos_ids, rope_config['attention_scaling'])
    q_fp32 = fp32_rope_rotate(q_raw, cos_q, sin_q)

    old_probs = []
    new_logits = []
    v_list = []

    for j in range(K):
        k_old_j = k_bf16_rot_exp[:, :, j, :]
        a_old = torch.einsum('bhsd,bhd->bhs', q_bf16_rot.float(), k_old_j.float()) * scale
        a_old[:, :, :j] = float('-inf')
        p_old_j = torch.exp(a_old - lse_cur)
        old_probs.append(p_old_j)

        pos_k_j = pos_ids[:, j:j+1]
        cos_k, sin_k = compute_fp32_cos_sin(inv_freq, pos_k_j, rope_config['attention_scaling'])
        k_raw_j = k_raw_exp[:, :, j:j+1, :]
        k_fp32_j = fp32_rope_rotate(k_raw_j, cos_k, sin_k).squeeze(2)
        a_new_j = torch.einsum('bhsd,bhd->bhs', q_fp32, k_fp32_j) * scale
        a_new_j[:, :, :j] = float('-inf')
        new_logits.append(a_new_j)
        v_list.append(v_exp[:, :, j, :].float())

    p_old_sum = torch.stack(old_probs, dim=0).sum(dim=0).clamp(max=1.0 - 1e-6)
    logZ_minus = lse_cur + torch.log1p(-p_old_sum)

    new_logits_t = torch.stack(new_logits, dim=0)
    lse_new = torch.logsumexp(new_logits_t, dim=0)
    lse_prime = torch.logaddexp(logZ_minus, lse_new)

    scale_f = torch.exp(lse_cur - lse_prime)
    wv_old = sum(old_probs[j].unsqueeze(-1) * v_list[j].unsqueeze(2) for j in range(K))
    wv_new = sum(torch.exp(new_logits[j] - lse_prime).unsqueeze(-1) * v_list[j].unsqueeze(2) for j in range(K))

    o = scale_f.unsqueeze(-1) * (o - wv_old) + wv_new
    return o

print('=== Batch correction WITH causal mask ===')
for K in [1, 4, 8, 16, 32, S]:
    o_corr = sinkcast_batch_causal(fa_out, lse, q_rot_bf16, k_rot_bf16_exp, q_raw, k_raw_exp, v_exp, pos_ids, rope_config, K)
    diff = (o_corr - o_fp32).abs()
    gc = 1.0 - diff.mean() / bf16_err.mean()
    print(f'K={K:3d}: max={diff.max():.6f}, mean={diff.mean():.6f}, gap_closure={gc:.4%}')
    import sys; sys.stdout.flush()

# Also test: what if we do ALL keys (should converge to FP32 oracle)?
print()
print('Checking K=S convergence to FP32 oracle...')
# With all keys corrected, the result should be equivalent to full FP32 attention
# The remaining error is only from: (1) flash_attn lse vs our recomp lse mismatch, 
# and (2) flash_attn output vs the output we'd get with our lse
"
