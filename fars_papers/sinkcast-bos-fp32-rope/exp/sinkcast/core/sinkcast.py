# SinkCast correction: recomputes sink-key logits in FP32 and applies
# closed-form output correction using FlashAttention's softmax_lse.
# Supports K>=1 keys with batch correction and causal masking.
# The causal mask ensures query i only corrects keys j where i >= j.

import math
from typing import Dict

import torch

from sinkcast.core.rope_utils import compute_fp32_cos_sin, fp32_rope_rotate


def sinkcast_correct(
    flash_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    q_bf16_rotated: torch.Tensor,
    k_bf16_rotated: torch.Tensor,
    q_raw: torch.Tensor,
    k_sink_raw: torch.Tensor,
    v_sink: torch.Tensor,
    position_ids: torch.Tensor,
    rope_config: Dict,
    K: int = 1,
) -> torch.Tensor:
    device = flash_output.device
    head_dim = rope_config["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)
    num_heads = q_bf16_rotated.shape[1]
    num_kv_heads = k_bf16_rotated.shape[1]
    num_groups = num_heads // num_kv_heads

    if num_groups > 1:
        k_bf16_rot_expanded = k_bf16_rotated.repeat_interleave(num_groups, dim=1)
        k_sink_raw_expanded = k_sink_raw.repeat_interleave(num_groups, dim=1)
        v_sink_expanded = v_sink.repeat_interleave(num_groups, dim=1)
    else:
        k_bf16_rot_expanded = k_bf16_rotated
        k_sink_raw_expanded = k_sink_raw
        v_sink_expanded = v_sink

    o = flash_output.transpose(1, 2).float()  # [B, H, S, D]
    lse = softmax_lse.float()  # [B, H, S]

    B, H, S, D = o.shape

    inv_freq = rope_config["inv_freq"].to(device)
    attention_scaling = rope_config["attention_scaling"]
    pos_q = position_ids  # [B, S]

    cos_q, sin_q = compute_fp32_cos_sin(inv_freq, pos_q, attention_scaling)
    q_rot_fp32 = fp32_rope_rotate(q_raw, cos_q, sin_q)  # [B, H, S, D]

    old_probs = []
    new_logits = []
    v_list = []

    for j in range(K):
        k_bf16_sink_j = k_bf16_rot_expanded[:, :, j, :]  # [B, H, D]
        a_old_j = torch.einsum(
            "bhsd,bhd->bhs", q_bf16_rotated.float(), k_bf16_sink_j.float()
        ) * scale
        a_old_j[:, :, :j] = float('-inf')
        p_old_j = torch.exp(a_old_j - lse)
        p_old_j[:, :, :j] = 0.0
        old_probs.append(p_old_j)

        pos_k_j = position_ids[:, j:j+1]  # [B, 1]
        cos_k_j, sin_k_j = compute_fp32_cos_sin(inv_freq, pos_k_j, attention_scaling)
        k_raw_j = k_sink_raw_expanded[:, :, j:j+1, :]  # [B, H, 1, D]
        k_rot_fp32_j = fp32_rope_rotate(k_raw_j, cos_k_j, sin_k_j).squeeze(2)  # [B, H, D]
        a_new_j = torch.einsum("bhsd,bhd->bhs", q_rot_fp32, k_rot_fp32_j) * scale
        a_new_j[:, :, :j] = float('-inf')
        new_logits.append(a_new_j)

        v_list.append(v_sink_expanded[:, :, j, :].float())  # [B, H, D]

    p_old_sum = torch.stack(old_probs, dim=0).sum(dim=0)  # [B, H, S]
    p_old_sum = p_old_sum.clamp(max=1.0 - 1e-6)

    logZ_minus = lse + torch.log1p(-p_old_sum)

    new_logits_t = torch.stack(new_logits, dim=0)  # [K, B, H, S]
    lse_new_keys = torch.logsumexp(new_logits_t, dim=0)  # [B, H, S]

    lse_prime = torch.logaddexp(logZ_minus, lse_new_keys)

    scale_factor = torch.exp(lse - lse_prime)  # [B, H, S]

    weighted_v_old = torch.zeros(B, H, S, D, device=device, dtype=torch.float32)
    for j in range(K):
        weighted_v_old += old_probs[j].unsqueeze(-1) * v_list[j].unsqueeze(2)

    weighted_v_new = torch.zeros(B, H, S, D, device=device, dtype=torch.float32)
    for j in range(K):
        p_new_j = torch.exp(new_logits[j] - lse_prime)
        p_new_j[:, :, :j] = 0.0
        weighted_v_new += p_new_j.unsqueeze(-1) * v_list[j].unsqueeze(2)

    o = scale_factor.unsqueeze(-1) * (o - weighted_v_old) + weighted_v_new

    corrected = o.transpose(1, 2).to(flash_output.dtype)
    return corrected
