# FP32 RoPE recomputation utilities for SinkCast correction.
# Provides functions to extract RoPE config from a model and recompute
# RoPE-rotated dot products entirely in FP32 for sink-key logit correction.

import math
from typing import Dict, Tuple

import torch
from transformers import PreTrainedModel


def extract_rope_config(model: PreTrainedModel) -> Dict:
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    rotary_emb = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        first_attn = model.model.layers[0].self_attn
        if hasattr(first_attn, "rotary_emb"):
            rotary_emb = first_attn.rotary_emb

    inv_freq = rotary_emb.inv_freq.float().clone() if rotary_emb is not None else None
    attention_scaling = getattr(rotary_emb, "attention_scaling", 1.0) if rotary_emb is not None else 1.0
    rope_theta = getattr(config, "rope_theta", 10000.0)

    return {
        "inv_freq": inv_freq,
        "rope_theta": rope_theta,
        "head_dim": head_dim,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "attention_scaling": attention_scaling,
    }


def compute_fp32_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def fp32_rope_rotate(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def fp32_rope_dot(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    rope_config: Dict,
) -> torch.Tensor:
    device = q_raw.device
    inv_freq = rope_config["inv_freq"].to(device)
    attention_scaling = rope_config["attention_scaling"]
    head_dim = rope_config["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)

    cos_q, sin_q = compute_fp32_cos_sin(inv_freq, pos_q, attention_scaling)
    cos_k, sin_k = compute_fp32_cos_sin(inv_freq, pos_k, attention_scaling)

    q_rot = fp32_rope_rotate(q_raw, cos_q, sin_q)
    k_rot = fp32_rope_rotate(k_raw, cos_k, sin_k)

    num_heads = q_rot.shape[1]
    num_kv_heads = k_rot.shape[1]
    if num_kv_heads < num_heads:
        num_groups = num_heads // num_kv_heads
        k_rot = k_rot.repeat_interleave(num_groups, dim=1)

    logits = torch.einsum("bhsd,bhkd->bhsk", q_rot.float(), k_rot.float()) * scale
    return logits
