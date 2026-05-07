# Standard BF16 FlashAttention inference baseline.
# Provides forward pass with explicit position_ids offsets, and a hook-based
# variant that captures post-RoPE Q/K to compute per-key attention logits
# for shift-error analysis.

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel


def bf16_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    position_ids = (torch.arange(seq_len, device=device) + position_offset).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, position_ids=position_ids)
    return outputs.logits


def _find_attention_layers(model: PreTrainedModel):
    layers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                layers.append(layer.self_attn)
    return layers


@contextmanager
def _capture_qk_hooks(attn_layers: list):
    buffers: Dict[int, Dict[str, torch.Tensor]] = {}
    handles = []

    for layer_idx, attn in enumerate(attn_layers):
        original_forward = attn.forward

        def make_hook(idx, orig_fwd):
            def hooked_forward(hidden_states, position_embeddings, attention_mask, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)

                q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                cos, sin = position_embeddings
                q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

                buffers[idx] = {"q": q_rot.detach(), "k": k_rot.detach()}

                return orig_fwd(hidden_states, position_embeddings, attention_mask, **kwargs)
            return hooked_forward

        attn.forward = make_hook(layer_idx, original_forward)
        handles.append((attn, original_forward))

    try:
        yield buffers
    finally:
        for attn, orig_fwd in handles:
            attn.forward = orig_fwd


def bf16_forward_with_hooks(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    position_offset: int = 0,
    key_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict[int, Dict[int, torch.Tensor]]]:
    if key_indices is None:
        key_indices = [0, 1, 2, 8, 64]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    position_ids = (torch.arange(seq_len, device=device) + position_offset).unsqueeze(0)

    attn_layers = _find_attention_layers(model)
    assert len(attn_layers) > 0, "No attention layers found"

    with _capture_qk_hooks(attn_layers) as qk_buffers:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, position_ids=position_ids)
    logits = outputs.logits

    attn_logits = {}
    for layer_idx, qk in qk_buffers.items():
        q = qk["q"]  # [batch, num_heads, seq_len, head_dim]
        k = qk["k"]  # [batch, num_kv_heads, seq_len, head_dim]

        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            k = k.repeat_interleave(num_groups, dim=1)

        scale = 1.0 / math.sqrt(q.shape[-1])
        layer_logits = {}
        for j in key_indices:
            if j >= seq_len:
                continue
            k_j = k[:, :, j, :]  # [batch, num_heads, head_dim]
            a_ij = torch.einsum("bhid,bhd->bhi", q, k_j) * scale  # [batch, heads, seq_len]
            layer_logits[j] = a_ij.detach().cpu()
        attn_logits[layer_idx] = layer_logits

    return logits, attn_logits
