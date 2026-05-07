# Offline Contextual Agreement (CA) profiling for FCBoost.
# Hooks into Qwen3-8B attention layers to capture post-RoPE Q,K tensors,
# then computes CA metric per (layer, query_head, RoPE_pair) following FASA's definition.
# CA measures how well a single RoPE frequency pair's attention pattern agrees
# with the full-head attention pattern via top-K overlap.
#
# Optimized: uses fully vectorized tensor ops for top-K overlap (no Python set loops).

import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


def topk_overlap_vectorized(full_topk_idx: torch.Tensor, fc_topk_idx: torch.Tensor, context_len: int) -> torch.Tensor:
    """Compute top-K overlap between full attention and per-FC attention using scatter.

    Args:
        full_topk_idx: [H, K] indices of top-K positions from full attention
        fc_topk_idx: [H, F, K] indices of top-K from each FC pair
        context_len: total context length (for scatter dimension)

    Returns:
        overlap: [H, F] number of overlapping indices / K
    """
    H, K = full_topk_idx.shape
    F = fc_topk_idx.shape[1]

    indicator = torch.zeros(H, context_len, device=full_topk_idx.device, dtype=torch.int8)
    indicator.scatter_(1, full_topk_idx, 1)

    indicator_expanded = indicator.unsqueeze(1).expand(H, F, context_len)
    overlap_count = indicator_expanded.gather(2, fc_topk_idx).sum(dim=-1).float()

    return overlap_count / K


def compute_ca_scores_for_layer_fast(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    num_kv_groups: int,
    topk: int = 256,
    head_dim: int = 128,
    max_sample_tokens: int = 256,
) -> torch.Tensor:
    """Fast vectorized CA computation for one layer.

    Args:
        query_states: [B, num_q_heads, T, D] post-RoPE query
        key_states: [B, num_kv_heads, T, D] post-RoPE key
        num_kv_groups: GQA ratio
        topk: K for top-K overlap
        head_dim: head dimension
        max_sample_tokens: max token positions to sample

    Returns:
        ca_scores: [num_q_heads, num_fc_pairs]
    """
    B, num_q_heads, T, D = query_states.shape
    num_fc_pairs = D // 2
    scaling = head_dim ** -0.5

    key_expanded = key_states.repeat_interleave(num_kv_groups, dim=1)

    min_ctx = topk + 1
    if T <= min_ctx:
        return torch.zeros(num_q_heads, num_fc_pairs, device=query_states.device)

    valid_positions = list(range(min_ctx, T))
    if len(valid_positions) > max_sample_tokens:
        step = len(valid_positions) // max_sample_tokens
        valid_positions = valid_positions[::step][:max_sample_tokens]

    ca_accum = torch.zeros(num_q_heads, num_fc_pairs, device=query_states.device)

    for t in valid_positions:
        q_t = query_states[0, :, t, :]
        K_past = key_expanded[0, :, :t, :]

        alpha_full = torch.matmul(q_t.unsqueeze(1), K_past.transpose(1, 2)).squeeze(1) * scaling

        actual_k = min(topk, t)
        _, full_topk_idx = alpha_full.topk(actual_k, dim=-1)

        q_t_reshaped = q_t.view(num_q_heads, num_fc_pairs, 2)
        K_past_reshaped = K_past.view(num_q_heads, t, num_fc_pairs, 2)
        alpha_fc = torch.einsum('hfi,htfi->hft', q_t_reshaped, K_past_reshaped)

        _, fc_topk_idx = alpha_fc.topk(actual_k, dim=-1)

        overlap = topk_overlap_vectorized(full_topk_idx, fc_topk_idx, t)
        ca_accum += overlap

    ca_accum /= len(valid_positions)
    return ca_accum


class CAProfiler:
    """Runs CA profiling on a model with calibration data."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        topk: int = 256,
        max_sample_tokens: int = 256,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.topk = topk
        self.max_sample_tokens = max_sample_tokens
        self.device = device
        self.captured_qk = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        captured = self.captured_qk

        def hook_fn(module, args, kwargs, output):
            q = module._fcboost_q_post_rope
            k = module._fcboost_k_post_rope
            captured[layer_idx] = (q.detach().cpu(), k.detach().cpu())
            del module._fcboost_q_post_rope
            del module._fcboost_k_post_rope
            return output

        return hook_fn

    def _patch_attention_forward(self, model):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as orig_rope

        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            original_forward = attn.forward.__func__

            def make_patched_forward(orig_fwd, li):
                def patched_forward(self_attn, hidden_states, position_embeddings, attention_mask=None,
                                    past_key_values=None, cache_position=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self_attn.head_dim)

                    query_states = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

                    cos, sin = position_embeddings
                    query_states, key_states = orig_rope(query_states, key_states, cos, sin)

                    self_attn._fcboost_q_post_rope = query_states.detach().clone()
                    self_attn._fcboost_k_post_rope = key_states.detach().clone()

                    return orig_fwd(self_attn, hidden_states, position_embeddings, attention_mask,
                                    past_key_values, cache_position, **kwargs)
                return patched_forward

            attn.forward = make_patched_forward(original_forward, layer_idx).__get__(attn)

            hook = attn.register_forward_hook(self._make_hook(layer_idx), with_kwargs=True)
            self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def profile(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        calibration_texts: list[str],
        max_seq_len: int = 4096,
    ) -> np.ndarray:
        """Run CA profiling on calibration texts.

        Returns:
            ca_scores: np.ndarray of shape [num_layers, num_q_heads, num_fc_pairs]
        """
        config = model.config
        num_layers = config.num_hidden_layers
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads
        head_dim = config.head_dim
        num_fc_pairs = head_dim // 2

        all_ca = np.zeros((num_layers, num_q_heads, num_fc_pairs), dtype=np.float64)
        num_sequences = 0

        self._patch_attention_forward(model)

        try:
            for text_idx, text in enumerate(calibration_texts):
                print(f"Processing calibration sequence {text_idx + 1}/{len(calibration_texts)}...")

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_seq_len,
                    truncation=True,
                ).to(self.device)

                seq_len = inputs["input_ids"].shape[1]
                print(f"  Sequence length: {seq_len} tokens")

                if seq_len <= self.topk + 1:
                    print(f"  Skipping: too short (need > {self.topk + 1} tokens)")
                    continue

                self.captured_qk.clear()

                with torch.no_grad():
                    model(**inputs, use_cache=False)

                print(f"  Forward pass complete. Captured {len(self.captured_qk)} layers.")

                import time
                for layer_idx in range(num_layers):
                    if layer_idx not in self.captured_qk:
                        print(f"  WARNING: Layer {layer_idx} not captured")
                        continue

                    t0 = time.time()
                    q, k = self.captured_qk[layer_idx]
                    q = q.to(self.device)
                    k = k.to(self.device)

                    ca = compute_ca_scores_for_layer_fast(
                        q, k,
                        num_kv_groups=num_kv_groups,
                        topk=self.topk,
                        head_dim=head_dim,
                        max_sample_tokens=self.max_sample_tokens,
                    )

                    all_ca[layer_idx] += ca.cpu().numpy()

                    del q, k
                    torch.cuda.empty_cache()

                    elapsed = time.time() - t0
                    if (layer_idx + 1) % 6 == 0 or layer_idx == 0:
                        print(f"  Layer {layer_idx + 1}/{num_layers} ({elapsed:.1f}s)")

                self.captured_qk.clear()
                torch.cuda.empty_cache()
                num_sequences += 1

        finally:
            self.remove_hooks()

        if num_sequences > 0:
            all_ca /= num_sequences

        print(f"\nCA profiling complete. Processed {num_sequences} sequences.")
        print(f"CA scores shape: {all_ca.shape}")

        return all_ca
