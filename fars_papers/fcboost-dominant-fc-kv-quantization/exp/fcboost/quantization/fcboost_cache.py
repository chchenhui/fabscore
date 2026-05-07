# FCBoost KV Cache: extends Kitty's KittyKVCache with static CA-derived channel masks.
# Replaces dynamic per-page magnitude-based channel selection with precomputed
# static RoPE-frequency masks. Optionally applies the same mask to value cache
# for mixed-precision value quantization (boost_values=True).

import torch
from typing import Optional, Any
from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.utils_quant import fake_quant_groupwise_lastdim


def _quant_value_with_promote(value_slice: torch.Tensor, group_size: int,
                               vbits: int, promote_mask: torch.Tensor,
                               promote_bit: int) -> torch.Tensor:
    """Quantize value cache with per-channel mixed precision.
    Processes in chunks of group_size tokens, transposing each chunk to
    [B, nh, D, group_size] for channel-level promote_mask application."""
    B, nh, T, D = value_slice.shape
    num_full_chunks = T // group_size
    remainder = T % group_size
    result = value_slice.clone()
    for i in range(num_full_chunks):
        start = i * group_size
        end = start + group_size
        chunk = result[:, :, start:end, :].transpose(2, 3).contiguous()
        chunk = fake_quant_groupwise_lastdim(chunk, group_size, vbits, promote_mask, promote_bit)
        result[:, :, start:end, :] = chunk.transpose(2, 3).contiguous()
    if remainder > 0:
        start = num_full_chunks * group_size
        for t in range(start, T):
            token = result[:, :, t:t+1, :]
            result[:, :, t:t+1, :] = _quant_single_token_value_with_promote(
                token, vbits, promote_mask, promote_bit)
    return result


def _quant_single_token_value_with_promote(value_slice: torch.Tensor, vbits: int,
                                            promote_mask: torch.Tensor,
                                            promote_bit: int) -> torch.Tensor:
    """Quantize a single-token value [B, nh, 1, D] with per-channel mixed precision.
    Uses simple min-max quantization per token since group_size constraints
    prevent using the standard transposed path for T=1."""
    B, nh, T, D = value_slice.shape
    mask_expanded = promote_mask.unsqueeze(2)
    mn = value_slice.min(dim=-1, keepdim=True).values
    mx = value_slice.max(dim=-1, keepdim=True).values
    eps = 1e-4 if value_slice.dtype in (torch.float16, torch.bfloat16) else 1e-6
    scale_base = (mx - mn).clamp(min=eps) / (2 ** vbits - 1)
    scale_promote = (mx - mn).clamp(min=eps) / (2 ** promote_bit - 1)
    scale = torch.where(mask_expanded, scale_promote, scale_base)
    max_val = torch.where(
        mask_expanded,
        torch.full_like(scale, 2 ** promote_bit - 1),
        torch.full_like(scale, 2 ** vbits - 1)
    )
    q = ((value_slice - mn) / scale).clamp(torch.zeros_like(max_val), max_val).round()
    return (q * scale + mn).to(value_slice.dtype)


class FCBoostKVCache(KittyKVCache):
    """KittyKVCache variant that uses static CA-derived channel masks
    instead of dynamic magnitude-based selection.
    When boost_values=True, also applies mixed-precision to value cache."""

    def __init__(self, cache_config: KittyKVCacheConfig, mask_path: str,
                 boost_values: bool = False):
        super().__init__(cache_config=cache_config)
        self.boost_values = boost_values
        self.static_masks = torch.load(mask_path, map_location="cpu", weights_only=True)
        num_layers = len(self.static_masks)
        sample_mask = self.static_masks[0]
        print(f"[FCBoost] Loaded static masks from {mask_path}")
        print(f"  Layers: {num_layers}, KV heads: {sample_mask.shape[0]}, "
              f"Head dim: {sample_mask.shape[1]}, "
              f"Boosted channels/head: {sample_mask[0].sum().item()}, "
              f"boost_values: {boost_values}")

    def _get_static_promote_mask(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get the static promote mask for a given layer.

        Returns:
            promote_mask: [B, num_kv_heads, D] bool tensor
        """
        mask = self.static_masks[layer_idx].to(device=device)
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")

        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states.detach().clone())
            self.value_cache.append(value_states.detach().clone())
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            assert current_cache_length > self.sink_length, \
                "Kitty-KV: sequence length must be greater than sink_length"

            if current_cache_length > self.sink_length + self.buffer_length:
                start_idx = self.sink_length
                num_tokens = current_cache_length - self.sink_length
                num_token_to_buffer = num_tokens % self.buffer_length
                num_token_to_quantize = num_tokens - num_token_to_buffer
                end_idx = start_idx + num_token_to_quantize

                B = current_key_cache.shape[0]
                device = current_key_cache.device
                promote_mask = self._get_static_promote_mask(layer_idx, B, device)

                for idx in range(start_idx, end_idx, self.buffer_length):
                    key_slice = current_key_cache[:, :, idx:idx+self.buffer_length, :].transpose(2, 3).contiguous()
                    key_slice = fake_quant_groupwise_lastdim(
                        key_slice, self.group_size, self.kbits, promote_mask, self.promote_bit
                    ).transpose(2, 3).contiguous()
                    current_key_cache[:, :, idx:idx+self.buffer_length, :] = key_slice

                if not self.VCache_BitDecoding:
                    num_token_to_quantize = num_tokens - self.buffer_length
                    end_idx = start_idx + num_token_to_quantize
                value_slice = current_value_cache[:, :, start_idx:end_idx, :]
                if self.boost_values:
                    value_slice = _quant_value_with_promote(
                        value_slice, self.group_size, self.vbits, promote_mask, self.promote_bit)
                else:
                    value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                current_value_cache[:, :, start_idx:end_idx, :] = value_slice

        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            num_tokens_kv_to_quantize = current_cache_length - self.sink_length - self.buffer_length
            B = current_key_cache.shape[0]
            device = current_key_cache.device
            promote_mask = self._get_static_promote_mask(layer_idx, B, device)

            if num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):
                key_slice = current_key_cache[:, :, -self.buffer_length-1:-1, :]
                key_slice = fake_quant_groupwise_lastdim(
                    key_slice.transpose(2, 3).contiguous(),
                    self.group_size, self.kbits, promote_mask, self.promote_bit
                ).transpose(2, 3).contiguous()
                current_key_cache[:, :, -self.buffer_length-1:-1, :] = key_slice

                if self.VCache_BitDecoding:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-1, :]
                    if self.boost_values:
                        value_slice = _quant_value_with_promote(
                            value_slice, self.group_size, self.vbits, promote_mask, self.promote_bit)
                    else:
                        value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-1, :] = value_slice

            if not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :]
                    if self.boost_values:
                        value_slice = _quant_single_token_value_with_promote(
                            value_slice, self.vbits, promote_mask, self.promote_bit)
                    else:
                        value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :] = value_slice

        if self.PostQuant:
            return keys_to_return, values_to_return
        else:
            return current_key_cache, current_value_cache
