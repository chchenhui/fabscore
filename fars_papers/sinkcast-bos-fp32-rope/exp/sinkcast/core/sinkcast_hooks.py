# SinkCast hook infrastructure that preserves HF's exact attention computation.
# Instead of replacing HF's attention with direct flash_attn_func calls (which
# changes d=0 accuracy), this hooks INTO HF's pipeline by:
# 1. Capturing pre-RoPE Q_raw, K_raw, V from projection outputs
# 2. Monkey-patching flash_attn functions to also capture softmax_lse
# 3. Capturing pre-o_proj attention output via o_proj input hook
# 4. Letting HF's original forward run (identical BF16 output as baseline)
# 5. Applying SinkCast correction only to real tokens, re-running o_proj
# This ensures d=0 accuracy is identical between BF16 baseline and SinkCast.

from contextlib import contextmanager

import torch
import transformers.modeling_flash_attention_utils as fau
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from sinkcast.core.sinkcast import sinkcast_correct


def find_attention_layers(model: PreTrainedModel):
    layers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                layers.append(layer.self_attn)
    return layers


class _LSECapture:
    def __init__(self):
        self.lse = None
        self.indices_q = None
        self.batch_size = None
        self.query_length = None
        self.is_varlen = False

    def clear(self):
        self.lse = None
        self.indices_q = None
        self.batch_size = None
        self.query_length = None
        self.is_varlen = False


def _make_flash_fn_wrapper(original_fn, capture: _LSECapture, is_varlen: bool):
    def wrapper(*args, **kwargs):
        kwargs["return_attn_probs"] = True
        result = original_fn(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            capture.lse = result[1]
            capture.is_varlen = is_varlen
        return result
    return wrapper


def _make_upad_wrapper(original_fn, capture: _LSECapture):
    def wrapper(query_layer, key_layer, value_layer, attention_mask,
                query_length, unpad_input_func):
        result = original_fn(query_layer, key_layer, value_layer,
                             attention_mask, query_length, unpad_input_func)
        _q_out, _k_out, _v_out, indices_q, _cu_seqlens, _max_seqlens = result
        capture.indices_q = indices_q
        capture.batch_size = query_layer.shape[0]
        capture.query_length = query_layer.shape[1]
        return result
    return wrapper


def _get_real_token_lse(capture: _LSECapture) -> torch.Tensor:
    """Get softmax_lse for real tokens only, shape [B, H, T_real].
    flash_attn_func returns [B, H, S].
    flash_attn_varlen_func returns [H, total_q] -- need to reshape to [1, H, T]."""
    lse = capture.lse
    if not capture.is_varlen or capture.indices_q is None:
        return lse
    return lse.unsqueeze(0)


@contextmanager
def sinkcast_hooks(attn_layers: list, rope_config: dict,
                   position_ids_ref: list, real_mask_ref: list,
                   K: int = 1):
    """Context manager that applies SinkCast correction while preserving
    HF's exact attention path. d=0 output is identical to vanilla BF16.

    Args:
        attn_layers: list of attention modules from find_attention_layers()
        rope_config: from extract_rope_config()
        position_ids_ref: [tensor] mutable ref; position_ids for REAL tokens only [1, T_real]
        real_mask_ref: [tensor or None] mutable ref; bool mask [1, S_full] True=real, or [None]
        K: number of sink keys to correct
    """
    capture = _LSECapture()
    q_raw_store = {}
    k_raw_store = {}
    v_store = {}
    pre_oproj_store = {}
    handles_fwd = []
    proj_handles = []

    if fau._flash_fn is None or fau._flash_varlen_fn is None:
        from flash_attn import flash_attn_func as _fa_func, flash_attn_varlen_func as _fa_varlen
        if fau._flash_fn is None:
            fau._flash_fn = _fa_func
        if fau._flash_varlen_fn is None:
            fau._flash_varlen_fn = _fa_varlen

    orig_flash_fn = fau._flash_fn
    orig_flash_varlen_fn = fau._flash_varlen_fn
    orig_upad = fau._upad_input

    fau._flash_fn = _make_flash_fn_wrapper(orig_flash_fn, capture, is_varlen=False)
    fau._flash_varlen_fn = _make_flash_fn_wrapper(orig_flash_varlen_fn, capture, is_varlen=True)
    fau._upad_input = _make_upad_wrapper(orig_upad, capture)

    for layer_idx, attn in enumerate(attn_layers):
        original_forward = attn.forward

        def _q_hook(module, inp, output, idx=layer_idx):
            q_raw_store[idx] = output.detach().clone()

        def _k_hook(module, inp, output, idx=layer_idx):
            k_raw_store[idx] = output.detach().clone()

        def _v_hook(module, inp, output, idx=layer_idx):
            v_store[idx] = output.detach().clone()

        def _oproj_input_hook(module, inp, idx=layer_idx):
            pre_oproj_store[idx] = inp[0].detach().clone()

        h_q = attn.q_proj.register_forward_hook(_q_hook)
        h_k = attn.k_proj.register_forward_hook(_k_hook)
        h_v = attn.v_proj.register_forward_hook(_v_hook)
        h_op = attn.o_proj.register_forward_pre_hook(_oproj_input_hook)
        proj_handles.extend([h_q, h_k, h_v, h_op])

        def make_hook(idx, orig_fwd, attn_module):
            def hooked_forward(hidden_states, position_embeddings,
                               attention_mask, **kwargs):
                seq_len = hidden_states.shape[1]
                if seq_len <= 1:
                    return orig_fwd(hidden_states, position_embeddings,
                                    attention_mask, **kwargs)

                capture.clear()

                hf_attn_out, attn_weights = orig_fwd(
                    hidden_states, position_embeddings,
                    attention_mask, **kwargs
                )

                if capture.lse is None:
                    return hf_attn_out, attn_weights

                input_shape = hidden_states.shape[:-1]
                B, S_full = input_shape

                q_raw = q_raw_store.get(idx)
                k_raw = k_raw_store.get(idx)
                v_raw = v_store.get(idx)
                pre_oproj = pre_oproj_store.get(idx)
                if q_raw is None or k_raw is None or v_raw is None or pre_oproj is None:
                    return hf_attn_out, attn_weights

                q_shape = (*input_shape, -1, attn_module.head_dim)
                k_shape = (*input_shape, -1, attn_module.head_dim)
                q_raw_4d = q_raw.view(q_shape).transpose(1, 2)
                k_raw_4d = k_raw.view(k_shape).transpose(1, 2)
                v_raw_4d = v_raw.view(k_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q_rot, k_rot = apply_rotary_pos_emb(q_raw_4d, k_raw_4d, cos, sin)

                num_heads_q = q_rot.shape[1]

                rmask = real_mask_ref[0]
                if rmask is not None and not rmask.all():
                    ridx = rmask[0].nonzero(as_tuple=True)[0]
                    q_rot_r = q_rot[:, :, ridx, :]
                    k_rot_r = k_rot[:, :, ridx, :]
                    q_raw_r = q_raw_4d[:, :, ridx, :]
                    k_raw_r = k_raw_4d[:, :, ridx, :]
                    v_raw_r = v_raw_4d[:, :, ridx, :]

                    pre_oproj_full = pre_oproj.view(B, S_full, num_heads_q,
                                                     attn_module.head_dim)
                    flash_output_r = pre_oproj_full[:, ridx, :, :]
                else:
                    ridx = None
                    q_rot_r = q_rot
                    k_rot_r = k_rot
                    q_raw_r = q_raw_4d
                    k_raw_r = k_raw_4d
                    v_raw_r = v_raw_4d
                    flash_output_r = pre_oproj.view(B, S_full, num_heads_q,
                                                     attn_module.head_dim)

                lse_real = _get_real_token_lse(capture)

                pos_ids = position_ids_ref[0]
                actual_K = min(K, k_raw_r.shape[2])
                k_sink_raw = k_raw_r[:, :, 0:actual_K, :].detach()
                v_sink = v_raw_r[:, :, 0:actual_K, :].detach()

                corrected_r = sinkcast_correct(
                    flash_output=flash_output_r,
                    softmax_lse=lse_real,
                    q_bf16_rotated=q_rot_r,
                    k_bf16_rotated=k_rot_r,
                    q_raw=q_raw_r,
                    k_sink_raw=k_sink_raw,
                    v_sink=v_sink,
                    position_ids=pos_ids,
                    rope_config=rope_config,
                    K=actual_K,
                )

                if ridx is not None:
                    full_corr = pre_oproj.view(B, S_full, num_heads_q,
                                               attn_module.head_dim).clone()
                    full_corr[:, ridx, :, :] = corrected_r.to(full_corr.dtype)
                    corrected_flat = full_corr.reshape(B, S_full, -1).contiguous()
                else:
                    corrected_flat = corrected_r.reshape(B, S_full, -1).contiguous()

                corrected_out = attn_module.o_proj(corrected_flat)
                return corrected_out, attn_weights

            return hooked_forward

        attn.forward = make_hook(layer_idx, original_forward, attn)
        handles_fwd.append((attn, original_forward))

    try:
        yield
    finally:
        fau._flash_fn = orig_flash_fn
        fau._flash_varlen_fn = orig_flash_varlen_fn
        fau._upad_input = orig_upad
        for h in proj_handles:
            h.remove()
        for attn_obj, orig_fwd in handles_fwd:
            attn_obj.forward = orig_fwd
