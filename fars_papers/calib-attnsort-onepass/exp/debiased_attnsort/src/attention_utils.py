# Attention extraction using two-step prefill+decode approach.
# Step 1: Prefill with flash-attention to build KV cache (memory-efficient, no attention output).
# Step 2: Switch to eager attention, run single decode token with output_attentions=True.
#   Since query_len=1, the attention tensor is only (batch, heads, 1, seq_len) -- small.
# Step 3: Switch back to flash attention for subsequent generation.
# This avoids the O(L^2) memory of full eager prefill while still extracting attention.

import torch
import numpy as np


def extract_first_token_attention(model, tokenizer, input_ids, doc_idx_ranges):
    """Extract per-document attention scores from the first generated token.

    Uses a two-step approach: flash-attention prefill for KV cache, then
    a single decode step with eager attention to get per-document scores.
    Temporarily switches model.config._attn_implementation to 'eager' for the
    decode step (query_len=1 so attention is only (batch, heads, 1, seq_len)),
    then restores to flash_attention_2.

    Args:
        model: HuggingFace CausalLM loaded with flash_attention_2.
        tokenizer: Corresponding tokenizer.
        input_ids: dict with 'input_ids' and 'attention_mask' tensors on model device.
        doc_idx_ranges: list of [start, end] token index pairs for each document.

    Returns:
        attention_masses: np.ndarray of shape (num_docs,) with raw attention mass per doc.
        doc_idx_ranges: the input doc_idx_ranges (passed through for convenience).
    """
    orig_impl = model.config._attn_implementation

    with torch.no_grad():
        prefill_out = model(
            **input_ids,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_kv = prefill_out.past_key_values
        next_token_logits = prefill_out.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

        model.config._attn_implementation = "eager"
        decode_out = model(
            input_ids=next_token_id,
            past_key_values=past_kv,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        model.config._attn_implementation = orig_impl

    decode_attentions = decode_out.attentions

    num_docs = len(doc_idx_ranges)
    attention_masses = np.zeros(num_docs, dtype=np.float64)

    for layer_attn in decode_attentions:
        attn = layer_attn[0, :, 0, :]  # (num_heads, key_len)
        for i, (start, end) in enumerate(doc_idx_ranges):
            attention_masses[i] += attn[:, start:end].sum().cpu().item()

    return attention_masses, doc_idx_ranges
