# Collects per-page magnitude-based channel importance statistics from Qwen3-8B
# key cache, replicating Kitty's dynamic channel selection logic offline.
# Hooks into attention layers to capture post-RoPE key states, splits into pages
# of buffer_length tokens, and records which top-K channels are selected per page.
# Outputs: per-channel frequency counts, mean magnitudes, and per-RoPE-pair stats.

import argparse
import os
import sys
import time

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_calibration_texts(num_sequences: int, max_seq_len: int, tokenizer=None):
    from datasets import load_dataset

    print(f"Loading WikiText-2 calibration data ({num_sequences} sequences, max {max_seq_len} tokens)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([t for t in dataset["text"] if len(t.strip()) > 100])

    texts = []
    chunk_size_chars = max_seq_len * 5
    for start in range(0, len(all_text), chunk_size_chars):
        if len(texts) >= num_sequences:
            break
        chunk = all_text[start:start + chunk_size_chars]
        if len(chunk.strip()) < 500:
            continue
        if tokenizer is not None:
            toks = tokenizer(chunk, return_tensors="pt", max_length=max_seq_len, truncation=True)
            if toks["input_ids"].shape[1] < 512:
                continue
        texts.append(chunk)

    print(f"  Loaded {len(texts)} calibration sequences")
    return texts


class MagnitudeCollector:
    """Hooks into model attention layers to capture post-RoPE key states
    and compute Kitty-style magnitude-based channel importance per page."""

    def __init__(self, num_layers, num_kv_heads, head_dim, buffer_length=128, k_chan=16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.buffer_length = buffer_length
        self.k_chan = k_chan

        self.channel_freq = np.zeros((num_layers, num_kv_heads, head_dim), dtype=np.float64)
        self.channel_mag_sum = np.zeros((num_layers, num_kv_heads, head_dim), dtype=np.float64)
        self.total_pages = np.zeros((num_layers, num_kv_heads), dtype=np.int64)

        self.captured_keys = {}
        self.hooks = []

    def _make_hook(self, layer_idx):
        captured = self.captured_keys

        def hook_fn(module, args, kwargs, output):
            k = module._mag_k_post_rope
            captured[layer_idx] = k.detach().cpu()
            del module._mag_k_post_rope
            return output

        return hook_fn

    def patch_and_hook(self, model):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as orig_rope

        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            original_forward = attn.forward.__func__

            def make_patched_forward(orig_fwd, li):
                def patched_forward(self_attn, hidden_states, position_embeddings,
                                    attention_mask=None, past_key_values=None,
                                    cache_position=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self_attn.head_dim)

                    key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states = self_attn.q_norm(self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    query_states, key_states = orig_rope(query_states, key_states, cos, sin)

                    self_attn._mag_k_post_rope = key_states.detach().clone()

                    return orig_fwd(self_attn, hidden_states, position_embeddings,
                                    attention_mask, past_key_values, cache_position, **kwargs)
                return patched_forward

            attn.forward = make_patched_forward(original_forward, layer_idx).__get__(attn)
            hook = attn.register_forward_hook(self._make_hook(layer_idx), with_kwargs=True)
            self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def process_captured_keys(self):
        for layer_idx in range(self.num_layers):
            if layer_idx not in self.captured_keys:
                continue

            key_states = self.captured_keys[layer_idx]
            B, nh, T, D = key_states.shape

            num_pages = T // self.buffer_length
            if num_pages == 0:
                continue

            for page_idx in range(num_pages):
                start = page_idx * self.buffer_length
                end = start + self.buffer_length
                page = key_states[:, :, start:end, :]

                page_t = page.transpose(2, 3).contiguous()
                score = page_t.abs().mean(dim=-1)

                for b in range(B):
                    for h in range(nh):
                        scores_bh = score[b, h].numpy()
                        top_idx = np.argsort(scores_bh)[-self.k_chan:]

                        self.channel_freq[layer_idx, h, top_idx] += 1
                        self.channel_mag_sum[layer_idx, h] += scores_bh
                        self.total_pages[layer_idx, h] += 1

        self.captured_keys.clear()

    def get_results(self):
        total_pages_safe = np.maximum(self.total_pages, 1)

        per_channel_mean_mag = self.channel_mag_sum / total_pages_safe[:, :, np.newaxis]

        per_pair_mean_mag = np.zeros((self.num_layers, self.num_kv_heads, self.head_dim // 2), dtype=np.float64)
        for i in range(self.head_dim // 2):
            per_pair_mean_mag[:, :, i] = (per_channel_mean_mag[:, :, 2*i] + per_channel_mean_mag[:, :, 2*i+1]) / 2

        freq_mask = np.zeros((self.num_layers, self.num_kv_heads, self.head_dim), dtype=np.bool_)
        for l in range(self.num_layers):
            for h in range(self.num_kv_heads):
                top_idx = np.argsort(self.channel_freq[l, h])[-self.k_chan:]
                freq_mask[l, h, top_idx] = True

        return {
            "channel_freq": self.channel_freq,
            "per_channel_mean_magnitude": per_channel_mean_mag,
            "per_pair_mean_magnitude": per_pair_mean_mag,
            "freq_mask": freq_mask,
            "total_pages": self.total_pages,
        }


def main():
    parser = argparse.ArgumentParser(description="Magnitude-based channel statistics collection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_sequences", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--buffer_length", type=int, default=128)
    parser.add_argument("--k_chan", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="fcboost/analysis")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Quick run with 1 sequence, 512 tokens")
    args = parser.parse_args()

    if args.sanity_check:
        args.num_sequences = 1
        args.max_seq_len = 512
        print("=== SANITY CHECK MODE ===")

    print("=" * 80)
    print("Magnitude-Based Channel Statistics Collection")
    print(f"  Model: {args.model}")
    print(f"  Sequences: {args.num_sequences}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Buffer length (page size): {args.buffer_length}")
    print(f"  K channels to select: {args.k_chan}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calibration_texts = load_calibration_texts(
        num_sequences=args.num_sequences,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
    )

    if not calibration_texts:
        print("ERROR: No valid calibration sequences found")
        sys.exit(1)

    print(f"\nLoading model {args.model} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")

    collector = MagnitudeCollector(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        buffer_length=args.buffer_length,
        k_chan=args.k_chan,
    )
    collector.patch_and_hook(model)

    try:
        for text_idx, text in enumerate(calibration_texts):
            print(f"\nProcessing sequence {text_idx + 1}/{len(calibration_texts)}...")
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=args.max_seq_len, truncation=True,
            ).to(model.device)
            seq_len = inputs["input_ids"].shape[1]
            print(f"  Sequence length: {seq_len} tokens, ~{seq_len // args.buffer_length} pages")

            collector.captured_keys.clear()
            with torch.no_grad():
                model(**inputs, use_cache=False)

            t0 = time.time()
            collector.process_captured_keys()
            elapsed = time.time() - t0
            print(f"  Processed in {elapsed:.1f}s")
            torch.cuda.empty_cache()
    finally:
        collector.remove_hooks()

    results = collector.get_results()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "magnitude_stats.npz")
    np.savez(
        output_path,
        channel_freq=results["channel_freq"],
        per_channel_mean_magnitude=results["per_channel_mean_magnitude"],
        per_pair_mean_magnitude=results["per_pair_mean_magnitude"],
        freq_mask=results["freq_mask"],
        total_pages=results["total_pages"],
    )
    print(f"\nSaved magnitude statistics to {output_path}")

    print("\n=== Summary ===")
    print(f"channel_freq shape: {results['channel_freq'].shape}")
    print(f"per_channel_mean_magnitude shape: {results['per_channel_mean_magnitude'].shape}")
    print(f"per_pair_mean_magnitude shape: {results['per_pair_mean_magnitude'].shape}")
    print(f"freq_mask shape: {results['freq_mask'].shape}")
    print(f"total_pages (layer 0): {results['total_pages'][0]}")
    print(f"Channels selected per head in freq_mask: {results['freq_mask'][0, 0].sum()}")

    print(f"\nMagnitude value range: [{results['per_channel_mean_magnitude'].min():.6f}, "
          f"{results['per_channel_mean_magnitude'].max():.6f}]")
    print(f"Pair magnitude range: [{results['per_pair_mean_magnitude'].min():.6f}, "
          f"{results['per_pair_mean_magnitude'].max():.6f}]")
    print(f"Max channel freq: {results['channel_freq'].max():.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
