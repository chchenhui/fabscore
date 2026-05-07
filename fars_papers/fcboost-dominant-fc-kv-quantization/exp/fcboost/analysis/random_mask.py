# Generate random static masks for FCBoost ablation study.
# For each (layer, KV head), randomly selects F RoPE pairs and maps to boosted channels.
# Produces masks in same format as CA-derived masks for direct substitution.

import torch
import numpy as np
import os
import argparse


def generate_random_mask(
    num_layers: int = 36,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_rope_pairs: int = 64,
    top_f: int = 8,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    masks = {}
    for layer_idx in range(num_layers):
        layer_mask = torch.zeros(num_kv_heads, head_dim, dtype=torch.bool)
        for kv_h in range(num_kv_heads):
            selected_pairs = rng.choice(num_rope_pairs, size=top_f, replace=False)
            for pair_idx in selected_pairs:
                layer_mask[kv_h, pair_idx * 2] = True
                layer_mask[kv_h, pair_idx * 2 + 1] = True
        masks[layer_idx] = layer_mask
    return masks


def verify_mask(masks: dict, num_layers: int = 36, num_kv_heads: int = 8,
                head_dim: int = 128, top_f: int = 8) -> bool:
    assert len(masks) == num_layers, f"Expected {num_layers} layers, got {len(masks)}"
    for layer_idx in range(num_layers):
        assert layer_idx in masks, f"Missing layer {layer_idx}"
        m = masks[layer_idx]
        assert m.shape == (num_kv_heads, head_dim), f"Layer {layer_idx}: shape {m.shape}"
        assert m.dtype == torch.bool, f"Layer {layer_idx}: dtype {m.dtype}"
        for kv_h in range(num_kv_heads):
            count = m[kv_h].sum().item()
            assert count == top_f * 2, (
                f"Layer {layer_idx} KV head {kv_h}: expected {top_f*2} True, got {count}")
            for i in range(head_dim // 2):
                if m[kv_h, i * 2]:
                    assert m[kv_h, i * 2 + 1], (
                        f"Layer {layer_idx} KV head {kv_h}: pair {i} has channel {i*2} "
                        f"but not {i*2+1}")
                if m[kv_h, i * 2 + 1]:
                    assert m[kv_h, i * 2], (
                        f"Layer {layer_idx} KV head {kv_h}: pair {i} has channel {i*2+1} "
                        f"but not {i*2}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate random static masks for ablation")
    parser.add_argument("--output_dir", type=str, default="fcboost/masks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--num_layers", type=int, default=36)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--top_f", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_masks = {}

    for seed in args.seeds:
        masks = generate_random_mask(
            num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim, top_f=args.top_f, seed=seed)

        verify_mask(masks, num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim, top_f=args.top_f)

        path = os.path.join(args.output_dir, f"qwen3_8b_random_mask_seed{seed}.pt")
        torch.save(masks, path)
        all_masks[seed] = masks
        print(f"[seed={seed}] Saved random mask to {path}")
        print(f"  Layers: {args.num_layers}, KV heads: {args.num_kv_heads}, "
              f"Boosted channels/head: {args.top_f * 2}")

    seeds = list(all_masks.keys())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            m_i = all_masks[seeds[i]]
            m_j = all_masks[seeds[j]]
            identical = all(torch.equal(m_i[l], m_j[l]) for l in range(args.num_layers))
            assert not identical, f"Masks for seeds {seeds[i]} and {seeds[j]} are identical!"
            overlap = sum(
                (m_i[l] & m_j[l]).sum().item() for l in range(args.num_layers)
            )
            total = sum(m_i[l].sum().item() for l in range(args.num_layers))
            print(f"[seed {seeds[i]} vs {seeds[j]}] Overlap: {overlap}/{total} "
                  f"({100*overlap/total:.1f}%)")

    print("\nAll masks generated and verified successfully.")


if __name__ == "__main__":
    main()
