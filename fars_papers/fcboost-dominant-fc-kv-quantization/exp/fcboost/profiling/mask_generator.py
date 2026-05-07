# GQA aggregation and static mask generation for FCBoost.
# Aggregates per-query-head CA scores to per-KV-head scores,
# selects top-F RoPE pairs, and generates binary channel masks.

import torch
import numpy as np
import os
from pathlib import Path


def aggregate_ca_for_gqa(
    ca_scores: np.ndarray,
    num_kv_heads: int = 8,
) -> np.ndarray:
    """Aggregate per-query-head CA scores to per-KV-head scores.

    Args:
        ca_scores: [num_layers, num_q_heads, num_fc_pairs]
        num_kv_heads: number of KV heads (8 for Qwen3-8B)

    Returns:
        kv_ca_scores: [num_layers, num_kv_heads, num_fc_pairs]
    """
    num_layers, num_q_heads, num_fc_pairs = ca_scores.shape
    num_kv_groups = num_q_heads // num_kv_heads

    kv_ca_scores = np.zeros((num_layers, num_kv_heads, num_fc_pairs), dtype=np.float64)

    for kv_h in range(num_kv_heads):
        q_start = kv_h * num_kv_groups
        q_end = q_start + num_kv_groups
        kv_ca_scores[:, kv_h, :] = ca_scores[:, q_start:q_end, :].mean(axis=1)

    return kv_ca_scores


def generate_masks(
    kv_ca_scores: np.ndarray,
    top_f: int = 8,
    head_dim: int = 128,
) -> dict:
    """Generate per-(layer, KV head) binary channel masks.

    Args:
        kv_ca_scores: [num_layers, num_kv_heads, num_fc_pairs]
        top_f: number of RoPE pairs to boost per KV head
        head_dim: dimension of each head

    Returns:
        masks: dict with masks[layer_idx] = tensor [num_kv_heads, head_dim] bool
    """
    num_layers, num_kv_heads, num_fc_pairs = kv_ca_scores.shape
    masks = {}

    for layer_idx in range(num_layers):
        layer_mask = torch.zeros(num_kv_heads, head_dim, dtype=torch.bool)

        for kv_h in range(num_kv_heads):
            scores = kv_ca_scores[layer_idx, kv_h]
            top_indices = np.argsort(scores)[-top_f:]

            for fc_i in top_indices:
                layer_mask[kv_h, fc_i * 2] = True
                layer_mask[kv_h, fc_i * 2 + 1] = True

        masks[layer_idx] = layer_mask

    return masks


def save_profiling_results(
    ca_scores_qh: np.ndarray,
    kv_ca_scores: np.ndarray,
    masks: dict,
    output_dir: str,
    top_f: int = 8,
):
    """Save CA scores and masks to disk.

    Args:
        ca_scores_qh: [num_layers, num_q_heads, num_fc_pairs] per-query-head scores
        kv_ca_scores: [num_layers, num_kv_heads, num_fc_pairs] aggregated scores
        masks: dict from generate_masks
        output_dir: directory to save files
        top_f: number of boosted RoPE pairs (for logging)
    """
    os.makedirs(output_dir, exist_ok=True)

    scores_path = os.path.join(output_dir, "qwen3_8b_ca_scores.npy")
    np.save(scores_path, kv_ca_scores)
    print(f"Saved KV-head CA scores: {scores_path} (shape {kv_ca_scores.shape})")

    qh_scores_path = os.path.join(output_dir, "qwen3_8b_ca_scores_qh.npy")
    np.save(qh_scores_path, ca_scores_qh)
    print(f"Saved query-head CA scores: {qh_scores_path} (shape {ca_scores_qh.shape})")

    masks_path = os.path.join(output_dir, "qwen3_8b_ca_masks.pt")
    torch.save(masks, masks_path)
    print(f"Saved masks: {masks_path}")

    num_layers = kv_ca_scores.shape[0]
    num_kv_heads = kv_ca_scores.shape[1]

    print(f"\nDominant FC indices (top-{top_f} RoPE pairs per KV head):")
    print("=" * 80)
    for layer_idx in range(num_layers):
        dominant_pairs = []
        for kv_h in range(num_kv_heads):
            scores = kv_ca_scores[layer_idx, kv_h]
            top_indices = np.argsort(scores)[-top_f:][::-1]
            dominant_pairs.append(top_indices.tolist())

        if layer_idx < 4 or layer_idx >= num_layers - 2 or layer_idx % 6 == 0:
            print(f"Layer {layer_idx:2d}: {dominant_pairs}")
    print("=" * 80)

    boosted_per_layer = {}
    for layer_idx in range(num_layers):
        mask = masks[layer_idx]
        total_boosted = mask.sum().item()
        boosted_per_layer[layer_idx] = total_boosted
    print(f"\nBoosted channels per layer (should be {num_kv_heads * top_f * 2} = {num_kv_heads}*{top_f}*2):")
    for li, count in boosted_per_layer.items():
        if count != num_kv_heads * top_f * 2:
            print(f"  Layer {li}: {count} (UNEXPECTED)")
    print("All layers have expected boosted channel count." if all(
        v == num_kv_heads * top_f * 2 for v in boosted_per_layer.values()
    ) else "WARNING: Some layers have unexpected counts!")
