# CPU-only visualization script: produce multi-panel attention map figures and spatial stats.
# Reads .pt files from extract_attention_data.py, generates heatmaps and token overlay plots.
# Token-to-pixel mapping accounts for InternVL2.5 dynamic tiling (256 tokens per 448x448 tile).

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom

BASE_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = BASE_DIR / "results" / "visualizations"
TILE_SIZE = 448
TOKENS_PER_TILE = 256
GRID_SIDE = 16
PATCH_PIX = TILE_SIZE // GRID_SIDE  # 28


def build_token_positions(n_vis, tile_grid, has_thumbnail):
    n_cols, n_rows = tile_grid
    n_tiles = n_cols * n_rows
    has_thumb = has_thumbnail and n_tiles > 1
    total_patches = n_tiles * TOKENS_PER_TILE + (TOKENS_PER_TILE if has_thumb else 0)

    canvas_w = n_cols * TILE_SIZE
    canvas_h = n_rows * TILE_SIZE

    positions = []
    for t in range(n_tiles):
        tc = t % n_cols
        tr = t // n_cols
        for j in range(TOKENS_PER_TILE):
            jr = j // GRID_SIDE
            jc = j % GRID_SIDE
            px = tc * TILE_SIZE + jc * PATCH_PIX
            py = tr * TILE_SIZE + jr * PATCH_PIX
            positions.append((px, py, PATCH_PIX, PATCH_PIX))

    if has_thumb:
        thumb_patch_w = canvas_w / GRID_SIDE
        thumb_patch_h = canvas_h / GRID_SIDE
        for j in range(TOKENS_PER_TILE):
            jr = j // GRID_SIDE
            jc = j % GRID_SIDE
            px = jc * thumb_patch_w
            py = jr * thumb_patch_h
            positions.append((px, py, thumb_patch_w, thumb_patch_h))

    return positions[:n_vis], canvas_w, canvas_h


def attention_to_heatmap(attn_vals, n_vis, tile_grid, has_thumbnail, canvas_w, canvas_h):
    n_cols, n_rows = tile_grid
    n_tiles = n_cols * n_rows
    tile_tokens = n_tiles * TOKENS_PER_TILE
    has_thumb = has_thumbnail and n_tiles > 1

    heatmap = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    count = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    tile_attn = attn_vals[:min(tile_tokens, n_vis)]
    for t in range(n_tiles):
        start = t * TOKENS_PER_TILE
        end = min(start + TOKENS_PER_TILE, len(tile_attn))
        if start >= len(tile_attn):
            break
        tile_vals = tile_attn[start:end]
        grid_2d = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.float32)
        for j, v in enumerate(tile_vals):
            grid_2d[j // GRID_SIDE, j % GRID_SIDE] = v

        tc = t % n_cols
        tr = t // n_cols
        upsampled = ndimage_zoom(grid_2d, PATCH_PIX, order=1)
        y0 = tr * TILE_SIZE
        x0 = tc * TILE_SIZE
        h_up, w_up = upsampled.shape
        heatmap[y0:y0+h_up, x0:x0+w_up] += upsampled
        count[y0:y0+h_up, x0:x0+w_up] += 1.0

    if has_thumb and n_vis > tile_tokens:
        thumb_attn = attn_vals[tile_tokens:n_vis]
        grid_2d = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.float32)
        for j, v in enumerate(thumb_attn):
            if j >= TOKENS_PER_TILE:
                break
            grid_2d[j // GRID_SIDE, j % GRID_SIDE] = v
        upsampled = ndimage_zoom(grid_2d, (canvas_h / GRID_SIDE, canvas_w / GRID_SIDE), order=1)
        heatmap[:canvas_h, :canvas_w] += upsampled[:canvas_h, :canvas_w]
        count[:canvas_h, :canvas_w] += 1.0

    count[count == 0] = 1.0
    heatmap = heatmap / count
    return heatmap


def plot_retained_tokens(ax, img_resized, positions, retained_indices, canvas_w, canvas_h, color, title):
    ax.imshow(img_resized)
    overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    for idx in retained_indices:
        if idx < len(positions):
            px, py, pw, ph = positions[idx]
            px, py, pw, ph = int(px), int(py), int(round(pw)), int(round(ph))
            overlay[py:py+ph, px:px+pw, :3] = color
            overlay[py:py+ph, px:px+pw, 3] = 0.5
    ax.imshow(overlay)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def compute_spatial_stats(positions, retained_indices, bbox, img_w, img_h, canvas_w, canvas_h):
    if len(retained_indices) == 0:
        return {"top_half_frac": 0.0, "token_bbox_iou": 0.0}

    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h
    gt_x1 = bbox[0] * scale_x
    gt_y1 = bbox[1] * scale_y
    gt_x2 = bbox[2] * scale_x
    gt_y2 = bbox[3] * scale_y

    gt_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    gt_mask[int(gt_y1):int(gt_y2), int(gt_x1):int(gt_x2)] = True

    token_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    n_top = 0
    n_total = 0
    for idx in retained_indices:
        if idx < len(positions):
            px, py, pw, ph = positions[idx]
            px, py, pw, ph = int(px), int(py), int(round(pw)), int(round(ph))
            token_mask[py:py+ph, px:px+pw] = True
            cy = py + ph / 2.0
            if cy < canvas_h / 2.0:
                n_top += 1
            n_total += 1

    top_half_frac = n_top / max(n_total, 1)

    intersection = np.logical_and(gt_mask, token_mask).sum()
    union = np.logical_or(gt_mask, token_mask).sum()
    token_bbox_iou = float(intersection) / max(float(union), 1.0)

    return {"top_half_frac": round(top_half_frac, 4), "token_bbox_iou": round(token_bbox_iou, 4)}


def process_one_example(data, idx, out_dir):
    a_shallow = data["a_shallow"]
    a_mid = data["a_mid"]
    a_debiased = data["a_debiased"]
    fastv_idx = data["fastv_indices"]
    d2_idx = data["d2pruner_indices"]
    ours_idx = data["ours_indices"]
    tile_grid = tuple(data["tile_grid"])
    has_thumb = data["has_thumbnail"]
    n_vis = data["n_vis"]
    image_path = str(BASE_DIR / data["image_path"])
    sent = data["sent"]
    bbox = data["bbox"]
    img_w, img_h = data["img_size"]

    positions, canvas_w, canvas_h = build_token_positions(n_vis, tile_grid, has_thumb)

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((canvas_w, canvas_h), Image.LANCZOS)
    img_arr = np.array(img_resized)

    scale_x = canvas_w / img_w
    scale_y = canvas_h / img_h
    gt_rect = [bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y]

    hm_shallow = attention_to_heatmap(a_shallow, n_vis, tile_grid, has_thumb, canvas_w, canvas_h)
    hm_mid = attention_to_heatmap(a_mid, n_vis, tile_grid, has_thumb, canvas_w, canvas_h)
    hm_debiased = attention_to_heatmap(a_debiased, n_vis, tile_grid, has_thumb, canvas_w, canvas_h)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Example {idx}: "{sent}" | Position: {bbox}', fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.imshow(img_arr)
    rect = mpatches.Rectangle((gt_rect[0], gt_rect[1]), gt_rect[2]-gt_rect[0], gt_rect[3]-gt_rect[1],
                                linewidth=3, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Original + GT BBox", fontsize=10)
    ax.axis("off")

    for ax_i, (hm, title) in zip(
        [axes[0, 1], axes[0, 2], axes[0, 3]],
        [(hm_shallow, "Shallow Attn (L2)"), (hm_mid, "Mid Attn (L12)"), (hm_debiased, "Debiased (L12/L2)")]
    ):
        ax_i.imshow(img_arr)
        vmax = np.percentile(hm, 98) if hm.max() > 0 else 1.0
        ax_i.imshow(hm, cmap="jet", alpha=0.55, vmin=0, vmax=vmax)
        rect = mpatches.Rectangle((gt_rect[0], gt_rect[1]), gt_rect[2]-gt_rect[0], gt_rect[3]-gt_rect[1],
                                    linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
        ax_i.add_patch(rect)
        ax_i.set_title(title, fontsize=10)
        ax_i.axis("off")

    methods = [
        (fastv_idx, [1.0, 0.0, 0.0], "FastV (L2 top-k)"),
        (d2_idx, [0.0, 0.0, 1.0], "D2Pruner (L2 debiased+MIS)"),
        (ours_idx, [0.0, 0.8, 0.0], "Ours (L12 raw+MIS)"),
    ]
    for ax_i, (indices, color, title) in zip([axes[1, 0], axes[1, 1], axes[1, 2]], methods):
        plot_retained_tokens(ax_i, img_arr, positions, indices, canvas_w, canvas_h, color, title)
        rect = mpatches.Rectangle((gt_rect[0], gt_rect[1]), gt_rect[2]-gt_rect[0], gt_rect[3]-gt_rect[1],
                                    linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
        ax_i.add_patch(rect)

    ax_stats = axes[1, 3]
    ax_stats.axis("off")
    stats_text = f"IoU Results:\n"
    stats_text += f"FastV:    IoU<0.5 (fail)\n"
    stats_text += f"D2Pruner: IoU>=0.5 (pass)\n"
    stats_text += f"Ours:     IoU>=0.5 (pass)\n\n"
    stats_text += f"Tokens kept: {len(fastv_idx)} / {n_vis}\n"
    stats_text += f"Keep ratio: {len(fastv_idx)/n_vis:.1%}\n"
    stats_text += f"Tiles: {tile_grid[0]}x{tile_grid[1]}"
    if has_thumb:
        stats_text += " + thumb"
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_stats.set_title("Info", fontsize=10)

    plt.tight_layout()
    fig_path = out_dir / f"viz_example_{idx}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    stats = {}
    for method_name, indices in [("FastV", fastv_idx), ("D2Pruner", d2_idx), ("Ours", ours_idx)]:
        s = compute_spatial_stats(positions, indices, bbox, img_w, img_h, canvas_w, canvas_h)
        stats[method_name] = s
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-dir", type=str, default=str(VIZ_DIR))
    parser.add_argument("--max-examples", type=int, default=-1)
    args = parser.parse_args()

    viz_dir = Path(args.viz_dir)
    examples_json = viz_dir / "selected_examples.json"
    with open(examples_json) as f:
        examples = json.load(f)

    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    all_stats = []
    for idx in range(len(examples)):
        pt_path = viz_dir / f"attn_data_{idx}.pt"
        if not pt_path.exists():
            print(f"Skipping example {idx}: {pt_path} not found")
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        for k in ["a_shallow", "a_mid", "a_debiased", "fastv_indices", "d2pruner_indices", "ours_indices"]:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].numpy()

        print(f"Processing example {idx}: {data['sent']}")
        stats = process_one_example(data, idx, viz_dir)
        stats["example_idx"] = idx
        stats["sent"] = data["sent"]
        stats["position"] = examples[idx].get("position", "unknown")
        all_stats.append(stats)

    stats_json_path = viz_dir / "spatial_stats.json"
    with open(stats_json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved spatial stats JSON: {stats_json_path}")

    md_lines = ["# Spatial Statistics of Retained Tokens", ""]
    md_lines.append("| Example | Position | Sentence | Method | Top-Half Frac | Token-BBox IoU |")
    md_lines.append("|---------|----------|----------|--------|:------------:|:-------------:|")
    for s in all_stats:
        for method in ["FastV", "D2Pruner", "Ours"]:
            ms = s[method]
            md_lines.append(
                f"| {s['example_idx']} | {s['position']} | {s['sent'][:30]} | {method} | "
                f"{ms['top_half_frac']:.3f} | {ms['token_bbox_iou']:.3f} |"
            )

    md_lines.append("")
    md_lines.append("## Aggregated Statistics")
    md_lines.append("")
    md_lines.append("| Method | Avg Top-Half Frac | Avg Token-BBox IoU |")
    md_lines.append("|--------|:-----------------:|:------------------:|")
    for method in ["FastV", "D2Pruner", "Ours"]:
        avg_top = np.mean([s[method]["top_half_frac"] for s in all_stats])
        avg_iou = np.mean([s[method]["token_bbox_iou"] for s in all_stats])
        md_lines.append(f"| {method} | {avg_top:.3f} | {avg_iou:.3f} |")

    md_path = viz_dir / "spatial_stats.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved spatial stats MD: {md_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
