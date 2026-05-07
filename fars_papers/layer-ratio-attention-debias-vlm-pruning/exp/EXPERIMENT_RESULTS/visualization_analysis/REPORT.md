# Visualization Analysis: Attention Maps and Retained Token Sets

## Experiment Overview

Qualitative visualization comparing how three pruning methods (FastV, D2Pruner, Ours) select visual tokens on representative RefCOCO grounding examples. Each method retains 10% of visual tokens.

## Setup

- **Model**: InternVL2.5-8B (32 LLM layers, dynamic tiling with 256 tokens per 448x448 tile)
- **Dataset**: refcoco_val (8 selected examples across all spatial positions)
- **Methods**:
  - FastV: raw attention top-k at L2 (layer index 1)
  - D2Pruner: offline bias prior debiased attention + MIS at L2
  - Ours: raw mid-layer attention + MIS at L12 (layer index 11)
- **Keep ratio**: 10% for all methods

## Example Selection

8 examples chosen from refcoco_val where FastV fails (IoU < 0.5) but Ours succeeds (IoU >= 0.5), covering all spatial positions:

| # | Position | Sentence | FastV IoU | D2Pruner IoU | Ours IoU |
|---|----------|----------|:---------:|:------------:|:--------:|
| 0 | top-left | cup behind hot dog | 0.000 | 0.809 | 0.981 |
| 1 | top-center | sandwich on left | 0.010 | 0.792 | 0.967 |
| 2 | top-right | tie on right | 0.417 | 0.597 | 0.980 |
| 3 | center-left | dude on left | 0.000 | 0.257 | 0.973 |
| 4 | center-center | right hot dog | 0.011 | 0.046 | 0.990 |
| 5 | center-right | kid on right | 0.000 | 0.653 | 0.994 |
| 6 | bottom-left | green pot on left | 0.000 | 0.869 | 0.983 |
| 7 | bottom-center | bottom most dish | 0.240 | 0.726 | 0.990 |

## Key Results

### Visualization Panels (per example)

Each figure has 2 rows:
- **Row 1**: (1) Original image + GT bbox, (2) Shallow attention heatmap (L2), (3) Mid-layer attention heatmap (L12), (4) Debiased heatmap (L12/L2)
- **Row 2**: Retained token overlays for (5) FastV, (6) D2Pruner, (7) Ours

### Spatial Distribution of Retained Tokens

| Method | Avg Top-Half Fraction | Avg Token-BBox IoU |
|--------|:---------------------:|:------------------:|
| FastV | 0.473 | 0.148 |
| D2Pruner | 0.482 | 0.177 |
| Ours | 0.494 | 0.212 |

### Key Observations

1. **Token-BBox IoU**: Ours achieves 43% higher token-bbox overlap than FastV (0.212 vs 0.148) and 20% higher than D2Pruner (0.212 vs 0.177). This means our method concentrates retained tokens more effectively around the referred object.

2. **Top-Half Fraction**: Ours has a slightly more balanced spatial distribution (0.494, closer to 0.5 ideal) than FastV (0.473) and D2Pruner (0.482). FastV shows a slight bottom bias (under 0.5 on average).

3. **Position-dependent behavior**: For top-positioned objects (examples 0-2), the improvement is most pronounced -- Ours achieves top-half fractions of 0.53-0.58 compared to FastV's 0.45-0.51, showing it allocates more tokens to the object region when it's in the upper portion of the image.

4. **L12 vs L2 attention**: The mid-layer (L12) attention heatmaps show much more semantically focused patterns than shallow (L2) attention. L2 attention appears relatively uniform across visual tokens, while L12 attention concentrates on task-relevant regions. This confirms that the key advantage of our method comes from using deeper-layer attention for token selection rather than from any debiasing mechanism.

5. **Debiased heatmap (L12/L2)**: The ratio A_mid/A_shallow amplifies regions where mid-layer attention exceeds shallow-layer attention. Since shallow attention is relatively uniform, this ratio preserves the L12 saliency pattern with slight amplification, but does not fundamentally change the selection. This is consistent with the ablation finding that shallow-layer normalization hurts rather than helps.

## Artifacts

- Figures: `results/visualizations/viz_example_{0..7}.png`
- Per-example attention data: `results/visualizations/attn_data_{0..7}.pt`
- Spatial statistics: `results/visualizations/spatial_stats.json`, `results/visualizations/spatial_stats.md`
- Selected examples: `results/visualizations/selected_examples.json`
