# Phase-0 Go/No-Go Diagnostic: Shallow-Layer Attention Characterization

## Experiment Overview

Tests whether shallow-layer text-to-vision attention in InternVL2.5-8B is (i) more prompt-stable and (ii) more position-correlated than mid-layer attention. If both hold, shallow-layer attention can serve as a per-instance positional bias prior for debiasing mid-layer scores in visual token pruning.

## Setup

- **Model**: InternVL2.5-8B (32 LLM layers, InternLM2 backbone)
- **Data**: 30 randomly sampled COCO train2014 images (seed=42), 5 prompt templates each = 150 forward passes
- **Target Layers**: Shallow candidates K_s in {1, 2, 3}, Mid candidates K_m in {4, 8, 12}
- **Attention Extraction**: Monkey-patched InternLM2FlashAttention2 to eager computation on target layers only; all other layers use flash attention for efficiency
- **Hardware**: 1x GPU (A100-80GB), ~4 min total runtime
- **Determinism**: Verified identical attention vectors across duplicate runs (max abs diff = 0.0 for all layers)

### Prompt Templates
1. "Describe the image."
2. "What do you see in this picture?"
3. "Please describe the provided image."
4. "Tell me about this photo."
5. "What is shown in this image?"

## Key Results

| Layer | Prompt Stability | Position Correlation (mean |Spearman rho|) | Entropy |
|-------|-----------------|-------------------------------------------|---------|
| 1     | 0.9966          | 0.1430                                    | 5.896   |
| 2     | 0.9945          | 0.1553                                    | 5.551   |
| 3     | 0.9969          | 0.1746                                    | 4.709   |
| 4     | 0.9973          | 0.1764                                    | 4.847   |
| 8     | 0.9787          | 0.1786                                    | 5.814   |
| 12    | 0.9283          | 0.1751                                    | 5.039   |

### Go/No-Go Decision: **PIVOT**

- **Selected K_s = 3** (maximizes stability * position_corr among shallow candidates)
- **Selected K_m = 4** (best mid-layer candidate)
- Prompt stability 0.997 > 0.8 threshold: **PASS**
- Position correlation 0.175 < 0.3 threshold: **FAIL**

## Key Observations

1. **All layers are highly prompt-stable** (cosine similarity > 0.93), with shallow layers (1-3) slightly more stable than deeper layers. The stability decreases monotonically with depth (0.997 at layer 1 -> 0.928 at layer 12).

2. **Position correlation is weak across all layers** (0.14-0.18), with no significant difference between shallow and mid layers. The hypothesis that shallow layers are dominated by positional encoding artifacts is NOT supported.

3. **Entropy varies across layers**: Layer 3 has the lowest entropy (4.71), suggesting more concentrated attention, while layers 1 and 8 have the highest (5.90, 5.81), suggesting more diffuse attention.

4. **PIVOT interpretation**: Shallow-layer attention IS more prompt-stable than mid-layer attention, suggesting it captures image-intrinsic properties less influenced by prompt wording. However, it is NOT position-dominated. This means shallow attention can still serve as a useful normalization prior (saliency prior), but the debiasing may not be specifically correcting positional bias - it may instead be correcting for general attention distribution differences between layers.

5. **Implications for the main experiment**: The online shallow-layer prior method ($A_{mid} / (A_{shallow} + \epsilon)$) should still be tested, as the high prompt stability of shallow attention suggests it captures stable image features that could serve as a normalization baseline. The method may work for reasons beyond positional bias correction.
