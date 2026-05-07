# DS Error, H_res Routing, and Product Stability Analysis

## Experiment Overview

Analyzes the structural effects of RRCS (Range-Capped Sinkhorn) beyond gradient flow, examining:
1. Whether better-conditioned Sinkhorn inputs lead to better doubly-stochastic (DS) approximations
2. Whether learned routing patterns differ qualitatively across conditions
3. Whether DS errors accumulate multiplicatively across depth

Three conditions compared (seed=42, 5000 training steps, 48 transformer layers with 96 HC modules):
- **mHC default**: tau=0.05, no range capping
- **Cap-init**: tau=0.2667 (constant effective temperature from initialization), no range capping
- **RRCS (r_cap=2.0)**: tau=0.05, adaptive range capping before Sinkhorn iterations

## Setup

- Model: 48-layer nanoGPT (~20M params) with 4-stream mHC (H_res is 4x4)
- Each transformer layer has 2 HC modules (hc_attn, hc_mlp) = 96 total
- Sinkhorn iterations: 10 for all conditions
- Data sources: diagnostic CSVs (every 10 steps) and final checkpoints

## Key Results

### DS Error Comparison (Figure: ds_error_comparison.png)

| Condition | Mean Row DS Error | Max Row DS Error | Mean Col DS Error | Max Col DS Error |
|-----------|------------------|------------------|-------------------|------------------|
| mHC default | 0.0 | 0.0 | 0.0 | 0.0 |
| Cap-init | 0.0 | 0.0 | 0.0 | 0.0 |
| RRCS (r=2) | 3.94e-07 | 1.31e-06 | 8.88e-08 | 1.79e-07 |

**Interpretation**: mHC default and Cap-init show zero DS error because their H_res_logits receive essentially zero gradients and remain at initialization (uniform logits, which are trivially DS). RRCS has non-zero but extremely small DS errors (~1e-7), confirming that range-capped inputs (capped to r=2.0 before 10 Sinkhorn iterations) still produce near-perfect DS approximations. The question of "does RRCS improve DS quality?" is nuanced: RRCS is the only condition where logits actually evolve from initialization, so it's the only condition where DS quality is a meaningful measure. The DS errors remain negligible throughout training.

### H_res Routing Patterns (Figures: hres_routing_heatmaps.png, hres_entropy_by_layer.png)

| Condition | Mean Entropy | Min Entropy | Max Entropy |
|-----------|-------------|-------------|-------------|
| mHC default | 0.0 | 0.0 | 0.0 |
| Cap-init | 6.5e-12 | 6.5e-12 | 6.5e-12 |
| RRCS (r=2) | 0.933 | 0.918 | 1.044 |

**Interpretation**: The routing patterns differ dramatically:

- **mHC default and Cap-init**: H_res matrices are effectively identity permutations (pure permutation-like routing with zero entropy). Since gradients don't flow into H_res_logits, these conditions never learn any routing -- they use the same fixed routing pattern from initialization across all layers and throughout training.

- **RRCS**: H_res matrices show meaningful non-trivial routing with entropy ~0.93 (between zero for pure permutation and log(4)=1.39 for uniform). This indicates learned routing that mixes streams in a non-trivial but structured way. The heatmaps show variation across layers, suggesting the model learns layer-specific routing strategies when gradients actually flow into H_res_logits.

The entropy varies across layers (min 0.918, max 1.044), suggesting some layers develop sharper routing while others maintain more diffuse patterns. This layer-wise variation is only possible with RRCS since the other conditions have frozen logits.

### Product Stability (Figure: hres_product_stability.png)

| Condition | Col-Sum Deviation at Depth 96 |
|-----------|-------------------------------|
| mHC default | 0.0 |
| Cap-init | 2.7e-11 |
| RRCS (r=2) | 9.3e-08 |

**Interpretation**: The cumulative product P_96 = H_res_0 @ H_res_1 @ ... @ H_res_95 remains nearly doubly stochastic for all conditions. RRCS shows the largest accumulated deviation (9.3e-08) but this is still negligible, demonstrating that the small per-layer DS errors from range capping do not compound into problematic deviations even at 96 layers deep. This is important for scaling to deeper models -- RRCS maintains excellent product stability while enabling actual routing learning.

## Key Observations

1. **DS quality is a secondary concern**: The primary effect of RRCS is enabling gradient flow into H_res_logits, not improving DS approximation quality. In the baseline conditions, DS error is trivially zero because logits don't change.

2. **RRCS enables qualitatively different routing**: Only RRCS produces non-trivial learned routing patterns. The other conditions are frozen at initialization.

3. **Product stability is excellent**: Even with 96 matrix products, the accumulated DS error under RRCS remains at ~1e-7, well within acceptable bounds for practical use.

4. **Entropy profile is informative**: RRCS routing entropy (~0.93) sits between pure permutation (0) and uniform (1.39), suggesting the model learns structured but non-trivial stream mixing.
