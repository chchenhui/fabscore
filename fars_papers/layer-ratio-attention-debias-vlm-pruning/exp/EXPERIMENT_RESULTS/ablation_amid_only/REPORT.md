# Ablation: A_mid-only vs Shallow-Layer Ratio Normalization

## Experiment Overview

This ablation tests whether the shallow-layer normalization `A_mid / (A_shallow + eps)` provides genuine debiasing value beyond using raw mid-layer attention `A_mid` directly. Both variants use D2Pruner's pivot-based MIS diverse token selection pipeline. We test at the Phase-0 layer (K_m=4) and the optimized layer (K_m=12), with two K_s values (2 and 3).

## Setup

- **Model**: InternVL2.5-8B (32 LLM layers, 256 visual tokens per patch)
- **Task**: Visual grounding on RefCOCO/RefCOCO+/RefCOCOg (8 splits, ~57k samples)
- **Metric**: Grounding accuracy (IoU >= 0.5)
- **Keep ratio**: 10%
- **Selection**: D2Pruner MIS (r_pivot=0.7, theta_sim=0.8, alpha=0.5)
- **Hardware**: 4x A100-80GB GPUs with data parallelism per run
- **Greedy decoding**: do_sample=False, max_new_tokens=100

## Key Results

### Ablation Table (All at 10% Keep Ratio)

| Variant | Debiasing | RefCOCO avg | RefCOCO+ avg | RefCOCOg avg | Overall avg |
|---------|-----------|:-----------:|:------------:|:------------:|:-----------:|
| No Pruning (100%) | N/A | 74.22 | 70.19 | 71.63 | 72.06 |
| FastV (L2) | None (raw attn top-k) | 35.92 | 30.91 | 35.13 | 33.85 |
| D2Pruner (L2) | Offline prior | 57.65 | 51.72 | 56.09 | 55.03 |
| A_mid+MIS (L4) | None (mid-layer + MIS) | 41.10 | 35.87 | 39.35 | 38.70 |
| **Ratio K_s=3 (L4)** | **Online A_mid/(A_shallow+eps)** | **14.67** | **11.96** | **12.99** | **13.24** |
| **A_mid+MIS (L12)** | **None (mid-layer + MIS)** | **68.70** | **63.82** | **66.50** | **66.32** |
| Ratio K_s=3 (L12) | Online A_mid/(A_shallow+eps) | 56.62 | 52.87 | 54.29 | 54.63 |
| Ratio K_s=2 (L12) | Online A_mid/(A_shallow+eps) | 63.00 | 59.15 | 61.16 | 61.10 |
| WeightedCombo K_s=2 (L12) | alpha*norm(A_mid)+(1-a)*norm(A_s) | 66.44 | 61.98 | 64.32 | 64.24 |

### Gap Analysis: Ratio Normalization vs Raw A_mid

| Layer | Variant | Raw A_mid | With Ratio | Gap | Verdict |
|-------|---------|:---------:|:----------:|:---:|---------|
| L4 | K_s=3 | 38.70 | 13.24 | **-25.46** | Catastrophic harm |
| L12 | K_s=3 | 66.32 | 54.63 | **-11.69** | Significant harm |
| L12 | K_s=2 | 66.32 | 61.10 | **-5.22** | Clear harm |
| L12 | WeightedCombo K_s=2 | 66.32 | 64.24 | **-2.08** | Moderate harm |

**Conclusion: The shallow-layer normalization does NOT outperform A_mid-only at any configuration. The gap is consistently negative (normalization hurts), far exceeding the +/-1 point threshold. The hypothesis is REFUTED.**

### Attention Entropy Analysis (from Phase-0 Data)

| Layer | Entropy | Position Correlation (Spearman rho) | Prompt Stability (cosine sim) |
|-------|:-------:|:-----------------------------------:|:-----------------------------:|
| K_s=3 | 4.71 | 0.175 | 0.997 |
| K_m=4 | 4.85 | 0.176 | 0.997 |
| L12 | 5.04 | 0.175 | 0.928 |

**Entropy analysis**: Entropies at K_s=3 (4.71) and K_m=4 (4.85) are similar (ratio ~0.97), and both close to L12 (5.04). Position correlation is uniformly weak (~0.17) across all layers. The ratio formula is NOT correcting for positional bias (there is minimal positional bias at any layer), nor correcting meaningful entropy differences (entropies are similar). Instead, dividing by A_shallow distorts the saliency signal, removing information about which tokens are genuinely important.

## Key Observations

1. **Ratio normalization is uniformly harmful.** At every layer and K_s configuration tested, `A_mid / (A_shallow + eps)` degrades performance compared to raw `A_mid`. The harm ranges from -2.08 (softest variant, weighted combo) to -25.46 (original ratio at L4) points.

2. **Deeper K_s reduces harm but never eliminates it.** With K_s=2 (closer to input) the harm is -5.22 at L12; with K_s=3 it's -11.69. This suggests the ratio's damage correlates with the informativeness of the shallow-layer attention being divided out.

3. **The denominator does non-trivial work, but that work is harmful.** The ratio formula actively distorts the saliency ranking rather than correcting positional bias, because there is minimal positional bias to correct (position correlation ~0.17 at all layers).

4. **Raw A_mid at deeper layers is the optimal strategy.** The best result (66.32% at L12) comes from using raw mid-layer attention without any normalization or debiasing, eliminating the need for both offline calibration (D2Pruner) and online shallow-layer processing.

5. **Even softer integration (weighted combination) hurts.** The `alpha*norm(A_mid) + (1-alpha)*norm(A_shallow)` formula (64.24%) also underperforms raw A_mid (66.32%), confirming that shallow-layer attention adds noise rather than useful information at deeper pruning layers.
