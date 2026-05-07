# CA-Magnitude Overlap Analysis Report

## Experiment Overview

This analysis investigates the relationship between FCBoost's static CA-derived channel mask and Kitty's dynamic magnitude-based channel selection. The goal is to understand whether CA is proxying magnitude (Scenario 1), whether static selection works via a different mechanism (Scenario 2), or whether dynamic selection is necessary (Scenario 3).

## Setup

- **Model**: Qwen3-8B (36 layers, 8 KV heads, head_dim=128, 64 RoPE pairs per head)
- **CA profiling**: 16 WikiText-2 sequences, 8192 tokens each, top-8 RoPE pairs (16 channels) per KV head
- **Magnitude collection**: Same 16 WikiText-2 sequences, pages of 128 tokens, top-16 channels by magnitude per page
- **Total pages per (layer, KV head)**: 1024 (16 seqs x 64 pages/seq)

### Mask definitions
- **M_CA**: 16 channels per (layer, KV head) from top-8 CA-scored RoPE pairs
- **M_mag**: 16 most-frequently-selected channels across 1024 pages by Kitty's magnitude heuristic

## Key Results

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| Jaccard overlap | **0.299** | 0.122 | 0.067 | 0.778 | 0.280 |
| Spearman rho | **0.670** | 0.185 | -0.022 | 0.960 | 0.706 |

- **97.2%** of (layer, KV head) pairs have statistically significant Spearman correlation (p < 0.05)

### Accuracy Context

| Method | AIME24 | AIME25 | Average |
|--------|--------|--------|---------|
| KIVI-KV2* | 67.78 | 64.44 | 66.11 |
| Kitty | 72.22 | 61.11 | 66.67 |
| **FCBoost v2** | **74.44** | **67.78** | **71.11** |

## Key Observations

### Finding: Scenario 2 -- Static selection works via a different mechanism

1. **Low Jaccard overlap (0.299)**: The actual 16-channel sets chosen by CA and magnitude are substantially different -- on average only about 30% of channels overlap (4-5 out of 16). This means CA and magnitude are selecting largely different channel subsets.

2. **High Spearman correlation (0.670)**: Despite selecting different channels, the overall ranking of RoPE pairs by CA score vs by mean magnitude is moderately well-correlated. CA and magnitude broadly agree on which channels are more vs less important, but they disagree on where to draw the top-16 threshold.

3. **FCBoost outperforms Kitty (+4.44pp)**: Despite selecting different channels, the CA-derived static mask produces better accuracy than Kitty's dynamic magnitude selection. This is strong evidence that (a) dynamic per-page selection is not necessary, and (b) CA identifies a qualitatively different and more effective set of channels.

### Mechanistic Interpretation

The discrepancy between high rank correlation and low set overlap can be explained by the CA and magnitude ranking curves having similar shapes but different breakpoints. Many channels at the decision boundary (around rank 16 out of 128) are close in both metrics. Small differences in ranking near the boundary cause different channels to be selected, even though the overall ranking is correlated.

CA measures structural attention agreement (how well a RoPE frequency pair's attention pattern matches the full-head pattern) while magnitude measures numerical scale. These capture related but distinct properties:
- **Magnitude**: Channels with large absolute values contribute more to dot-product attention and suffer more from quantization error in absolute terms
- **CA**: Channels whose individual attention patterns most closely replicate the full-head attention, capturing structural importance regardless of scale

The fact that CA-derived selection outperforms magnitude-based selection suggests that structural agreement (CA) is a more informative criterion for identifying quantization-sensitive channels than numerical scale (magnitude).

## Figures

- `results/figures/jaccard_heatmap.png`: Per-(layer, KV head) Jaccard overlap heatmap
- `results/figures/spearman_heatmap.png`: Per-(layer, KV head) Spearman correlation heatmap
- `results/figures/jaccard_histogram.png`: Distribution of Jaccard overlap values
- `results/figures/ca_vs_magnitude_scatter.png`: CA score vs magnitude scatter plots for representative (layer, head) combinations

## Conclusion

The analysis classifies the outcome as **Scenario 2**: low mask overlap but FCBoost beats Kitty. Static CA-based channel selection succeeds via a different mechanism than magnitude-based selection. CA identifies structurally important RoPE frequencies that are quantization-sensitive, and this structural criterion appears more effective than magnitude for selecting channels to boost from INT2 to INT4.
