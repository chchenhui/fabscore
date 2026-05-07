# Visual Replay Baseline on VLAA-Thinker-7B

## Experiment Overview

Reproduced the visual replay (VR) inference-time baseline on VLAA-Thinker-7B, following the protocol described in the VAPO paper (arXiv 2509.25848, Appendix A.5). Visual replay re-inserts a downsampled copy of the input image at 4 approximately evenly spaced punctuation boundaries during the reasoning trace to re-ground the model's visual attention.

## Setup

- **Model**: UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B
- **Decoding**: Two-pass visual replay with greedy decoding (do_sample=False)
  - Pass 1: Vanilla generation (max_new_tokens=512) to get full trace
  - Pass 2: Re-generate with 4 downsampled (50% resolution) image insertions at punctuation boundaries
- **Benchmarks**: MMStar (1500 val items), HallusionBench (1129 items)
- **Hardware**: 4x A100-80GB, data-parallel (1 model per GPU, each processes 1/4 of data)
- **Precision**: bfloat16

## Key Results

| Benchmark | Metric | Visual Replay | Vanilla | Delta | Published VR (VAPO) |
|-----------|--------|:---:|:---:|:---:|:---:|
| MMStar | Accuracy | 62.47% | 62.13% | +0.34pp | 52.9% |
| HallusionBench | aAcc | 67.14% | 66.25% | +0.89pp | 56.2% |

### HallusionBench Breakdown

| Subset | Visual Replay | Vanilla |
|--------|:---:|:---:|
| VD (Visual Dependent) | 62.10% | 60.24% |
| VS (Visual Supplement) | 72.68% | 72.86% |

## Key Observations

1. Visual replay provides a small but consistent improvement over vanilla decoding (+0.34pp MMStar, +0.89pp HallusionBench), matching the directional improvement reported in the published paper.
2. The improvement is most pronounced on HallusionBench VD subset (+1.86pp), where visual grounding is most critical.
3. Our absolute values are ~10pp above published VAPO numbers, consistent with the systematic evaluation pipeline discrepancy observed in the vanilla baseline (~12pp gap due to different prompt formatting, answer extraction, and evaluation methods in VLMEvalKit).
4. The pipeline is internally consistent for comparing decoding strategies within our experimental framework.
5. Runtime overhead vs vanilla: ~2x slower due to two-pass generation (6.8s/item vs ~3.4s/item for vanilla on MMStar).
