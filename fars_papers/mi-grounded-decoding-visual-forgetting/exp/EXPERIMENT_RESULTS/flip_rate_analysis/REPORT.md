# Flip Rate Analysis: Short (128) vs Long (512) Budget

## Experiment Overview

Analyzes how increasing the reasoning budget from 128 to 512 tokens affects answer correctness across three decoding methods: vanilla, visual replay, and adaptive MI decoding. "Correct->Wrong" flips (short correct, long wrong) isolate the regime where overthinking/visual forgetting harms performance. "Wrong->Correct" flips capture beneficial extended reasoning.

## Setup

- **Model**: VLAA-Thinker-Qwen2.5VL-7B
- **Benchmarks**: MMStar (1500 items), HallusionBench (1129 items)
- **Short budget**: max_new_tokens=128
- **Long budget**: max_new_tokens=512
- **Methods**: Vanilla decoding, Visual Replay (4 image re-insertions at 50% resolution), Adaptive MI (lambda=0.005, alpha=0.8, max_weight=5.0)
- **Infrastructure**: 4-GPU data-parallel inference per run
- **Scripts**: `mi_decoding/scripts/compute_flip_rates.py`, `mi_decoding/scripts/plot_flip_rates.py`
- **Outputs**: `mi_decoding/results/flip_rate_analysis.json`, `mi_decoding/results/flip_rate_analysis.pdf`

## Key Results

### Overall Flip Rates

| Method | Benchmark | C->W% | W->C% | Net% | Short Acc% | Long Acc% |
|--------|-----------|:-----:|:-----:|:----:|:----------:|:---------:|
| Vanilla | MMStar | 3.07 | 26.93 | -23.87 | 38.27 | 62.13 |
| Vanilla | HallusionBench | 9.74 | 12.05 | -2.30 | 63.95 | 66.25 |
| Visual Replay | MMStar | 2.87 | 7.80 | -4.93 | 57.53 | 62.47 |
| Visual Replay | HallusionBench | 4.43 | 5.93 | -1.51 | 65.63 | 67.14 |
| Adaptive MI | MMStar | 4.33 | 25.07 | -20.73 | 41.33 | 62.07 |
| Adaptive MI | HallusionBench | 11.43 | 14.26 | -2.83 | 64.92 | 67.76 |

### MMStar Category Breakdown (C->W% / W->C% / Net%)

| Category | Vanilla | Visual Replay | Adaptive MI |
|----------|:-------:|:-------------:|:-----------:|
| Coarse Perception | 2.0/21.2/-19.2 | 1.2/2.4/-1.2 | 2.0/21.2/-19.2 |
| Fine-grained Perception | 2.4/10.8/-8.4 | 1.6/1.6/0.0 | 3.6/16.4/-12.8 |
| Instance Reasoning | 2.4/28.0/-25.6 | 2.0/4.8/-2.8 | 2.4/28.0/-25.6 |
| Logical Reasoning | 2.0/35.2/-33.2 | 2.4/10.4/-8.0 | 4.8/28.0/-23.2 |
| Math | 2.4/42.8/-40.4 | 5.2/18.8/-13.6 | 3.2/33.2/-30.0 |
| Science & Technology | 7.2/23.6/-16.4 | 4.8/8.8/-4.0 | 10.0/23.6/-13.6 |

## Key Observations

1. **All net flip rates are negative**: Longer reasoning budgets generally improve accuracy for all methods. The 128-token budget is too short for the thinking model to complete its reasoning (especially vanilla with 27.1% empty answer extractions on MMStar).

2. **Visual replay has the lowest C->W rates**: On both benchmarks, visual replay produces the fewest correct->wrong flips (2.87% MMStar, 4.43% HBench), consistent with the hypothesis that image re-insertion mitigates visual forgetting during extended generation.

3. **Visual replay also has lower W->C rates**: This is because visual replay already achieves higher short-budget accuracy (57.53% vs 38.27% vanilla on MMStar), leaving fewer items that can flip from wrong to correct. The image re-insertion helps even at short budgets.

4. **Adaptive MI has higher C->W than vanilla on both benchmarks**: 4.33% vs 3.07% (MMStar), 11.43% vs 9.74% (HBench). The MI correction may sometimes over-adjust, causing items that were correct at short budget to become incorrect. However, adaptive MI also has high W->C rates, leading to comparable final accuracy.

5. **Category analysis**: Science & technology has the highest C->W rates across methods (7.2% vanilla, 10.0% adaptive MI), suggesting these questions are most susceptible to overthinking-induced errors. Math and logical reasoning benefit most from extended reasoning (highest W->C rates).

6. **Short-budget accuracy gap**: Vanilla short (38.27%) is much lower than visual replay short (57.53%) on MMStar. This is because vanilla truncation at 128 tokens often cuts off before the `<answer>` tag, while visual replay's two-pass approach tends to produce more complete responses even at shorter budgets.

7. **HallusionBench shows more balanced flips**: C->W and W->C rates are closer on HallusionBench than MMStar, reflecting that Yes/No questions can be answered in fewer tokens, making the 128-token budget less of a bottleneck. The C->W flips on HallusionBench (9.74% vanilla, 11.43% MI) represent genuine visual forgetting rather than truncation artifacts.
