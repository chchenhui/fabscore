# Ablation: Fixed-gamma vs Adaptive MI Decoding

## Experiment Overview

Ablation study comparing time-varying (adaptive) MI decoding against time-independent (fixed-gamma) MI decoding to test whether the time-varying schedule is a key mechanism for counteracting visual forgetting.

- **Fixed-gamma MI**: Sets gamma_t = 0.5 for all generation steps, yielding corrected logits l_hat = 2*l_c - l_u when the confidence gate is active. This is equivalent to standard contrastive decoding.
- **Adaptive MI**: Uses exponential decay gamma_t = exp(-lambda*(t + t0)) with lambda=0.02, t0=0, so the correction weight increases over time.
- Both variants share alpha=0.3 (confidence gate threshold) and max_weight=5.0.

## Setup

- Model: VLAA-Thinker-7B (UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B)
- MMStar: 200-item random subset (first 200 IDs from seed=42 300-item subset)
- HallusionBench: full benchmark (1129 items)
- max_new_tokens: 512
- 8 GPUs: 4 for fixed-gamma, 4 for adaptive MI (data-parallel sharding)

## Key Results

| Variant | MMStar (200 subset) Acc | HallusionBench aAcc |
|---|:---:|:---:|
| Vanilla | 66.00% | 66.25% |
| Fixed-gamma MI (gamma=0.5) | 64.00% | 68.11% |
| Adaptive MI (lambda=0.02) | 65.00% | 67.85% |

### HallusionBench Breakdown

| Variant | aAcc | VD Acc | VS Acc |
|---|:---:|:---:|:---:|
| Vanilla | 66.25% | 60.24% | 72.86% |
| Fixed-gamma MI | 68.11% | 62.61% | 74.16% |
| Adaptive MI | 67.85% | 62.27% | 73.98% |

## Key Observations

1. **Both MI variants improve HallusionBench over vanilla**: Fixed-gamma achieves +1.86pp and adaptive MI achieves +1.60pp over vanilla aAcc. Both improve VD accuracy by ~2pp, indicating genuine visual grounding improvement.

2. **Both MI variants slightly hurt MMStar**: Fixed-gamma drops -2.0pp and adaptive drops -1.0pp vs vanilla on the 200-item subset. This is consistent with the full-benchmark finding that MI decoding with alpha=0.3 can occasionally over-correct on knowledge-intensive questions.

3. **Adaptive MI does NOT consistently outperform fixed-gamma**: On MMStar subset, adaptive MI is 1pp better than fixed-gamma (65% vs 64%). On HallusionBench, fixed-gamma is 0.26pp better (68.11% vs 67.85%). The differences are small and within noise margins.

4. **The time-varying schedule is not clearly essential with these hyperparameters**: With alpha=0.3, the confidence gate activates frequently (most tokens have max p_c < 0.3), so the correction is applied almost always. Both the constant weight=1.0 (fixed-gamma) and the growing weight of adaptive MI produce similar effects. The time-varying benefit may be more pronounced with different alpha/lambda settings.

5. **Context**: The optimized adaptive MI (alpha=0.8, lambda=0.005) from the main experiments achieved 67.76% HallusionBench aAcc with 62.07% MMStar (full benchmark). The higher alpha=0.8 makes the gate more selective, which may be where time-varying scheduling matters more.
