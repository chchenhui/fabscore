# Condition C (Discriminative DUT) -- Qwen2.5-Math-7B-Instruct

## Experiment Overview

Evaluated the proposed Definition Unit Tests (DUT) method on Qwen2.5-Math-7B-Instruct. In Condition C, the model receives the glossary, k=3 discriminative verified facts (whose answers differ under glossary vs. alternate conventions), and then the main question. Uses the V3 "verified facts" approach where check Q&A pairs are provided as demonstrations, binding the model to the glossary convention before solving the main question.

## Setup

- **Model**: Qwen/Qwen2.5-Math-7B-Instruct
- **Condition**: C (glossary + 3 discriminative verified facts + main question)
- **Benchmark**: ErdosConventionsBench (300 items: 100 asymptotics, 100 completeness, 100 convolution)
- **Decoding**: Greedy (temperature=0), max_tokens=2048
- **Prompt version**: V3 (verified facts, chat format)
- **Inference**: vLLM batched chat inference via TrainService (1x GPU)

## Key Results

| Condition | Overall | Asymptotics | Completeness | Convolution | Alt Rate |
|-----------|---------|-------------|--------------|-------------|----------|
| A (Glossary-only) | 90.3% | 99% | 80% | 92% | 6.3% |
| B (Neutral facts) | 90.0% | 97% | 79% | 94% | 7.0% |
| **C (Discriminative DUT)** | **95.0%** | **99%** | **91%** | **95%** | **1.3%** |

### Bootstrap CI (C minus B)

- Observed difference: **+5.0pp**
- 95% CI: **[+2.0%, +8.3%]**
- Excludes zero: **Yes** (statistically significant)

### Bootstrap CI (C minus A)

- Observed difference: **+4.7pp**
- 95% CI: **[+1.3%, +8.0%]**
- Excludes zero: **Yes** (statistically significant)

## Key Observations

1. **DUT method works**: C (95.0%) significantly outperforms both B (90.0%, +5.0pp) and A (90.3%, +4.7pp). The bootstrap CIs exclude zero, confirming statistical significance.

2. **Largest gain on completeness**: C achieves 91% vs B's 79% (+12pp) on completeness, where the glossary vs. alternate distinction is most subtle. Discriminative checks that probe this boundary are most effective.

3. **Very low alternate convention match rate**: C has only 1.3% alternate matches vs B's 7.0% and A's 6.3%. The discriminative facts actively suppress the alternate convention.

4. **Verified facts key to success**: The V3 approach of providing check Q&A pairs as demonstrations (rather than asking the model to generate them) is what makes the method work. Earlier attempts where the model generated its own check answers showed no C-B advantage.

## Files

- Raw outputs: `dut_project/outputs/qwen25_math_7b_v3/condition_c.jsonl`
- Inference script: `dut_project/scripts/run_optimized_v3.sh`
