# K-Ablation: Effect of Number of Discriminative Checks (k=1 vs k=3)

## Experiment Overview

This ablation investigates how the number of discriminative checks (k) affects DUT performance. The default configuration uses k=3 checks. This experiment tests k=1 (a single discriminative check) to determine whether the benefit comes from a single definition-binding moment or requires multiple checks.

## Setup

- **Benchmark**: ErdosConventionsBench (300 items, 3 families)
- **Models**: Qwen2.5-Math-7B-Instruct, Llama-3.1-8B-Instruct
- **Conditions compared**:
  - A: Glossary-only (no checks)
  - B: Neutral checks (k=3)
  - C(k=1): Discriminative DUT with 1 check (first check only)
  - C(k=3): Discriminative DUT with 3 checks (default)
- **Inference**: Greedy decoding (temperature=0), max_tokens=2048, bfloat16, 1xGPU
- **Prompt format**: V3 verified-facts chat format (identical to main experiments except k=1 uses only the first discriminative check)

## Key Results

### Qwen2.5-Math-7B-Instruct

| Condition | Overall | Asymptotics | Completeness | Convolution | Alt Rate |
|-----------|---------|-------------|--------------|-------------|----------|
| A (Glossary-only) | 90.3% | 99% | 80% | 92% | 6.3% |
| B (Neutral k=3) | 90.0% | 97% | 79% | 94% | 7.0% |
| C (DUT k=1) | 92.3% | 99% | 88% | 90% | 2.7% |
| C (DUT k=3) | 95.0% | 99% | 91% | 95% | 1.3% |

- k=3 minus k=1: +2.7pp, 95% CI [-0.7%, +6.0%], **does not exclude zero**
- k=1 minus B: +2.3pp, 95% CI [-1.7%, +6.3%], does not exclude zero
- k=1 minus A: +2.0pp, 95% CI [-2.0%, +6.0%], does not exclude zero

### Llama-3.1-8B-Instruct

| Condition | Overall | Asymptotics | Completeness | Convolution | Alt Rate |
|-----------|---------|-------------|--------------|-------------|----------|
| A (Glossary-only) | 56.7% | 74% | 30% | 66% | 11.0% |
| B (Neutral k=3) | 58.7% | 92% | 21% | 63% | 9.3% |
| C (DUT k=1) | 63.0% | 77% | 44% | 68% | 6.3% |
| C (DUT k=3) | 81.3% | 96% | 82% | 66% | 2.0% |

- k=3 minus k=1: +18.3pp, 95% CI [+12.0%, +24.7%], **excludes zero**
- k=1 minus B: +4.3pp, 95% CI [-2.3%, +11.0%], does not exclude zero
- k=1 minus A: +6.3pp, 95% CI [-0.7%, +13.0%], does not exclude zero

## Key Observations

1. **Model-dependent response to k**: The effect of increasing k from 1 to 3 differs dramatically between models:
   - **Qwen**: k=1 captures most of the DUT benefit (92.3% vs 95.0%), with the k3-k1 difference (+2.7pp) not statistically significant.
   - **Llama**: k=3 is substantially better than k=1 (81.3% vs 63.0%), with a highly significant +18.3pp difference.

2. **k=1 alone is insufficient for Llama**: With k=1, Llama only achieves 63.0% — barely above the baselines (A=56.7%, B=58.7%). The k=1 vs B difference (+4.3pp) does not exclude zero. This suggests that Llama requires repeated definition binding to internalize the convention.

3. **k=1 is nearly sufficient for Qwen**: Qwen achieves 92.3% with k=1, already above the baselines (A=90.3%, B=90.0%), though the improvement is not statistically significant on its own. The additional checks (k=3) provide a modest further boost.

4. **Completeness family drives the k-effect on Llama**: The largest k=1 vs k=3 gap on Llama is in completeness (44% vs 82%, +38pp), suggesting that the hardest convention-binding task benefits most from repeated reinforcement.

5. **Interpretation**: For a strong base model (Qwen), DUT is lightweight — even a single discriminative check captures most of the benefit. For a weaker model (Llama), repeated definition binding through multiple checks is essential. This suggests DUT's mechanism involves progressive convention anchoring, and the number of checks needed scales with the model's baseline adherence.

## Output Files

- Results JSON: `results/analysis/k_ablation.json`
- Bar chart: `results/figures/k_ablation.png`
- k=1 Qwen outputs: `dut_project/outputs/qwen25_math_7b/condition_c_k1.jsonl`
- k=1 Llama outputs: `dut_project/outputs/llama31_8b/condition_c_k1.jsonl`
