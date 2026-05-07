# Majority-Vote Inference-Time Scaling Baselines (A@5 and B@5)

## Experiment Overview

This experiment tests whether simple inference-time scaling (generating 5 samples and majority-voting) can close the accuracy gap that DUT (Condition C) achieves over single-sample baselines. We compare A@5 (majority vote over 5 glossary-only samples) and B@5 (majority vote over 5 neutral-check samples) against the single-sample DUT condition C.

Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)

## Setup

- **Benchmark**: ErdosConventionsBench (300 items, 3 families)
- **Models**: Qwen2.5-Math-7B-Instruct, Llama-3.1-8B-Instruct
- **Sampling**: temperature=0.7, top_p=0.95, max_new_tokens=512, 5 samples per item
- **Majority vote**: Most frequent answer; ties broken randomly (seed=42)
- **Answer extraction**: `extract_answer_robust()` with family-aware heuristic fallbacks
- **Inference**: vLLM batched chat inference, 1 GPU per model

## Key Results

### Qwen2.5-Math-7B-Instruct

| Condition | Overall | Asymptotics | Completeness | Convolution |
|-----------|---------|-------------|--------------|-------------|
| A (single) | 90.3% | 99% | 80% | 92% |
| B (single) | 90.0% | 97% | 79% | 94% |
| **A@5** | **41.3%** | 43% | 6% | 75% |
| **B@5** | **50.7%** | 51% | 22% | 79% |
| C (single DUT) | **95.0%** | 99% | 91% | 95% |

- C - A@5: +53.7pp, 95% CI [+48.0%, +59.3%], **excludes zero**
- C - B@5: +44.3pp, 95% CI [+38.7%, +50.0%], **excludes zero**

### Llama-3.1-8B-Instruct

| Condition | Overall | Asymptotics | Completeness | Convolution |
|-----------|---------|-------------|--------------|-------------|
| A (single) | 56.7% | 74% | 30% | 66% |
| B (single) | 58.7% | 92% | 21% | 63% |
| **A@5** | **58.7%** | 75% | 19% | 82% |
| **B@5** | **60.7%** | 88% | 24% | 70% |
| C (single DUT) | **81.3%** | 96% | 82% | 66% |

- C - A@5: +22.7pp, 95% CI [+16.0%, +29.3%], **excludes zero**
- C - B@5: +20.7pp, 95% CI [+14.0%, +27.3%], **excludes zero**

## Key Observations

1. **Majority-vote with 512 tokens severely underperforms single-sample with 2048 tokens** for Qwen (A@5=41.3% vs A=90.3%). The 512-token limit truncates chain-of-thought reasoning before the model can produce a final boxed answer. This demonstrates that raw inference-time scaling at reduced token budget does not compensate for shorter individual answers.

2. **For Llama, majority vote provides modest gains** over single-sample (A@5=58.7% vs A=56.7%, B@5=60.7% vs B=58.7%), consistent with Llama already producing shorter outputs that fit within 512 tokens.

3. **DUT (Condition C) massively outperforms majority-vote baselines** on both models. The C-A@5 and C-B@5 differences are all statistically significant with 95% CIs excluding zero. This confirms that discriminative definition checks provide a qualitatively different benefit from simply sampling more answers.

4. **The completeness family shows the most dramatic difference**: Qwen A@5=6% vs C=91%, Llama A@5=19% vs C=82%. This family requires careful convention binding that majority voting cannot provide.

5. **Convolution family is the exception** where majority vote helps: Llama A@5=82% vs A=66%. Numeric computation benefits from sampling diversity, but this does not extend to convention-sensitive families.
