# Per-Family Accuracy and Error Breakdown Analysis

## Experiment Overview

Analyzed DUT effectiveness broken down by convention family (convolution, asymptotics, completeness) and performed detailed error categorization under Condition C (k=3 DUT) on both models.

## Setup

- **Models**: Qwen2.5-Math-7B-Instruct, Llama-3.1-8B-Instruct
- **Conditions**: A (glossary-only), B (neutral checks), C (discriminative DUT, k=3)
- **Families**: asymptotics (100 items), completeness (100 items), convolution (100 items)
- **Scoring**: Qwen uses boxed+first-line extraction; Llama uses robust extraction (boxed + heuristic fallbacks)
- **Check accuracy**: N/A for all conditions (V3 verified-facts design provides check answers as demonstrations)

## Key Results

### Per-Family Main Accuracy (%)

| Family | Model | Cond A | Cond B | Cond C | C-B |
|--------|-------|--------|--------|--------|-----|
| Asymptotics | Qwen | 99 | 97 | 99 | +2 |
| Asymptotics | Llama | 74 | 92 | 96 | +4 |
| Completeness | Qwen | 80 | 79 | 91 | +12 |
| Completeness | Llama | 30 | 21 | 82 | +61 |
| Convolution | Qwen | 92 | 94 | 95 | +1 |
| Convolution | Llama | 66 | 63 | 66 | +3 |

### Per-Family Alternate-Convention Match Rate (%)

| Family | Model | Cond A | Cond B | Cond C |
|--------|-------|--------|--------|--------|
| Asymptotics | Qwen | 1.0 | 2.0 | 0.0 |
| Asymptotics | Llama | 18.0 | 0.0 | 1.0 |
| Completeness | Qwen | 16.0 | 18.0 | 4.0 |
| Completeness | Llama | 11.0 | 24.0 | 0.0 |
| Convolution | Qwen | 2.0 | 1.0 | 0.0 |
| Convolution | Llama | 4.0 | 4.0 | 5.0 |

### Error Breakdown Under Condition C

| Family | Model | Correct | Convention Err | Arith/Reasoning Err | Parsing Fail |
|--------|-------|---------|----------------|---------------------|--------------|
| Asymptotics | Qwen | 99% | 0% | 1% | 0% |
| Asymptotics | Llama | 96% | 1% | 1% | 2% |
| Completeness | Qwen | 91% | 4% | 5% | 0% |
| Completeness | Llama | 82% | 0% | 0% | 18% |
| Convolution | Qwen | 95% | 0% | 5% | 0% |
| Convolution | Llama | 66% | 5% | 29% | 0% |

## Key Observations

1. **Completeness is the most DUT-responsive family**: DUT (C) produces the largest gains over B in completeness (+12pp Qwen, +61pp Llama). This family has the highest baseline convention confusion (alternate match rates of 16-24% under A/B).

2. **Convolution is the most resistant to DUT on Llama**: Llama shows no improvement from DUT on convolution (66% for both A and C). The 29% arithmetic/reasoning error rate indicates the difficulty is computational, not convention-related.

3. **Asymptotics is near-ceiling for Qwen**: Qwen achieves 99% on asymptotics under all conditions, leaving no room for DUT improvement.

4. **Error types differ sharply by family and model**:
   - **Qwen**: Errors are split between convention errors (completeness: 4%) and arithmetic errors (convolution: 5%, completeness: 5%). Zero parsing failures.
   - **Llama**: Dominated by parsing failures on completeness (18%) and arithmetic errors on convolution (29%). Convention errors are rare under C (0-5%).

5. **DUT nearly eliminates convention confusion**: Under Condition C, alternate-convention match rates drop to 0-5% across all families on both models, compared to 0-24% under baseline conditions.

6. **Remaining Llama errors on convolution are computational**: The 29% arithmetic error rate on convolution under C indicates that Llama struggles with the numeric computation required for additive convolution problems, not with convention adherence.
