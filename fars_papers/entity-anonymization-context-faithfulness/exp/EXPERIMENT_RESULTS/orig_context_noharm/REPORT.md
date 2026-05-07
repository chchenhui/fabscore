# Orig-Context No-Harm Control Experiment

## Experiment Overview

This experiment evaluates whether entity anonymization (condition C / EACP) harms basic QA accuracy when there is **no knowledge conflict** -- i.e., when the provided context aligns with the model's parametric memory. This is a required confound control: if EACP hurts accuracy even when the model could answer correctly from either context or memory, the anonymization is introducing unacceptable noise.

## Setup

- **Model**: Llama-3.1-8B-Instruct
- **Benchmark**: ConFiQA-MC, 500-example subset (seed=123)
- **Context**: `orig_context` used instead of `cf_context` (no knowledge conflict)
- **Gold answer**: `orig_answer` + aliases (aligned with parametric memory)
- **Decoding**: Greedy, max_new_tokens=32
- **Scoring**: EM = exact match against orig_answer/aliases (is_cf=False, no negation filter). Context-follow rate = recall of orig_answer in response.
- **Conditions**:
  - A (O&I): Standard opinion-and-instruction template with natural-language answer
  - B (Inventory+IDs): Entity inventory with real names, ID-based answer
  - C (EACP): Anonymized context, no-name inventory, ID-based answer

## Key Results

| Condition | EM | Context-Follow Rate | ID Parse Rate |
|-----------|---:|-------------------:|-------------:|
| A (O&I) | 66.40 | 79.00 | -- |
| B (Inventory+IDs) | 79.80 | 81.00 | 99.80 |
| **C (EACP)** | **88.00** | **88.40** | **99.80** |

### No-Harm Criterion: **PASS**

The success criterion is: "C's EM should not drop by more than 2 points compared to A or B."

- C's EM (88.00) is **21.6 points above A** (66.40) and **8.2 points above B** (79.80).
- C does not drop below either condition. The no-harm criterion is satisfied.

## Key Observations

1. **EACP improves accuracy even without knowledge conflict.** C outperforms both A and B on EM. This is because:
   - The ID-based output format (shared by B and C) constrains answers to entity IDs, avoiding verbose free-text answers that fail exact match even when semantically correct.
   - Anonymization in C further helps by removing entity name distractions -- the model focuses purely on contextual relationships.

2. **Condition A underperforms due to free-text format.** A's EM is only 66.4% despite a 79.0% context-follow rate. Many responses contain the correct answer but fail exact match due to extra words, explanations, or formatting differences.

3. **Error patterns in C (60 errors out of 500):**
   - Most errors are UNKNOWN responses where the entity was not found in the anonymized inventory or context.
   - A few cases involve entity mismatches where the model selected the wrong entity ID.
   - No evidence that anonymization systematically confuses the model.

4. **ID parse success rate is near-perfect (99.8%) for both B and C**, confirming the model reliably outputs entity IDs when instructed.

5. **Conclusion**: Entity anonymization does not harm basic QA accuracy. In fact, the structured ID-based output format and anonymization appear to provide a beneficial constraint that improves exact-match scoring. This validates EACP as a safe intervention.
