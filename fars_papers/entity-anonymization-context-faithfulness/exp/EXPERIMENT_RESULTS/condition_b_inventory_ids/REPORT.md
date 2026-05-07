# Condition B: Inventory+IDs Control on ConFiQA-MC

## Experiment Overview

Evaluated the Inventory+IDs control condition (B) on Llama-3.1-8B-Instruct using the full ConFiQA-MC dataset (6,000 examples). Condition B prepends an entity inventory block to the O&I prompt that assigns each path entity an ID (ENT_1, ENT_2, ...) with its real name visible (e.g., "ENT_2 = United States (type=country)"), and requires the model to answer using an ID (ENT_k or UNKNOWN). This controls for the effect of structured prompting and ID-constrained output space.

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct (frozen, bf16)
- **Benchmark**: ConFiQA-MC (6,000 multi-hop counterfactual QA instances)
- **Prompt Template**: Entity inventory (real names) + O&I + ID answer constraint
- **Decoding**: Greedy (temperature=0, max_tokens=32)
- **Inference**: vLLM offline batch inference, 1x A100-80GB
- **Evaluation**: ConFiQA alias-matching scorer (recall with negation filtering)
- **ID Parsing**: Regex extraction of first ENT_\d+ or UNKNOWN from model output

### Prompt Format

```
Entity Inventory:
- ENT_1 = {EntityName_1} (type={TYPE_1})
- ENT_2 = {EntityName_2} (type={TYPE_2})
...

Instruction: read the given information and answer the corresponding question. Answer with exactly one entity ID (e.g., ENT_1) or UNKNOWN. Do not explain.

Bob said "{cf_context}"
Q: {question} in Bob's opinion?
A:
```

System message: "Answer concisely with the answer only."

## Key Results

| Metric | Condition B | Condition A (baseline) | Delta |
|--------|------------|----------------------|-------|
| Pc (context-faithful rate) | 32.47 | 52.43 | -19.96 |
| Po (original/parametric answer rate) | 19.75 | 13.40 | +6.35 |
| MR (memorization ratio) | 37.82 | 20.35 | +17.47 |
| EM (exact match) | 37.95 | 49.43 | -11.48 |
| ID parse success rate | 99.95 | N/A | -- |

## Key Observations

- Condition B performs substantially worse than condition A across all metrics. Pc drops from 52.43 to 32.47 and MR rises from 20.35 to 37.82.
- The 99.95% ID parse success rate confirms the model reliably produces ENT_k/UNKNOWN outputs when instructed.
- The degradation is likely because: (1) the inventory block with real names actually reinforces entity associations, (2) the ID constraint adds complexity that confuses the model's reasoning, and (3) many instances produce UNKNOWN instead of identifying the correct entity.
- This establishes an important baseline: any improvement of condition C (anonymized) over B is attributable to entity anonymization itself, not to the inventory structure or shorter ID-based answers.
- The fact that B is worse than A suggests structured ID-based prompting alone is harmful, making C vs B the critical comparison for isolating anonymization effects.
