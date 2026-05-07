# Condition A: Opinion & Instruction (O&I) Baseline on ConFiQA-MC

## Experiment Overview

Evaluated the O&I context-faithful prompting baseline (Zhou et al., 2023) on Llama-3.1-8B-Instruct using the full ConFiQA-MC dataset (6,000 examples). This is the strongest known prompting baseline for context faithfulness, used in Context-DPO and ContextFocus evaluations.

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct (frozen, bf16)
- **Benchmark**: ConFiQA-MC (6,000 multi-hop counterfactual QA instances)
- **Prompt Template**: O&I (instruction + opinion), matching Context-DPO `instr+opin` schema
- **Decoding**: Greedy (temperature=0, max_tokens=32)
- **Inference**: vLLM offline batch inference, 1x A100-80GB
- **Evaluation**: ConFiQA alias-matching scorer (recall with negation filtering)

### Prompt Format

```
Instruction: read the given information and answer the corresponding question.

Bob said "{cf_context}"
Q: {question} in Bob's opinion?
A:
```

System message: "Answer concisely with the answer only."

## Key Results

| Metric | Value |
|--------|-------|
| Pc (context-faithful rate) | 52.43 |
| Po (original/parametric answer rate) | 13.40 |
| MR (memorization ratio = Po/(Pc+Po)*100) | 20.35 |
| EM (exact match) | 49.43 |

## Key Observations

- Pc of 52.43% means the model follows the counterfactual context slightly more than half the time with O&I prompting.
- MR of 20.35% indicates that when the model produces a recognizable answer, about 1 in 5 times it falls back to parametric memory.
- EM of 49.43% is lower than Pc because it requires exact match (not just recall).
- These numbers serve as the baseline that conditions B and C must improve upon.
- Reference: Context-DPO reports Pc=54.9, MR=27.9, EM=21.9 for Qwen2-7B on ConFiQA-MC (training-based, different model).
