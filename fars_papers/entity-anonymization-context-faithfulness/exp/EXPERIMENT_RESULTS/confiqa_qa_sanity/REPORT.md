# ConFiQA-QA Sanity Check: EACP Generalization to Single-Hop Conflicts

## Experiment Overview

Evaluate conditions A (O&I baseline), B (Inventory+IDs control), and C (EACP) on a 500-example subset of ConFiQA-QA to check whether EACP generalizes to simpler single-hop, single-conflict QA. ConFiQA-QA has single-hop questions with a single counterfactual entity swap, which is an easier conflict setting than ConFiQA-MC (multi-hop, multiple conflicts).

## Setup

- **Model**: Llama-3.1-8B-Instruct
- **Dataset**: ConFiQA-QA, 500 examples (seed=42)
- **Decoding**: Greedy (temperature=0), max_new_tokens=32
- **Conditions**: A (O&I), B (Inventory+IDs), C (EACP)
- **GPU**: 1x A100-80GB via TrainService

## Key Results

### ConFiQA-QA (500 examples, single-hop)

| Condition | Pc | Po | MR | EM |
|-----------|---:|---:|---:|---:|
| A (O&I) | 82.20 | 6.60 | 7.43 | 75.60 |
| B (Inventory+IDs) | 65.00 | 11.20 | 14.70 | 65.60 |
| **C (EACP)** | **87.80** | **1.80** | **2.01** | **88.40** |

### C-vs-B Delta (EACP anonymization effect)

| Split | delta-Pc | delta-Po | delta-MR | delta-EM |
|-------|--------:|--------:|--------:|--------:|
| QA-500 | +22.80 | -9.40 | -12.69 | +22.80 |
| MC-6000 | +42.28 | -8.98 | -25.23 | +43.85 |
| **QA-MC diff** | **-19.48** | **-0.42** | **+12.54** | **-21.05** |

## Key Observations

1. **EACP generalizes to single-hop QA**: Condition C achieves Pc=87.8 and MR=2.01 on QA, confirming EACP works in simpler conflict settings. The anonymization effect (C vs B: +22.8 Pc, -12.69 MR) is substantial and consistent with MC findings.

2. **QA is inherently easier than MC**: All conditions perform better on QA than MC. Condition A already achieves Pc=82.2 on QA vs 52.43 on MC. The single-hop, single-conflict structure provides a simpler task with less room for ambiguity.

3. **EACP benefit is larger on MC (multi-hop)**: The C-vs-B Pc delta is +42.28 on MC vs +22.80 on QA (a 19.48pp difference). This suggests multi-hop conflicts with more entity interactions benefit more from anonymization. When there are multiple interleaved entities (MC), the model's parametric recall of entity associations is stronger and harder to override -- anonymization breaks more of those associations, yielding a bigger improvement.

4. **Ceiling effect on QA**: The smaller EACP delta on QA is partly due to higher baselines (A=82.2, B=65.0 on QA vs A=52.43, B=32.47 on MC). With less room to improve, the absolute gains are naturally smaller. The relative reduction in MR is still large: from 14.7 to 2.01 (-86% relative).

5. **ID parsing works well on QA**: Both B (99.4%) and C (99.0%) achieve near-perfect ID parsing, consistent with MC performance.

6. **Conclusion for mechanism hypothesis**: Single-entity conflicts are more amenable to all prompting strategies (higher A baseline), but multi-entity conflicts benefit *more* from anonymization specifically. This supports the hypothesis that EACP's primary mechanism is breaking entity-association recall patterns, which are more numerous and stronger in multi-hop settings.
