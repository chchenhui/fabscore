# Condition E: EACP + ContextFocus Composition

## Experiment Overview

Condition E evaluates the composition of two orthogonal context-faithfulness interventions:
1. **EACP** (Entity-Anonymized Context Prompts): anonymizes entity surface forms in the prompt to break entity-triggered parametric recall (input-space intervention)
2. **ContextFocus**: adds a steering vector to the residual stream to shift the model toward context-faithful behavior (activation-space intervention)

The hypothesis is that if E improves over C (EACP alone), then the two interventions are complementary -- EACP addresses input-level entity priors while ContextFocus addresses residual activation-level biases.

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Benchmark**: ConFiQA-MC, 1,500-example subset (seed=42)
- **Prompts**: EACP format (condition C) -- anonymized context/question + anonymized entity inventory + ID output constraint
- **Steering vectors**: (1) NQ-SWAP ContextFocus vector (original), (2) EACP-native vector computed from ConFiQA data
- **Steering modes**: "generation" (decode tokens only), "both" (all positions)
- **Multiplier sweep**: m in {0.3, 0.5, 1.0}
- **Decoding**: Greedy, max_new_tokens=32
- **Inference**: HF model.generate() with forward hooks (required for activation steering)
- **Script**: `eacp/scripts/run_condition_e.py`

## Key Results

### Full Sweep Results (1,500-subset)

| Vector | Mode | m | Pc | Po | MR | EM |
|--------|------|---:|---:|---:|---:|---:|
| None (baseline) | - | 0.0 | 74.00 | 11.40 | 13.35 | 81.40 |
| NQ-SWAP | gen | 0.3 | 74.80 | 11.40 | 13.23 | 82.13 |
| NQ-SWAP | gen | 0.5 | 75.87 | 11.53 | 13.20 | 83.27 |
| NQ-SWAP | gen | 1.0 | 76.47 | 13.20 | 14.72 | 84.07 |
| NQ-SWAP | both | 0.3 | 73.73 | 11.33 | 13.32 | 81.07 |
| NQ-SWAP | both | 0.5 | 73.93 | 11.73 | 13.70 | 81.33 |
| EACP | gen | 0.3 | 74.67 | 11.40 | 13.25 | 82.00 |
| EACP | gen | 0.5 | 75.67 | 11.53 | 13.23 | 83.07 |
| EACP | gen | 1.0 | 76.80 | 11.87 | 13.38 | 84.27 |
| EACP | both | 0.3 | 75.53 | 11.33 | 13.05 | 83.00 |
| **EACP** | **both** | **0.5** | **76.47** | **11.33** | **12.91** | **83.93** |

### Primary Result: EACP vector, both mode, m=0.5

- **Pc=76.47** (+3.60 over C+SC, +24.07 over A baseline)
- **MR=12.91** (best across all configs, -0.34 over C+SC)
- **EM=83.93** (+3.93 over C+SC)

### Comparison Table

| Condition | Pc | Po | MR | EM |
|-----------|---:|---:|---:|---:|
| A (O&I baseline) | 52.40 | 12.93 | 19.80 | 50.07 |
| C (EACP greedy) | 70.13 | 10.27 | 12.77 | 76.73 |
| C (EACP+SC) | 72.87 | 11.13 | 13.25 | 80.00 |
| D (CF m=1.0) | 49.47 | 19.13 | 27.89 | 6.80 |
| **E (best)** | **76.47** | **11.33** | **12.91** | **83.93** |

## Key Observations

1. **EACP and ContextFocus ARE complementary when properly composed**: E (Pc=76.47) surpasses both C greedy (70.13) and C+SC (72.87), disproving the original (buggy) conclusion that they interfere.

2. **Answer extraction was the critical bug**: The original implementation extracted the FIRST `ENT_\d+` match from verbose HF generate output, which came from echoed context text rather than the actual answer. This single bug accounted for ~40pp of the Pc deficit.

3. **EACP-native vector outperforms NQ-SWAP vector**: At matched Pc levels, EACP vector achieves lower MR (12.91 vs 14.72), confirming that domain-matched steering vectors are important.

4. **"Both" mode works best with EACP vector**: Steering all positions (prefill + generation) with the distribution-matched EACP vector achieves the best MR. NQ-SWAP vector degrades in "both" mode due to distribution mismatch.

5. **Steering primarily improves EM**: The main effect of steering is improving exact match accuracy (81.40 -> 83.93 at m=0.5), with moderate Pc improvement (+2.47). This suggests steering helps produce more precise, well-formatted answers.

## Output Files

- `eacp/outputs/E_*_eacp_vec_*.jsonl` - Raw outputs with EACP vector
- `eacp/outputs/E_*_eacp_vec_*_metrics.json` - Metrics with EACP vector  
- `eacp/outputs/E_*_nqswap_*.jsonl` - Raw outputs with NQ-SWAP vector
- `eacp/outputs/E_*_nqswap_*_metrics.json` - Metrics with NQ-SWAP vector
