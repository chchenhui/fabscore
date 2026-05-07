# Qwen2.5-7B-Instruct A/B/C Replication

## Experiment Overview

Replicate the core A/B/C comparison from Llama-3.1-8B-Instruct on Qwen2.5-7B-Instruct to test whether the EACP effect generalizes across model families. Qwen2.5-7B-Instruct is independently trained with different pretraining data and architecture choices (GQA, RoPE base, SwiGLU, different tokenizer).

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
- **Dataset**: ConFiQA-MC (6,000 multi-hop knowledge-conflict examples)
- **Conditions**:
  - A: O&I baseline (greedy decoding)
  - B: Inventory+IDs control (greedy decoding)
  - C: EACP with self-consistency (n=5, temp=0.7, majority vote)
- **Inference**: bfloat16, max_new_tokens=32, 1xA100-80GB, vLLM
- **Chat template**: vLLM auto-applies Qwen's `<|im_start|>` format from tokenizer config. No prompt builder changes needed -- same system/user message structure as Llama.

## Key Results

### Per-Model Results (ConFiQA-MC, 6,000 examples)

| Model | Condition | Pc | Po | MR | EM |
|-------|-----------|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | A (O&I) | 52.43 | 13.40 | 20.35 | 49.43 |
| Llama-3.1-8B-Instruct | B (Inv+IDs) | 32.47 | 19.75 | 37.82 | 37.95 |
| Llama-3.1-8B-Instruct | **C (EACP)** | **74.75** | **10.77** | **12.59** | **81.80** |
| Qwen2.5-7B-Instruct | A (O&I) | 53.73 | 14.02 | 20.69 | 50.32 |
| Qwen2.5-7B-Instruct | B (Inv+IDs) | 27.23 | 19.03 | 41.14 | 31.78 |
| Qwen2.5-7B-Instruct | **C (EACP)** | **76.43** | **12.38** | **13.94** | **83.73** |

### Cross-Model C-vs-B Deltas

| Model | Delta Pc | Delta Po | Delta MR | Delta EM |
|-------|-------:|-------:|-------:|-------:|
| Llama-3.1-8B-Instruct | +42.28 | -8.98 | -25.23 | +43.85 |
| Qwen2.5-7B-Instruct | **+49.20** | **-6.65** | **-27.20** | **+51.95** |

## Key Observations

1. **EACP generalizes across model families.** Both Llama and Qwen show massive C-vs-B improvements, confirming the mechanism (entity anonymization reducing parametric recall) is not model-specific.

2. **Qwen benefits even more from EACP.** The C-vs-B Pc delta is +49.20 for Qwen vs +42.28 for Llama (+6.92 larger). This may indicate Qwen has stronger parametric recall that is more effectively disrupted by anonymization.

3. **Baseline behavior is consistent.** Both models show similar A performance (Pc ~52-54), confirming the O&I template works across architectures. Both show B degradation (Pc drops to 27-32), confirming the ID format without anonymization hinders context following.

4. **Absolute Pc levels are similar.** Qwen C achieves Pc=76.43 vs Llama C=74.75, a difference of only +1.68. The models converge to similar performance under EACP despite different training.

5. **MR reduction is consistent.** Llama MR: 37.82 -> 12.59 (-25.23). Qwen MR: 41.14 -> 13.94 (-27.20). Both drop to ~13% memory ratio under EACP.

6. **Qwen B has lower parse rate.** Qwen B id_parse_success_rate=98.13% vs Llama B=99.95%. Qwen is slightly less compliant with the ID output format when entity names are visible, but this is fully resolved under anonymization (C parse rate=99.87%).

## Conclusion

The EACP effect is robust across independently trained model families. Consistent results on both Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct strengthen the mechanistic claim that entity anonymization reduces entity-triggered parametric recall, as this mechanism does not depend on specific pretraining data or architecture details.
