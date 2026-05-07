# Condition D: ContextFocus Activation Steering on Llama-3.1-8B-Instruct

## Experiment Overview

Implemented and evaluated ContextFocus (Anand et al., 2026) activation steering on Llama-3.1-8B-Instruct using a 1,500-example random subset of ConFiQA-MC with the O&I prompt template. ContextFocus constructs a steering vector by contrasting model activations on prompts with vs. without context, then adds this vector to the residual stream at an intermediate layer during inference.

**Reference**: "ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models" (Anand et al., 2026) | https://arxiv.org/abs/2601.04131

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct (bf16)
- **Dataset**: ConFiQA-MC, 1,500-example subset (seed=42)
- **Steering vector data**: NQ-SWAP (pminervini/NQ-Swap, dev split), 1,501 examples for vector estimation, 200 held-out for layer selection
- **Steering vector format**: Paper format ("Context: ... Question: ...") with 20 system instruction variants
- **Layer**: 13 (paper recommendation)
- **Multiplier**: m=2.0 (paper specification), also tested m=1.0, m=0.5
- **Decoding**: Greedy, max_new_tokens=32
- **Inference**: HuggingFace model.generate() with forward hooks (not vLLM-compatible)

## Key Results

### Primary Result (m=2.0, paper spec)

| Condition | Pc | Po | MR | EM |
|-----------|----|----|----|----|
| A (O&I baseline) | 52.40 | 12.93 | 19.80 | 50.07 |
| B (Inventory+IDs) | 30.07 | 20.40 | 40.42 | 35.93 |
| **D (ContextFocus m=2.0)** | **23.20** | **16.20** | **41.12** | **2.33** |

### Alternative Multipliers

| Multiplier | Pc | Po | MR | EM |
|------------|----|----|----|----|
| m=2.0 (paper) | 23.20 | 16.20 | 41.12 | 2.33 |
| m=1.0 | 49.47 | 19.13 | 27.89 | 6.80 |
| m=0.5 | 54.93 | 19.47 | 26.16 | 6.93 |

### Paper's Reported Results (full ConFiQA-MC)

| Metric | Paper (Ps) | Our m=1.0 (Pc) |
|--------|-----------|----------------|
| Context-faithful rate | 48.60 | 49.47 |
| Parametric answer rate | 10.27 | 19.13 |
| Memorization ratio | 17.44 | 27.89 |

## Key Observations

1. **m=2.0 produces degenerate outputs**: With the paper-specified multiplier of 2.0, the model generates verbose context continuations and repetitive gibberish (e.g., "[B]. [B]. [B]..."). This results in very low EM (2.33%) and Pc drops to 23.2%, well below the unsteered baseline A (52.4%).

2. **Compounding effect during autoregressive generation**: The steering vector is added at every generation step when using KV-cache (each step processes 1 new token). With m=2.0, this compounds over 32 generation steps, causing severe output degradation. Layer selection on NQ-SWAP confirmed this: all 32 layers showed negative Ps improvement with m=2.0.

3. **m=1.0 matches paper's Ps metric**: At m=1.0, our Pc (49.47%) closely matches the paper's reported Ps (48.60%), suggesting the paper may have used a different effective multiplier or a different implementation detail (e.g., not applying steering at every KV-cache step).

4. **Low EM across all multipliers**: Even at m=0.5/1.0 where Pc is reasonable, EM is ~7%. The steered model tends to continue/reproduce context passages rather than giving concise answers. The cf_answer appears embedded within verbose output, so recall-based metrics (Pc, Ps) detect it, but exact match fails.

5. **ContextFocus hurts vs. O&I baseline**: Across all tested configurations, Condition D does not improve over the O&I baseline (Condition A) on any metric. At m=2.0 (paper spec), it substantially degrades both Pc and EM.

## Layer Selection Results

- Baseline Ps (m=0) on NQ-SWAP 200 held-out: 93.0%
- Best layer with m=2.0: Layer 29 (Ps=89.0%, delta=-4.0pp)
- Paper's recommended layer 13: Ps=82.5% (delta=-10.5pp)
- All 32 layers showed negative deltas with m=2.0

## Output Files

- `eacp/outputs/D_llama31_8b_confiqa_mc_1500.jsonl` - m=2.0 outputs (primary)
- `eacp/outputs/D_llama31_8b_confiqa_mc_1500_m1.0.jsonl` - m=1.0 outputs
- `eacp/outputs/D_llama31_8b_confiqa_mc_1500_m0.5.jsonl` - m=0.5 outputs
- `eacp/steering/vectors/llama31_8b_contextfocus.pt` - Steering vectors (32 layers x 4096)
- `eacp/steering/vectors/selected_layer.json` - Layer selection results
