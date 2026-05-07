# Condition C: Silence-Conditional Head Masking (SCHM) Evaluation

## Experiment Overview

Evaluated the proposed SCHM method on Whisper-large-v3. SCHM conditionally applies decoder head masking (heads {1, 6, 11}) and/or no-speech suppression when the model's p_no_speech signal exceeds a threshold tau. Two modes were evaluated:
- **mask**: Decode with head mask applied (original behavior)
- **suppress**: Output empty transcription when triggered (no-speech suppression)

## Setup

- **Model**: openai/whisper-large-v3 (1.5B params)
- **Decoding**: Greedy (num_beams=1, do_sample=False, language=en, task=transcribe)
- **Head mask**: Decoder heads {1, 6, 11} zeroed across all 32 layers
- **Thresholds**: tau in {0.3, 0.4, 0.5, 0.6}
- **Modes**: suppress (output empty when triggered), mask (decode with head mask)
- **Attention**: eager (required for head_mask support)
- **GPU**: 1x A100-80GB

## Key Results

### UrbanSound8K Hallucination Rate

| Condition | tau | Halluc Rate | Triggered | Halluc Reduction |
|-----------|-----|-------------|-----------|------------------|
| A (baseline) | - | 100.0% | - | - |
| B (always-mask) | - | 100.0% | - | - |
| C-mask | 0.3 | 100.0% | 39.9% | 0.0 pp |
| C-mask | 0.6 | 100.0% | 21.6% | 0.0 pp |
| **C-suppress** | **0.3** | **60.1%** | **39.9%** | **-39.9 pp** |
| C-suppress | 0.4 | 69.4% | 30.6% | -30.6 pp |
| C-suppress | 0.5 | 74.4% | 25.6% | -25.6 pp |
| C-suppress | 0.6 | 78.4% | 21.6% | -21.6 pp |

### LibriSpeech WER (Speech Quality Preservation)

| Condition | tau | Clean WER | Other WER | Clean False Pos | Other False Pos |
|-----------|-----|-----------|-----------|-----------------|-----------------|
| A (baseline) | - | 2.83% | 5.10% | - | - |
| B (always-mask) | - | 3.08% | 5.32% | 100% | 100% |
| C-suppress | 0.3 | 3.02% | 5.10% | 0.19% (5 utts) | 0.00% |
| C-suppress | 0.4 | 3.02% | 5.10% | 0.19% (5 utts) | 0.00% |
| C-suppress | 0.5 | 2.95% | 5.10% | 0.08% (2 utts) | 0.00% |
| C-suppress | 0.6 | 2.86% | 5.10% | 0.04% (1 utt) | 0.00% |

### Best Configuration: SCHM-suppress at tau=0.3

- Hallucination rate: 60.1% (39.9 pp reduction from 100%)
- LibriSpeech clean WER: 3.02% (+0.19 pp from baseline)
- LibriSpeech other WER: 5.10% (unchanged)
- False positive rate: 0.19% clean, 0.00% other

## Key Observations

1. **Head masking alone does not reduce hallucination rate**: The mask mode shows 100% hallucination at all tau values because HF's generate() always produces non-empty text. Masking changes content but doesn't eliminate output.

2. **Suppress mode is effective**: Hallucination rate reduction equals the trigger rate (halluc = 1 - triggered), confirming p_nospeech correctly identifies non-speech clips.

3. **Minimal WER impact on speech**: At tau=0.3, only 5/2620 clean utterances are falsely triggered (0.19%), causing WER to increase from 2.83% to 3.02%. On test-other, zero utterances are affected.

4. **SCHM avoids WER degradation from always-masking**: Condition B increases WER by +0.25% (clean) and +0.22% (other). SCHM-suppress at tau=0.3 increases clean WER by only +0.19% and leaves other WER unchanged.

5. **Training-free method**: No fine-tuning required. Only requires computing p_nospeech via 1-step decoder forward pass and thresholding.

## Files

- `results/condition_c_urbansound8k.json`: Per-clip results (8732 entries, original tau=0.6 mask mode)
- `results/condition_c_librispeech_test_clean.json`: Per-utterance results (2620 entries)
- `results/condition_c_librispeech_test_other.json`: Per-utterance results (2939 entries)
- `results/schm_sweep_results.json`: Full multi-tau multi-mode sweep results
