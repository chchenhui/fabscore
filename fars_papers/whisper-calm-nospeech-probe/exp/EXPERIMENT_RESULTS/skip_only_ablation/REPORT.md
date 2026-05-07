# Skip-Only Ablation: Disentangling Trigger Policy from Head Masking

## Experiment Overview

This ablation tests whether SCHM's hallucination reduction comes from the p_no_speech trigger policy or the head masking mechanism. Condition D (Skip-Only) replaces head-masked decoding with empty output when p_no_speech >= tau, using default Whisper transcription otherwise. No GPU inference required -- purely post-hoc re-labeling of Condition A results.

## Setup

- **Model**: openai/whisper-large-v3
- **Threshold**: tau = 0.6
- **Datasets**: UrbanSound8K (8732 clips), LibriSpeech test-clean (2620 utts), test-other (2939 utts)
- **Method**: For each clip, load pre-computed p_no_speech. If p_no_speech >= 0.6, output empty string; otherwise use Condition A transcription.
- **Script**: `schm/inference/run_skip_only.py`

## Key Results

| Method | US8K Halluc Rate | LS-clean WER | LS-other WER |
|--------|---:|---:|---:|
| A: Default Whisper | 100.00% | 2.83% | 5.10% |
| B: Always-Mask | 100.00% | 3.08% | 5.32% |
| C: SCHM (tau=0.6, suppress) | 78.37% | 2.86% | 5.10% |
| D: Skip-Only (tau=0.6) | **78.37%** | **2.86%** | **5.10%** |

Skip-Only and SCHM-suppress produce **identical results** on all three metrics.

## Key Observations

1. **Trigger alone suffices for non-speech**: Skip-Only matches SCHM-suppress exactly (78.37% halluc rate), confirming the p_no_speech threshold is the sole mechanism reducing hallucinations. The 21.63 pp reduction from baseline comes entirely from correctly identifying and suppressing non-speech clips.

2. **No WER difference on speech**: Both Skip-Only and SCHM-suppress yield identical WER (2.86% clean, 5.10% other). Only 1 utterance on test-clean is triggered (false positive); 0 on test-other. For non-triggered clips, both use default Whisper output -- head masking is never applied.

3. **Head masking is not contributing**: In the SCHM-suppress pipeline, when p_no_speech >= tau, the clip is suppressed before head-masked decoding runs. For non-triggered clips, no masking is applied. Thus head masking is effectively unused in suppress mode.

4. **Head masking alone (Condition B) hurts**: Always-mask degrades WER (+0.25 pp clean, +0.22 pp other) without reducing hallucinations at all. HF generate() always produces text regardless of head masks.

5. **Disentanglement conclusion**: SCHM's value comes entirely from the trigger policy (conditioning on p_no_speech), not from the head masking mechanism. For non-speech detection, the binary skip decision is necessary and sufficient.
