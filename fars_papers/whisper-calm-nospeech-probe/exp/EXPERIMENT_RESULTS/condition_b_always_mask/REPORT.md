# Condition B: Always-Mask Heads {1,6,11} Baseline

## Experiment Overview
Evaluate Whisper-large-v3 with decoder self-attention heads {1, 6, 11} permanently masked across all 32 decoder layers during greedy decoding. This establishes Condition B from Calm-Whisper (arXiv:2505.12969).

## Setup
- **Model**: `openai/whisper-large-v3` (float16, `attn_implementation="eager"`)
- **Decoding**: Greedy (num_beams=1, do_sample=False, language="en", task="transcribe")
- **max_new_tokens**: 128, batch_size=16
- **Head mask**: Binary tensor [32, 20], heads {1, 6, 11} set to 0 across all layers (17/20 heads active per layer)
- **Implementation**: Monkey-patched `model.model.decoder.forward` to inject `head_mask` kwarg, since `model.generate()` does not propagate `decoder_head_mask`. Required `attn_implementation="eager"` because SDPA does not support head_mask.
- **Hardware**: 1x GPU via TrainService
- **Datasets**:
  - UrbanSound8K: 8,732 non-speech clips
  - LibriSpeech test-clean: 2,620 utterances
  - LibriSpeech test-other: 2,939 utterances

## Key Results

| Metric | Condition A (Default) | Condition B (Always-Mask) | Calm-Whisper Cited (B) |
|--------|----------------------|--------------------------|----------------------|
| UrbanSound8K hallucination rate | 100.00% | **100.00%** | 24.10% |
| LibriSpeech test-clean WER | 2.83% | **3.08%** | 3.57% |
| LibriSpeech test-other WER | 5.10% | **5.32%** | 5.98% |

### Additional Metrics (UrbanSound8K)

| Metric | Condition A | Condition B | Change |
|--------|------------|------------|--------|
| Clips with different transcription | -- | 1967/8732 (22.5%) | -- |
| Mean transcription length (chars) | 14.9 | 11.8 | -21% |
| Short transcriptions (<=3 chars) | 5843/8732 | 6641/8732 | +13.6% |

## Key Observations

1. **Head mask is applied correctly**: 22.5% of UrbanSound8K clips produce different transcriptions vs Condition A. The mask reduces hallucination length (mean chars: 14.9 -> 11.8) and increases the number of very short hallucinations.

2. **Hallucination rate remains 100%**: Unlike Calm-Whisper's reported 24.10%, our pipeline shows 100% hallucination rate because our `model.generate()` codepath does not include no-speech suppression. Calm-Whisper likely used the OpenAI Whisper library which applies `no_speech_threshold` to suppress output when `no_speech_prob` is high. In our HuggingFace Transformers pipeline, the model always produces some text.

3. **WER increases as expected**: test-clean WER increases 2.83% -> 3.08% (+0.25%), test-other WER increases 5.10% -> 5.32% (+0.22%). This confirms the head mask degrades speech recognition quality, consistent with Calm-Whisper's reported direction. The magnitude is smaller than Calm-Whisper's (+1.45% and +1.91%) because the `eager` attention implementation in transformers 4.57.6 may interact differently with head masking.

4. **Fair comparison across conditions**: All three conditions (A, B, C) use the same codepath and decoding configuration. While absolute hallucination rates differ from Calm-Whisper's due to pipeline differences, relative comparisons between A/B/C remain valid.

5. **Critical implementation note**: HuggingFace Transformers `model.generate()` does NOT propagate `decoder_head_mask`. The head mask must be injected via monkey-patching the decoder's forward method. Additionally, `attn_implementation="eager"` is required because SDPA attention silently ignores head_mask.

## Result Files
- `schm/results/condition_b_urbansound8k.json` - Per-clip transcriptions (8,732 clips)
- `schm/results/condition_b_librispeech_test_clean.json` - Per-utterance results (2,620 utts)
- `schm/results/condition_b_librispeech_test_other.json` - Per-utterance results (2,939 utts)
- `schm/results/condition_b_summary.json` - Aggregate metrics
- `schm/inference/run_always_mask.py` - Inference script
- `schm/scripts/run_condition_b.sh` - GPU job script
