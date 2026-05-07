"""Compute corpus-level Word Error Rate (WER) using jiwer.
Text normalization: lowercase, strip punctuation to match Whisper output format.
"""

import re
import jiwer


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for WER computation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    normalize: bool = True,
) -> dict:
    """Compute corpus-level WER.

    Args:
        references: Ground-truth transcription strings.
        hypotheses: Model-generated transcription strings.
        normalize: If True, apply lowercase + strip punctuation.

    Returns:
        Dict with "wer" (as percentage), "num_utterances", and raw jiwer output.
    """
    assert len(references) == len(hypotheses)

    if normalize:
        refs = [normalize_text(r) for r in references]
        hyps = [normalize_text(h) for h in hypotheses]
    else:
        refs = references
        hyps = hypotheses

    wer_value = jiwer.wer(refs, hyps)

    return {
        "wer_percent": round(wer_value * 100, 2),
        "wer_raw": wer_value,
        "num_utterances": len(references),
    }
