"""Compute hallucination rate for non-speech audio clips (UrbanSound8K).
Hallucination = model produced a non-empty transcription for a clip that
contains no speech. Rate = (# hallucinated clips) / (total clips).
"""

from typing import Optional


def compute_hallucination_rate(
    clip_ids: list[str],
    transcriptions: list[str],
    p_no_speech_values: Optional[list[float]] = None,
) -> dict:
    """Compute hallucination rate and per-clip binary flags.

    Args:
        clip_ids: Identifiers for each clip.
        transcriptions: Raw transcription strings from the model.
        p_no_speech_values: Optional p_no_speech probabilities per clip.

    Returns:
        Dict with keys:
            - "hallucination_rate": float (0-1)
            - "num_hallucinated": int
            - "total_clips": int
            - "per_clip": list of dicts with clip_id, transcription,
              is_hallucinated, and optionally p_no_speech.
    """
    assert len(clip_ids) == len(transcriptions)
    if p_no_speech_values is not None:
        assert len(clip_ids) == len(p_no_speech_values)

    per_clip = []
    num_hallucinated = 0

    for i, (cid, text) in enumerate(zip(clip_ids, transcriptions)):
        is_hall = len(text.strip()) > 0
        if is_hall:
            num_hallucinated += 1

        entry = {
            "clip_id": cid,
            "transcription": text,
            "is_hallucinated": is_hall,
        }
        if p_no_speech_values is not None:
            entry["p_no_speech"] = p_no_speech_values[i]
        per_clip.append(entry)

    total = len(clip_ids)
    rate = num_hallucinated / total if total > 0 else 0.0

    return {
        "hallucination_rate": rate,
        "num_hallucinated": num_hallucinated,
        "total_clips": total,
        "per_clip": per_clip,
    }
