"""Condition A: Default Whisper-large-v3 inference (greedy decoding, no modifications).
Evaluates on UrbanSound8K (hallucination rate) and LibriSpeech test-clean/test-other (WER).
Supports batched inference, progress logging, WandB logging, and a sanity-check mode.
Reusable transcribe_batch() function for Conditions B and C.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schm.data.load_urbansound8k import load_urbansound8k
from schm.data.load_librispeech import load_librispeech
from schm.evaluation.hallucination_rate import compute_hallucination_rate
from schm.evaluation.wer_eval import compute_wer
from schm.evaluation.results_io import save_results_json, save_summary

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_ID = "openai/whisper-large-v3"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_model(device="cuda"):
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model, processor


def transcribe_batch(model, processor, audio_arrays, sr=16000, device="cuda"):
    inputs = processor(
        audio_arrays,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            language="en",
            task="transcribe",
        )

    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts


def run_urbansound8k(model, processor, batch_size=16, max_clips=None, device="cuda"):
    clip_ids = []
    transcriptions = []
    t0 = time.time()
    batch_audio = []
    batch_ids = []

    print("[UrbanSound8K] Loading dataset...", flush=True)
    data_iter = load_urbansound8k(streaming=True)
    print("[UrbanSound8K] Dataset iterator created, starting transcription...", flush=True)

    for i, (clip_id, audio_16k, meta) in enumerate(data_iter):
        if max_clips is not None and i >= max_clips:
            break
        batch_ids.append(clip_id)
        batch_audio.append(audio_16k)

        if len(batch_audio) == batch_size:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            clip_ids.extend(batch_ids)
            transcriptions.extend(texts)
            batch_audio = []
            batch_ids = []

            processed = len(clip_ids)
            if processed % 100 < batch_size:
                elapsed = time.time() - t0
                print(f"[UrbanSound8K] {processed} clips processed ({elapsed:.1f}s)", flush=True)
                wandb.log({"urbansound8k/clips_processed": processed})

    if batch_audio:
        texts = transcribe_batch(model, processor, batch_audio, device=device)
        clip_ids.extend(batch_ids)
        transcriptions.extend(texts)

    elapsed = time.time() - t0
    print(f"[UrbanSound8K] Done: {len(clip_ids)} clips in {elapsed:.1f}s", flush=True)

    result = compute_hallucination_rate(clip_ids, transcriptions)
    print(f"[UrbanSound8K] Hallucination rate: {result['hallucination_rate']:.4f} "
          f"({result['num_hallucinated']}/{result['total_clips']})", flush=True)

    per_clip = [{"clip_id": cid, "transcription": t} for cid, t in zip(clip_ids, transcriptions)]
    save_results_json(per_clip, RESULTS_DIR / "condition_a_urbansound8k.json")

    wandb.log({
        "urbansound8k/hallucination_rate": result["hallucination_rate"],
        "urbansound8k/num_hallucinated": result["num_hallucinated"],
        "urbansound8k/total_clips": result["total_clips"],
    })

    return result


def run_librispeech(model, processor, split, batch_size=16, max_utts=None, device="cuda"):
    split_label = split.replace(".", "_")
    utt_ids = []
    references = []
    hypotheses = []
    t0 = time.time()
    batch_audio = []
    batch_ids = []
    batch_refs = []

    print(f"[LibriSpeech {split}] Loading dataset...", flush=True)
    data_iter = load_librispeech(split=split, streaming=True)
    print(f"[LibriSpeech {split}] Dataset iterator created, starting transcription...", flush=True)

    for i, (utt_id, audio_16k, ref) in enumerate(data_iter):
        if max_utts is not None and i >= max_utts:
            break
        batch_ids.append(utt_id)
        batch_audio.append(audio_16k)
        batch_refs.append(ref)

        if len(batch_audio) == batch_size:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            utt_ids.extend(batch_ids)
            references.extend(batch_refs)
            hypotheses.extend(texts)
            batch_audio = []
            batch_ids = []
            batch_refs = []

            processed = len(utt_ids)
            if processed % 100 < batch_size:
                elapsed = time.time() - t0
                print(f"[LibriSpeech {split}] {processed} utts processed ({elapsed:.1f}s)", flush=True)
                wandb.log({f"librispeech_{split_label}/utts_processed": processed})

    if batch_audio:
        texts = transcribe_batch(model, processor, batch_audio, device=device)
        utt_ids.extend(batch_ids)
        references.extend(batch_refs)
        hypotheses.extend(texts)

    elapsed = time.time() - t0
    print(f"[LibriSpeech {split}] Done: {len(utt_ids)} utts in {elapsed:.1f}s", flush=True)

    wer_result = compute_wer(references, hypotheses)
    print(f"[LibriSpeech {split}] WER: {wer_result['wer_percent']:.2f}%", flush=True)

    per_utt = [
        {"utt_id": uid, "reference": ref, "hypothesis": hyp}
        for uid, ref, hyp in zip(utt_ids, references, hypotheses)
    ]
    save_results_json(per_utt, RESULTS_DIR / f"condition_a_librispeech_{split_label}.json")

    wandb.log({
        f"librispeech_{split_label}/wer_percent": wer_result["wer_percent"],
        f"librispeech_{split_label}/num_utterances": wer_result["num_utterances"],
    })

    return wer_result


def main():
    parser = argparse.ArgumentParser(description="Condition A: Default Whisper inference")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run on 10 samples per dataset for pipeline verification")
    args = parser.parse_args()

    max_clips = 10 if args.sanity_check else None
    max_utts = 10 if args.sanity_check else None
    run_name = "condition_a_sanity_check" if args.sanity_check else "condition_a_default_whisper"

    wandb.init(project=None, name=run_name, tags=["condition_a"])

    print(f"Loading model: {MODEL_ID}", flush=True)
    model, processor = load_model()
    print("Model loaded.", flush=True)

    try:
        print("=== UrbanSound8K ===", flush=True)
        us8k_result = run_urbansound8k(model, processor, batch_size=args.batch_size,
                                        max_clips=max_clips)

        print("=== LibriSpeech test.clean ===", flush=True)
        clean_result = run_librispeech(model, processor, split="test.clean",
                                        batch_size=args.batch_size, max_utts=max_utts)

        print("=== LibriSpeech test.other ===", flush=True)
        other_result = run_librispeech(model, processor, split="test.other",
                                        batch_size=args.batch_size, max_utts=max_utts)

        summary = {
            "condition": "A (Default Whisper)",
            "model": MODEL_ID,
            "decoding": "greedy (num_beams=1, do_sample=False, language=en, task=transcribe)",
            "batch_size": args.batch_size,
            "sanity_check": args.sanity_check,
            "urbansound8k": {
                "hallucination_rate": us8k_result["hallucination_rate"],
                "num_hallucinated": us8k_result["num_hallucinated"],
                "total_clips": us8k_result["total_clips"],
            },
            "librispeech_test_clean": {
                "wer_percent": clean_result["wer_percent"],
                "num_utterances": clean_result["num_utterances"],
            },
            "librispeech_test_other": {
                "wer_percent": other_result["wer_percent"],
                "num_utterances": other_result["num_utterances"],
            },
        }

        save_summary(summary, RESULTS_DIR / "condition_a_summary.json")
        print(f"\nSummary saved to {RESULTS_DIR / 'condition_a_summary.json'}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)

        wandb.log({
            "final/hallucination_rate": us8k_result["hallucination_rate"],
            "final/wer_clean": clean_result["wer_percent"],
            "final/wer_other": other_result["wer_percent"],
        })
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
