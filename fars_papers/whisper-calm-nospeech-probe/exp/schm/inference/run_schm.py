"""Condition C: Silence-Conditional Head Masking (SCHM) inference.
For each audio clip: (1) check p_no_speech, (2) if p_nospeech >= tau, decode with
heads {1,6,11} masked; otherwise decode normally (reuse Condition A results).
Reuses pre-computed p_nospeech for UrbanSound8K; computes p_nospeech fresh for LibriSpeech.
Evaluates on UrbanSound8K (hallucination rate) and LibriSpeech test-clean/test-other (WER).
"""

import argparse
import json
import os
import sys
import time
import traceback
from functools import wraps
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
from schm.evaluation.results_io import save_results_json, load_results_json, save_summary

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_ID = "openai/whisper-large-v3"
NUM_DECODER_LAYERS = 32
NUM_ATTENTION_HEADS = 20
MASKED_HEADS = [1, 6, 11]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_TAU = 0.6


def load_model(device="cuda"):
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device)
    model.eval()
    return model, processor


def build_decoder_head_mask(device="cuda", dtype=torch.float16):
    mask = torch.ones(NUM_DECODER_LAYERS, NUM_ATTENTION_HEADS, device=device, dtype=dtype)
    for h in MASKED_HEADS:
        mask[:, h] = 0.0
    return mask


def apply_decoder_head_mask(model, head_mask):
    original_forward = model.model.decoder.forward

    @wraps(original_forward)
    def patched_forward(*args, **kwargs):
        kwargs["head_mask"] = head_mask
        return original_forward(*args, **kwargs)

    model.model.decoder.forward = patched_forward
    return original_forward


def remove_decoder_head_mask(model, original_forward):
    model.model.decoder.forward = original_forward


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


def compute_p_nospeech_batch(model, processor, audio_arrays, nospeech_token_id, device="cuda"):
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)
    batch_size = input_features.shape[0]
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]] * batch_size, device=device
    )

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        p_values = probs[:, nospeech_token_id].cpu().tolist()

    return p_values


def verify_nospeech_token(model, processor):
    gen_no_ts = model.generation_config.no_timestamps_token_id
    nospeech_id = gen_no_ts - 1
    tokenizer_id = processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")
    assert nospeech_id == tokenizer_id, (
        f"Mismatch: generation_config-derived {nospeech_id} != tokenizer {tokenizer_id}"
    )
    print(f"[Token Verification] Confirmed no-speech token ID = {nospeech_id}", flush=True)
    return nospeech_id


def run_urbansound8k_schm(model, processor, head_mask, tau, batch_size=16,
                          max_clips=None, device="cuda"):
    """Run SCHM on UrbanSound8K using pre-computed p_nospeech and Condition A results."""
    p_nospeech_data = load_results_json(RESULTS_DIR / "p_nospeech_urbansound8k.json")
    cond_a_data = load_results_json(RESULTS_DIR / "condition_a_urbansound8k.json")

    p_lookup = {r["clip_id"]: r["p_no_speech"] for r in p_nospeech_data}
    a_lookup = {r["clip_id"]: r["transcription"] for r in cond_a_data}

    all_clip_ids = [r["clip_id"] for r in p_nospeech_data]
    if max_clips is not None:
        all_clip_ids = all_clip_ids[:max_clips]

    clips_needing_mask = set()
    for cid in all_clip_ids:
        if p_lookup[cid] >= tau:
            clips_needing_mask.add(cid)

    print(f"[UrbanSound8K SCHM] tau={tau}, total={len(all_clip_ids)}, "
          f"need_mask={len(clips_needing_mask)} ({len(clips_needing_mask)/len(all_clip_ids)*100:.1f}%)",
          flush=True)

    masked_transcriptions = {}
    if clips_needing_mask:
        original_forward = apply_decoder_head_mask(model, head_mask)
        print("[UrbanSound8K SCHM] Head mask applied, running masked decoding...", flush=True)

        batch_audio = []
        batch_ids = []
        t0 = time.time()

        print("[UrbanSound8K SCHM] Loading dataset to get audio for masked clips...", flush=True)
        data_iter = load_urbansound8k(streaming=True)

        clip_count = 0
        for clip_id, audio_16k, meta in data_iter:
            if max_clips is not None and clip_count >= max_clips:
                break
            clip_count += 1

            if clip_id not in clips_needing_mask:
                continue

            batch_ids.append(clip_id)
            batch_audio.append(audio_16k)

            if len(batch_audio) == batch_size:
                texts = transcribe_batch(model, processor, batch_audio, device=device)
                for cid, t in zip(batch_ids, texts):
                    masked_transcriptions[cid] = t
                batch_audio = []
                batch_ids = []

                if len(masked_transcriptions) % 100 < batch_size:
                    elapsed = time.time() - t0
                    print(f"[UrbanSound8K SCHM] {len(masked_transcriptions)}/{len(clips_needing_mask)} "
                          f"masked clips decoded ({elapsed:.1f}s)", flush=True)
                    wandb.log({"urbansound8k_schm/masked_clips_decoded": len(masked_transcriptions)})

        if batch_audio:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            for cid, t in zip(batch_ids, texts):
                masked_transcriptions[cid] = t

        elapsed = time.time() - t0
        print(f"[UrbanSound8K SCHM] Masked decoding done: {len(masked_transcriptions)} clips in {elapsed:.1f}s",
              flush=True)

        remove_decoder_head_mask(model, original_forward)
        print("[UrbanSound8K SCHM] Head mask removed.", flush=True)

    per_clip = []
    for cid in all_clip_ids:
        p_val = p_lookup[cid]
        mask_applied = p_val >= tau
        if mask_applied:
            transcription = masked_transcriptions.get(cid, "")
        else:
            transcription = a_lookup.get(cid, "")
        per_clip.append({
            "clip_id": cid,
            "transcription": transcription,
            "p_no_speech": p_val,
            "mask_applied": mask_applied,
        })

    clip_ids = [r["clip_id"] for r in per_clip]
    transcriptions = [r["transcription"] for r in per_clip]
    hall_result = compute_hallucination_rate(clip_ids, transcriptions)

    frac_masked = sum(1 for r in per_clip if r["mask_applied"]) / len(per_clip)

    save_results_json(per_clip, RESULTS_DIR / "condition_c_urbansound8k.json")
    print(f"[UrbanSound8K SCHM] Hallucination rate: {hall_result['hallucination_rate']:.4f} "
          f"({hall_result['num_hallucinated']}/{hall_result['total_clips']})", flush=True)
    print(f"[UrbanSound8K SCHM] Fraction masked: {frac_masked:.4f}", flush=True)

    wandb.log({
        "urbansound8k_schm/hallucination_rate": hall_result["hallucination_rate"],
        "urbansound8k_schm/frac_masked": frac_masked,
        "urbansound8k_schm/num_masked": sum(1 for r in per_clip if r["mask_applied"]),
    })

    return hall_result, frac_masked


def run_librispeech_p_nospeech(model, processor, nospeech_token_id, split,
                               batch_size=16, max_utts=None, device="cuda"):
    """Compute p_nospeech for all LibriSpeech utterances in a split."""
    split_label = split.replace(".", "_")
    results = []
    batch_audio = []
    batch_ids = []
    t0 = time.time()

    print(f"[LibriSpeech {split} p_nospeech] Loading dataset...", flush=True)
    data_iter = load_librispeech(split=split, streaming=True)

    for i, (utt_id, audio_16k, ref) in enumerate(data_iter):
        if max_utts is not None and i >= max_utts:
            break
        batch_ids.append(utt_id)
        batch_audio.append(audio_16k)

        if len(batch_audio) == batch_size:
            p_values = compute_p_nospeech_batch(
                model, processor, batch_audio, nospeech_token_id, device=device
            )
            for uid, pv in zip(batch_ids, p_values):
                results.append({"utt_id": uid, "p_no_speech": pv})
            batch_audio = []
            batch_ids = []

            if len(results) % 100 < batch_size:
                elapsed = time.time() - t0
                print(f"[LibriSpeech {split} p_nospeech] {len(results)} utts ({elapsed:.1f}s)",
                      flush=True)
                wandb.log({f"librispeech_{split_label}/p_nospeech_utts": len(results)})

    if batch_audio:
        p_values = compute_p_nospeech_batch(
            model, processor, batch_audio, nospeech_token_id, device=device
        )
        for uid, pv in zip(batch_ids, p_values):
            results.append({"utt_id": uid, "p_no_speech": pv})

    elapsed = time.time() - t0
    p_arr = np.array([r["p_no_speech"] for r in results])
    print(f"[LibriSpeech {split} p_nospeech] Done: {len(results)} utts in {elapsed:.1f}s", flush=True)
    print(f"[LibriSpeech {split} p_nospeech] mean={p_arr.mean():.4f}, "
          f"median={np.median(p_arr):.4f}, std={p_arr.std():.4f}", flush=True)
    print(f"[LibriSpeech {split} p_nospeech] frac >= 0.6: {np.mean(p_arr >= 0.6):.4f}", flush=True)

    out_path = RESULTS_DIR / f"p_nospeech_librispeech_{split_label}.json"
    save_results_json(results, out_path)
    print(f"Saved to {out_path}", flush=True)

    wandb.log({
        f"librispeech_{split_label}/p_nospeech_mean": float(p_arr.mean()),
        f"librispeech_{split_label}/p_nospeech_frac_ge_0.6": float(np.mean(p_arr >= 0.6)),
    })

    return results


def run_librispeech_schm(model, processor, head_mask, p_nospeech_data, split, tau,
                         batch_size=16, max_utts=None, device="cuda"):
    """Run SCHM on LibriSpeech: masked decoding for utts with p_nospeech >= tau, else reuse Cond A."""
    split_label = split.replace(".", "_")

    cond_a_data = load_results_json(RESULTS_DIR / f"condition_a_librispeech_{split_label}.json")
    a_lookup = {r["utt_id"]: r for r in cond_a_data}

    p_lookup = {r["utt_id"]: r["p_no_speech"] for r in p_nospeech_data}

    all_utt_ids = [r["utt_id"] for r in p_nospeech_data]
    if max_utts is not None:
        all_utt_ids = all_utt_ids[:max_utts]

    utts_needing_mask = set()
    for uid in all_utt_ids:
        if p_lookup[uid] >= tau:
            utts_needing_mask.add(uid)

    print(f"[LibriSpeech {split} SCHM] tau={tau}, total={len(all_utt_ids)}, "
          f"need_mask={len(utts_needing_mask)} ({len(utts_needing_mask)/max(len(all_utt_ids),1)*100:.1f}%)",
          flush=True)

    masked_transcriptions = {}
    if utts_needing_mask:
        original_forward = apply_decoder_head_mask(model, head_mask)
        print(f"[LibriSpeech {split} SCHM] Head mask applied, running masked decoding...", flush=True)

        batch_audio = []
        batch_ids = []
        t0 = time.time()

        data_iter = load_librispeech(split=split, streaming=True)
        utt_count = 0
        for utt_id, audio_16k, ref in data_iter:
            if max_utts is not None and utt_count >= max_utts:
                break
            utt_count += 1

            if utt_id not in utts_needing_mask:
                continue

            batch_ids.append(utt_id)
            batch_audio.append(audio_16k)

            if len(batch_audio) == batch_size:
                texts = transcribe_batch(model, processor, batch_audio, device=device)
                for uid, t in zip(batch_ids, texts):
                    masked_transcriptions[uid] = t
                batch_audio = []
                batch_ids = []

                if len(masked_transcriptions) % 50 < batch_size:
                    elapsed = time.time() - t0
                    print(f"[LibriSpeech {split} SCHM] {len(masked_transcriptions)}/{len(utts_needing_mask)} "
                          f"masked utts decoded ({elapsed:.1f}s)", flush=True)

        if batch_audio:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            for uid, t in zip(batch_ids, texts):
                masked_transcriptions[uid] = t

        elapsed = time.time() - t0
        print(f"[LibriSpeech {split} SCHM] Masked decoding done: {len(masked_transcriptions)} utts in {elapsed:.1f}s",
              flush=True)

        remove_decoder_head_mask(model, original_forward)
        print(f"[LibriSpeech {split} SCHM] Head mask removed.", flush=True)

    per_utt = []
    references = []
    hypotheses = []
    for uid in all_utt_ids:
        p_val = p_lookup[uid]
        mask_applied = p_val >= tau
        a_entry = a_lookup.get(uid, {})
        ref = a_entry.get("reference", "")

        if mask_applied:
            hyp = masked_transcriptions.get(uid, "")
        else:
            hyp = a_entry.get("hypothesis", "")

        per_utt.append({
            "utt_id": uid,
            "reference": ref,
            "hypothesis": hyp,
            "p_no_speech": p_val,
            "mask_applied": mask_applied,
        })
        references.append(ref)
        hypotheses.append(hyp)

    wer_result = compute_wer(references, hypotheses)
    frac_masked = sum(1 for r in per_utt if r["mask_applied"]) / max(len(per_utt), 1)

    save_results_json(per_utt, RESULTS_DIR / f"condition_c_librispeech_{split_label}.json")
    print(f"[LibriSpeech {split} SCHM] WER: {wer_result['wer_percent']:.2f}%", flush=True)
    print(f"[LibriSpeech {split} SCHM] Fraction masked (false positive): {frac_masked:.4f}", flush=True)

    wandb.log({
        f"librispeech_{split_label}_schm/wer_percent": wer_result["wer_percent"],
        f"librispeech_{split_label}_schm/frac_masked": frac_masked,
    })

    return wer_result, frac_masked


def main():
    parser = argparse.ArgumentParser(description="Condition C: SCHM inference")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run on 10 samples per dataset for pipeline verification")
    args = parser.parse_args()

    max_clips = 10 if args.sanity_check else None
    max_utts = 10 if args.sanity_check else None
    run_name = "condition_c_sanity_check" if args.sanity_check else "condition_c_schm"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=run_name,
        tags=["condition_c", "schm"],
        config={"tau": args.tau, "batch_size": args.batch_size, "masked_heads": MASKED_HEADS},
    )

    print(f"Loading model: {MODEL_ID} (eager attention for head mask support)", flush=True)
    model, processor = load_model()
    print("Model loaded.", flush=True)

    nospeech_token_id = verify_nospeech_token(model, processor)
    head_mask = build_decoder_head_mask(device=model.device)
    print(f"Head mask built: shape {head_mask.shape}, masking heads {MASKED_HEADS}", flush=True)

    try:
        print(f"\n{'='*60}", flush=True)
        print(f"=== UrbanSound8K SCHM (tau={args.tau}) ===", flush=True)
        print(f"{'='*60}", flush=True)
        us8k_hall, us8k_frac_masked = run_urbansound8k_schm(
            model, processor, head_mask, tau=args.tau,
            batch_size=args.batch_size, max_clips=max_clips,
        )

        print(f"\n{'='*60}", flush=True)
        print(f"=== LibriSpeech p_nospeech computation ===", flush=True)
        print(f"{'='*60}", flush=True)
        p_clean = run_librispeech_p_nospeech(
            model, processor, nospeech_token_id, split="test.clean",
            batch_size=args.batch_size, max_utts=max_utts,
        )
        p_other = run_librispeech_p_nospeech(
            model, processor, nospeech_token_id, split="test.other",
            batch_size=args.batch_size, max_utts=max_utts,
        )

        print(f"\n{'='*60}", flush=True)
        print(f"=== LibriSpeech test.clean SCHM (tau={args.tau}) ===", flush=True)
        print(f"{'='*60}", flush=True)
        clean_wer, clean_frac_masked = run_librispeech_schm(
            model, processor, head_mask, p_clean, split="test.clean",
            tau=args.tau, batch_size=args.batch_size, max_utts=max_utts,
        )

        print(f"\n{'='*60}", flush=True)
        print(f"=== LibriSpeech test.other SCHM (tau={args.tau}) ===", flush=True)
        print(f"{'='*60}", flush=True)
        other_wer, other_frac_masked = run_librispeech_schm(
            model, processor, head_mask, p_other, split="test.other",
            tau=args.tau, batch_size=args.batch_size, max_utts=max_utts,
        )

        summary = {
            "condition": f"C (SCHM, tau={args.tau})",
            "model": MODEL_ID,
            "decoding": "greedy (num_beams=1, do_sample=False, language=en, task=transcribe)",
            "masked_heads": MASKED_HEADS,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "sanity_check": args.sanity_check,
            "urbansound8k": {
                "hallucination_rate": us8k_hall["hallucination_rate"],
                "num_hallucinated": us8k_hall["num_hallucinated"],
                "total_clips": us8k_hall["total_clips"],
                "frac_masked": us8k_frac_masked,
                "num_masked": int(us8k_frac_masked * us8k_hall["total_clips"]),
            },
            "librispeech_test_clean": {
                "wer_percent": clean_wer["wer_percent"],
                "num_utterances": clean_wer["num_utterances"],
                "frac_masked": clean_frac_masked,
            },
            "librispeech_test_other": {
                "wer_percent": other_wer["wer_percent"],
                "num_utterances": other_wer["num_utterances"],
                "frac_masked": other_frac_masked,
            },
        }

        save_summary(summary, RESULTS_DIR / "condition_c_summary.json")
        print(f"\n{'='*60}", flush=True)
        print("=== CONDITION C SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)

        wandb.log({
            "final/hallucination_rate": us8k_hall["hallucination_rate"],
            "final/wer_clean": clean_wer["wer_percent"],
            "final/wer_other": other_wer["wer_percent"],
            "final/us8k_frac_masked": us8k_frac_masked,
            "final/ls_clean_frac_masked": clean_frac_masked,
            "final/ls_other_frac_masked": other_frac_masked,
        })

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
