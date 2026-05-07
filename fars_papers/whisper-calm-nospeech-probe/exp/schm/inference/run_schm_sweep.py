"""Optimized SCHM sweep: evaluates multiple tau values and modes (mask, suppress)
in a single run. For UrbanSound8K, reuses pre-computed p_nospeech and Condition A
results. For LibriSpeech, computes p_nospeech once then evaluates all tau/mode combos.

Modes:
  - mask: decode with head mask when p_nospeech >= tau (original SCHM)
  - suppress: output empty string when p_nospeech >= tau (no-speech suppression)
  - mask+suppress: apply head mask AND suppress if still short hallucination
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
    assert nospeech_id == tokenizer_id
    print(f"[Token Verification] Confirmed no-speech token ID = {nospeech_id}", flush=True)
    return nospeech_id


def eval_urbansound8k_sweep(model, processor, head_mask, tau_list, modes,
                            batch_size=16, max_clips=None, device="cuda"):
    p_nospeech_data = load_results_json(RESULTS_DIR / "p_nospeech_urbansound8k.json")
    cond_a_data = load_results_json(RESULTS_DIR / "condition_a_urbansound8k.json")

    p_lookup = {r["clip_id"]: r["p_no_speech"] for r in p_nospeech_data}
    a_lookup = {r["clip_id"]: r["transcription"] for r in cond_a_data}

    all_clip_ids = [r["clip_id"] for r in p_nospeech_data]
    if max_clips is not None:
        all_clip_ids = all_clip_ids[:max_clips]

    max_tau = max(tau_list)
    clips_needing_mask = {cid for cid in all_clip_ids if p_lookup[cid] >= max_tau}

    for tau in tau_list:
        clips_at_tau = sum(1 for cid in all_clip_ids if p_lookup[cid] >= tau)
        print(f"[US8K] tau={tau:.1f}: {clips_at_tau}/{len(all_clip_ids)} clips trigger "
              f"({clips_at_tau/len(all_clip_ids)*100:.1f}%)", flush=True)

    any_need_mask = any("mask" in m for m in modes)
    if not any_need_mask:
        clips_needing_mask = set()

    if any_need_mask:
        min_tau_for_mask = min(tau_list)
        clips_needing_mask = {cid for cid in all_clip_ids if p_lookup[cid] >= min_tau_for_mask}

    masked_transcriptions = {}
    if clips_needing_mask and any_need_mask:
        original_forward = apply_decoder_head_mask(model, head_mask)
        print(f"[US8K] Running masked decoding for {len(clips_needing_mask)} clips...", flush=True)

        batch_audio = []
        batch_ids = []
        t0 = time.time()

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
                    print(f"[US8K] {len(masked_transcriptions)}/{len(clips_needing_mask)} "
                          f"masked ({elapsed:.1f}s)", flush=True)

        if batch_audio:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            for cid, t in zip(batch_ids, texts):
                masked_transcriptions[cid] = t

        elapsed = time.time() - t0
        print(f"[US8K] Masked decoding done: {len(masked_transcriptions)} clips in {elapsed:.1f}s", flush=True)
        remove_decoder_head_mask(model, original_forward)

    results = {}
    for mode in modes:
        for tau in tau_list:
            key = f"{mode}_tau{tau}"
            per_clip = []
            for cid in all_clip_ids:
                p_val = p_lookup[cid]
                triggered = p_val >= tau

                if triggered:
                    if mode == "suppress":
                        transcription = ""
                    elif mode == "mask":
                        transcription = masked_transcriptions.get(cid, a_lookup.get(cid, ""))
                    elif mode == "mask+suppress":
                        masked_text = masked_transcriptions.get(cid, "")
                        transcription = "" if len(masked_text.strip()) <= 3 else masked_text
                    else:
                        transcription = a_lookup.get(cid, "")
                else:
                    transcription = a_lookup.get(cid, "")

                per_clip.append({
                    "clip_id": cid,
                    "transcription": transcription,
                    "p_no_speech": p_val,
                    "triggered": triggered,
                })

            clip_ids = [r["clip_id"] for r in per_clip]
            transcriptions = [r["transcription"] for r in per_clip]
            hall_result = compute_hallucination_rate(clip_ids, transcriptions)
            n_triggered = sum(1 for r in per_clip if r["triggered"])
            frac_triggered = n_triggered / len(per_clip)

            results[key] = {
                "mode": mode,
                "tau": tau,
                "hallucination_rate": hall_result["hallucination_rate"],
                "num_hallucinated": hall_result["num_hallucinated"],
                "total_clips": hall_result["total_clips"],
                "frac_triggered": frac_triggered,
                "num_triggered": n_triggered,
            }

            print(f"[US8K] {key}: halluc={hall_result['hallucination_rate']*100:.1f}%, "
                  f"triggered={frac_triggered*100:.1f}%", flush=True)
            wandb.log({f"us8k/{key}/hallucination_rate": hall_result["hallucination_rate"],
                       f"us8k/{key}/frac_triggered": frac_triggered})

    return results


def eval_librispeech_sweep(model, processor, head_mask, nospeech_token_id,
                           tau_list, modes, split, batch_size=16,
                           max_utts=None, device="cuda"):
    split_label = split.replace(".", "_")

    p_nospeech_path = RESULTS_DIR / f"p_nospeech_librispeech_{split_label}.json"
    if p_nospeech_path.exists():
        print(f"[LS {split}] Loading pre-computed p_nospeech from {p_nospeech_path}", flush=True)
        p_nospeech_data = load_results_json(p_nospeech_path)
    else:
        print(f"[LS {split}] Computing p_nospeech...", flush=True)
        p_nospeech_data = []
        batch_audio = []
        batch_ids = []
        t0 = time.time()
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
                    p_nospeech_data.append({"utt_id": uid, "p_no_speech": pv})
                batch_audio = []
                batch_ids = []
                if len(p_nospeech_data) % 100 < batch_size:
                    print(f"[LS {split}] {len(p_nospeech_data)} utts", flush=True)
        if batch_audio:
            p_values = compute_p_nospeech_batch(
                model, processor, batch_audio, nospeech_token_id, device=device
            )
            for uid, pv in zip(batch_ids, p_values):
                p_nospeech_data.append({"utt_id": uid, "p_no_speech": pv})
        save_results_json(p_nospeech_data, p_nospeech_path)
        print(f"[LS {split}] p_nospeech computed for {len(p_nospeech_data)} utts in {time.time()-t0:.1f}s",
              flush=True)

    cond_a_data = load_results_json(RESULTS_DIR / f"condition_a_librispeech_{split_label}.json")
    a_lookup = {r["utt_id"]: r for r in cond_a_data}
    p_lookup = {r["utt_id"]: r["p_no_speech"] for r in p_nospeech_data}

    all_utt_ids = [r["utt_id"] for r in p_nospeech_data]
    if max_utts is not None:
        all_utt_ids = all_utt_ids[:max_utts]

    any_need_mask = any("mask" in m for m in modes)
    min_tau = min(tau_list)
    utts_needing_mask = {uid for uid in all_utt_ids if p_lookup[uid] >= min_tau} if any_need_mask else set()

    for tau in tau_list:
        n = sum(1 for uid in all_utt_ids if p_lookup[uid] >= tau)
        print(f"[LS {split}] tau={tau:.1f}: {n}/{len(all_utt_ids)} utts trigger ({n/len(all_utt_ids)*100:.2f}%)",
              flush=True)

    masked_transcriptions = {}
    if utts_needing_mask:
        original_forward = apply_decoder_head_mask(model, head_mask)
        print(f"[LS {split}] Running masked decoding for {len(utts_needing_mask)} utts...", flush=True)
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
        if batch_audio:
            texts = transcribe_batch(model, processor, batch_audio, device=device)
            for uid, t in zip(batch_ids, texts):
                masked_transcriptions[uid] = t
        elapsed = time.time() - t0
        print(f"[LS {split}] Masked decoding done: {len(masked_transcriptions)} utts in {elapsed:.1f}s", flush=True)
        remove_decoder_head_mask(model, original_forward)

    results = {}
    for mode in modes:
        for tau in tau_list:
            key = f"{mode}_tau{tau}"
            references = []
            hypotheses = []
            n_triggered = 0

            for uid in all_utt_ids:
                p_val = p_lookup[uid]
                triggered = p_val >= tau
                a_entry = a_lookup.get(uid, {})
                ref = a_entry.get("reference", "")

                if triggered:
                    n_triggered += 1
                    if mode == "suppress":
                        hyp = ""
                    elif mode == "mask":
                        hyp = masked_transcriptions.get(uid, a_entry.get("hypothesis", ""))
                    elif mode == "mask+suppress":
                        masked_text = masked_transcriptions.get(uid, "")
                        hyp = "" if len(masked_text.strip()) <= 3 else masked_text
                    else:
                        hyp = a_entry.get("hypothesis", "")
                else:
                    hyp = a_entry.get("hypothesis", "")

                references.append(ref)
                hypotheses.append(hyp)

            wer_result = compute_wer(references, hypotheses)
            frac_triggered = n_triggered / max(len(all_utt_ids), 1)

            results[key] = {
                "mode": mode,
                "tau": tau,
                "wer_percent": wer_result["wer_percent"],
                "num_utterances": wer_result["num_utterances"],
                "frac_triggered": frac_triggered,
                "num_triggered": n_triggered,
            }

            print(f"[LS {split}] {key}: WER={wer_result['wer_percent']:.2f}%, "
                  f"triggered={frac_triggered*100:.2f}%", flush=True)
            wandb.log({f"ls_{split_label}/{key}/wer_percent": wer_result["wer_percent"],
                       f"ls_{split_label}/{key}/frac_triggered": frac_triggered})

    return results


def main():
    parser = argparse.ArgumentParser(description="SCHM sweep: multi-tau, multi-mode evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tau-list", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--modes", type=str, nargs="+", default=["suppress", "mask"],
                        choices=["suppress", "mask", "mask+suppress"])
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    max_clips = 10 if args.sanity_check else None
    max_utts = 10 if args.sanity_check else None
    run_name = "schm_sweep_sanity" if args.sanity_check else "schm_sweep"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=run_name,
        tags=["condition_c", "schm", "sweep"],
        config={
            "tau_list": args.tau_list,
            "modes": args.modes,
            "batch_size": args.batch_size,
            "masked_heads": MASKED_HEADS,
        },
    )

    print(f"Loading model: {MODEL_ID} (eager attention for head mask)", flush=True)
    model, processor = load_model()
    nospeech_token_id = verify_nospeech_token(model, processor)
    head_mask = build_decoder_head_mask(device=model.device)
    print(f"Head mask: shape {head_mask.shape}, masking heads {MASKED_HEADS}", flush=True)
    print(f"Tau values: {args.tau_list}, Modes: {args.modes}", flush=True)

    try:
        print(f"\n{'='*60}", flush=True)
        print("=== UrbanSound8K SCHM Sweep ===", flush=True)
        us8k_results = eval_urbansound8k_sweep(
            model, processor, head_mask,
            tau_list=args.tau_list, modes=args.modes,
            batch_size=args.batch_size, max_clips=max_clips,
        )

        print(f"\n{'='*60}", flush=True)
        print("=== LibriSpeech test.clean SCHM Sweep ===", flush=True)
        clean_results = eval_librispeech_sweep(
            model, processor, head_mask, nospeech_token_id,
            tau_list=args.tau_list, modes=args.modes, split="test.clean",
            batch_size=args.batch_size, max_utts=max_utts,
        )

        print(f"\n{'='*60}", flush=True)
        print("=== LibriSpeech test.other SCHM Sweep ===", flush=True)
        other_results = eval_librispeech_sweep(
            model, processor, head_mask, nospeech_token_id,
            tau_list=args.tau_list, modes=args.modes, split="test.other",
            batch_size=args.batch_size, max_utts=max_utts,
        )

        sweep_summary = {
            "config": {
                "model": MODEL_ID,
                "tau_list": args.tau_list,
                "modes": args.modes,
                "masked_heads": MASKED_HEADS,
            },
            "urbansound8k": us8k_results,
            "librispeech_test_clean": clean_results,
            "librispeech_test_other": other_results,
            "baselines": {
                "condition_a": {
                    "urbansound8k_hallucination_rate": 1.0,
                    "librispeech_clean_wer": 2.83,
                    "librispeech_other_wer": 5.10,
                },
                "condition_b": {
                    "urbansound8k_hallucination_rate": 1.0,
                    "librispeech_clean_wer": 3.08,
                    "librispeech_other_wer": 5.32,
                },
            },
        }

        out_path = RESULTS_DIR / "schm_sweep_results.json"
        with open(out_path, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print(f"\nSweep results saved to {out_path}", flush=True)

        print(f"\n{'='*60}", flush=True)
        print("=== SWEEP SUMMARY TABLE ===", flush=True)
        print(f"{'Config':<25} {'US8K Halluc':>12} {'LS-clean WER':>13} {'LS-other WER':>13}", flush=True)
        print("-" * 65, flush=True)
        print(f"{'A (Default)':<25} {'100.00%':>12} {'2.83%':>13} {'5.10%':>13}", flush=True)
        print(f"{'B (Always-Mask)':<25} {'100.00%':>12} {'3.08%':>13} {'5.32%':>13}", flush=True)

        for mode in args.modes:
            for tau in args.tau_list:
                key = f"{mode}_tau{tau}"
                us = us8k_results.get(key, {})
                cl = clean_results.get(key, {})
                ot = other_results.get(key, {})
                label = f"C-{mode} tau={tau}"
                print(f"{label:<25} {us.get('hallucination_rate', 0)*100:>11.2f}% "
                      f"{cl.get('wer_percent', 0):>12.2f}% {ot.get('wer_percent', 0):>12.2f}%",
                      flush=True)

        wandb.log({"sweep_complete": 1})

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
