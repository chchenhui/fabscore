"""Compute p_no_speech for audio clips using Whisper-large-v3.
Extracts the probability of the no-speech token at the first decoder position
via a single encoder+decoder forward pass (no full generation).
Used for the Phase-1 go/no-go diagnostic and as the SCHM trigger signal.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import WhisperForConditionalGeneration, WhisperProcessor


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schm.data.load_urbansound8k import load_urbansound8k
from schm.evaluation.results_io import save_results_json

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


def verify_nospeech_token(model, processor):
    gen_no_ts = model.generation_config.no_timestamps_token_id
    nospeech_id = gen_no_ts - 1

    tokenizer_id = processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")

    print(f"[Token Verification] no_timestamps_token_id (generation_config): {gen_no_ts}", flush=True)
    print(f"[Token Verification] no_speech token id (gen_config - 1):        {nospeech_id}", flush=True)
    print(f"[Token Verification] <|nospeech|> from tokenizer:                {tokenizer_id}", flush=True)

    assert nospeech_id == tokenizer_id, (
        f"Mismatch: generation_config-derived {nospeech_id} != tokenizer {tokenizer_id}"
    )
    print(f"[Token Verification] Confirmed no-speech token ID = {nospeech_id}", flush=True)
    return nospeech_id


def compute_p_nospeech(model, processor, audio_array, nospeech_token_id, device="cuda"):
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)
        decoder_input_ids = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        p_nospeech = probs[0, nospeech_token_id].item()

    return p_nospeech


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


def main():
    parser = argparse.ArgumentParser(description="Compute p_no_speech for UrbanSound8K clips")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run on 10 samples for pipeline verification")
    args = parser.parse_args()

    max_clips = 10 if args.sanity_check else None
    run_name = "phase1_p_nospeech_sanity" if args.sanity_check else "phase1_p_nospeech"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=run_name,
        tags=["phase1", "diagnostic"],
    )

    print(f"Loading model: {MODEL_ID}", flush=True)
    model, processor = load_model()
    print("Model loaded.", flush=True)

    nospeech_token_id = verify_nospeech_token(model, processor)

    results = []
    batch_audio = []
    batch_ids = []
    p_values_all = []
    t0 = time.time()

    try:
        print("[UrbanSound8K] Loading dataset...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        data_iter = load_urbansound8k(streaming=True)
        print("[UrbanSound8K] Dataset iterator created.", flush=True)

        if max_clips is not None:
            import itertools
            data_iter = itertools.islice(data_iter, max_clips)

        all_clips = []
        for item in data_iter:
            all_clips.append(item)
            if len(all_clips) == 1:
                print(f"[UrbanSound8K] First clip loaded: {item[0]}", flush=True)

        print(f"[UrbanSound8K] Collected {len(all_clips)} clips.", flush=True)
        print("[UrbanSound8K] Starting p_no_speech computation...", flush=True)

        for i, (clip_id, audio_16k, meta) in enumerate(all_clips):
            if i < 3:
                print(f"  clip {i}: {clip_id}, len={len(audio_16k)}, "
                      f"min={audio_16k.min():.4f}, max={audio_16k.max():.4f}", flush=True)
            batch_ids.append(clip_id)
            batch_audio.append(audio_16k)

            if len(batch_audio) == args.batch_size:
                p_values = compute_p_nospeech_batch(
                    model, processor, batch_audio, nospeech_token_id
                )
                for cid, pv in zip(batch_ids, p_values):
                    results.append({"clip_id": cid, "p_no_speech": pv})
                p_values_all.extend(p_values)
                batch_audio = []
                batch_ids = []

                processed = len(results)
                if processed % 100 < args.batch_size:
                    elapsed = time.time() - t0
                    running_mean = np.mean(p_values_all)
                    print(
                        f"[UrbanSound8K] {processed} clips | "
                        f"mean p_no_speech={running_mean:.4f} | "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                    wandb.log({
                        "clips_processed": processed,
                        "running_mean_p_nospeech": running_mean,
                    })

        if batch_audio:
            p_values = compute_p_nospeech_batch(
                model, processor, batch_audio, nospeech_token_id
            )
            for cid, pv in zip(batch_ids, p_values):
                results.append({"clip_id": cid, "p_no_speech": pv})
            p_values_all.extend(p_values)

        elapsed = time.time() - t0
        print(f"\n[UrbanSound8K] Done: {len(results)} clips in {elapsed:.1f}s", flush=True)

        p_arr = np.array(p_values_all)
        stats = {
            "total_clips": len(results),
            "mean_p_nospeech": float(np.mean(p_arr)),
            "median_p_nospeech": float(np.median(p_arr)),
            "std_p_nospeech": float(np.std(p_arr)),
            "min_p_nospeech": float(np.min(p_arr)),
            "max_p_nospeech": float(np.max(p_arr)),
            "frac_gt_0.5": float(np.mean(p_arr > 0.5)),
            "frac_gt_0.6": float(np.mean(p_arr > 0.6)),
            "frac_gt_0.7": float(np.mean(p_arr > 0.7)),
        }
        print(f"[Stats] {json.dumps(stats, indent=2)}", flush=True)

        wandb.log({
            "total_clips": stats["total_clips"],
            "mean_p_nospeech": stats["mean_p_nospeech"],
            "median_p_nospeech": stats["median_p_nospeech"],
            "p_nospeech_gt_0.5_frac": stats["frac_gt_0.5"],
            "p_nospeech_gt_0.6_frac": stats["frac_gt_0.6"],
            "p_nospeech_gt_0.7_frac": stats["frac_gt_0.7"],
        })

        out_path = RESULTS_DIR / "p_nospeech_urbansound8k.json"
        save_results_json(results, out_path)
        print(f"Saved to {out_path}", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
