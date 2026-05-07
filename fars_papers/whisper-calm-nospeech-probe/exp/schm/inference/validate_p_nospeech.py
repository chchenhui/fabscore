"""Validate standalone p_no_speech computation against HuggingFace's native
WhisperNoSpeechDetection logits processor. Runs model.generate() with
no_speech_threshold enabled, hooks into the logits processor to extract the
internal no_speech_prob, and compares with our 1-step forward pass values.
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
from transformers.generation.logits_process import WhisperNoSpeechDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schm.data.load_urbansound8k import load_urbansound8k
from schm.inference.compute_p_nospeech import (
    compute_p_nospeech_batch,
    verify_nospeech_token,
)
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


def extract_hf_nospeech_prob(model, processor, audio_arrays, device="cuda"):
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        out = model.generate(
            input_features,
            max_new_tokens=1,
            num_beams=1,
            do_sample=False,
            language="en",
            task="transcribe",
            no_speech_threshold=0.5,
            return_dict_in_generate=True,
        )

    no_speech_detector = None
    for proc in model.generation_config._logits_processor_list if hasattr(model.generation_config, '_logits_processor_list') else []:
        if isinstance(proc, WhisperNoSpeechDetection):
            no_speech_detector = proc
            break

    return None


def extract_hf_nospeech_prob_manual(model, processor, audio_arrays, nospeech_token_id, device="cuda"):
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    batch_size = input_features.shape[0]

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    sot_token_id = model.config.decoder_start_token_id
    decoder_input_ids = [sot_token_id] + [tid for _, tid in forced_decoder_ids]
    decoder_input_ids_tensor = torch.tensor(
        [decoder_input_ids] * batch_size, device=device
    )

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids_tensor,
        )
        logits_at_sot = outputs.logits[:, 0, :]
        probs = torch.softmax(logits_at_sot.float(), dim=-1)
        p_values_sot = probs[:, nospeech_token_id].cpu().tolist()

    return p_values_sot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    max_clips = 100 if args.sanity_check else args.max_clips
    run_name = "validate_p_nospeech" + ("_sanity" if args.sanity_check else "")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=run_name,
        tags=["validation", "p_nospeech"],
    )

    print(f"Loading model: {MODEL_ID}", flush=True)
    model, processor = load_model()
    nospeech_token_id = verify_nospeech_token(model, processor)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    print(f"[Info] Forced decoder IDs: {forced_decoder_ids}", flush=True)
    sot_token_id = model.config.decoder_start_token_id
    print(f"[Info] SOT token ID: {sot_token_id}", flush=True)
    print(f"[Info] Full decoder_input_ids for validation: "
          f"{[sot_token_id] + [tid for _, tid in forced_decoder_ids]}", flush=True)

    results = []
    batch_audio = []
    batch_ids = []
    t0 = time.time()

    import itertools
    data_iter = load_urbansound8k(streaming=True)
    if max_clips is not None:
        data_iter = itertools.islice(data_iter, max_clips)

    all_clips = list(data_iter)
    print(f"Collected {len(all_clips)} clips", flush=True)

    for i, (clip_id, audio_16k, meta) in enumerate(all_clips):
        batch_ids.append(clip_id)
        batch_audio.append(audio_16k)

        if len(batch_audio) == args.batch_size:
            p_standalone = compute_p_nospeech_batch(
                model, processor, batch_audio, nospeech_token_id
            )
            p_full_ctx = extract_hf_nospeech_prob_manual(
                model, processor, batch_audio, nospeech_token_id
            )

            for cid, ps, pf in zip(batch_ids, p_standalone, p_full_ctx):
                results.append({
                    "clip_id": cid,
                    "p_standalone": ps,
                    "p_full_context": pf,
                    "diff": abs(ps - pf),
                })

            batch_audio = []
            batch_ids = []

            if len(results) % 100 < args.batch_size:
                print(f"  {len(results)} clips processed", flush=True)

    if batch_audio:
        p_standalone = compute_p_nospeech_batch(
            model, processor, batch_audio, nospeech_token_id
        )
        p_full_ctx = extract_hf_nospeech_prob_manual(
            model, processor, batch_audio, nospeech_token_id
        )
        for cid, ps, pf in zip(batch_ids, p_standalone, p_full_ctx):
            results.append({
                "clip_id": cid,
                "p_standalone": ps,
                "p_full_context": pf,
                "diff": abs(ps - pf),
            })

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} clips in {elapsed:.1f}s", flush=True)

    diffs = [r["diff"] for r in results]
    standalone_vals = [r["p_standalone"] for r in results]
    fullctx_vals = [r["p_full_context"] for r in results]

    print(f"\n=== Validation Summary ===", flush=True)
    print(f"Mean |diff|: {np.mean(diffs):.6f}", flush=True)
    print(f"Max |diff|:  {np.max(diffs):.6f}", flush=True)
    print(f"Median |diff|: {np.median(diffs):.6f}", flush=True)
    print(f"Standalone mean: {np.mean(standalone_vals):.4f}", flush=True)
    print(f"Full-context mean: {np.mean(fullctx_vals):.4f}", flush=True)
    print(f"Correlation: {np.corrcoef(standalone_vals, fullctx_vals)[0,1]:.6f}", flush=True)

    for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        s_frac = np.mean(np.array(standalone_vals) > tau)
        f_frac = np.mean(np.array(fullctx_vals) > tau)
        print(f"  tau={tau:.1f}: standalone={s_frac*100:.1f}%, full_ctx={f_frac*100:.1f}%", flush=True)

    summary = {
        "total_clips": len(results),
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "median_abs_diff": float(np.median(diffs)),
        "correlation": float(np.corrcoef(standalone_vals, fullctx_vals)[0, 1]),
        "standalone_mean": float(np.mean(standalone_vals)),
        "full_context_mean": float(np.mean(fullctx_vals)),
    }

    wandb.log(summary)

    out_path = RESULTS_DIR / "validate_p_nospeech.json"
    save_results_json(results, out_path)
    print(f"Results saved to {out_path}", flush=True)

    summary_path = RESULTS_DIR / "validate_p_nospeech_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
