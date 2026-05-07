"""Skip-Only Ablation (Condition D): Disentangle trigger policy from head masking.
When p_no_speech >= tau, output empty transcription (skip); otherwise use default
Whisper (Condition A) transcription. No model inference needed -- purely post-hoc
re-labeling of Condition A results using pre-computed p_no_speech values.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schm.evaluation.hallucination_rate import compute_hallucination_rate
from schm.evaluation.wer_eval import compute_wer
from schm.evaluation.results_io import load_results_json, save_results_json, save_summary

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_TAU = 0.6


def load_and_index(path: Path, key: str) -> dict:
    data = load_results_json(path)
    return {item[key]: item for item in data}


def run_skip_only_urbansound(tau: float = DEFAULT_TAU) -> dict:
    cond_a = load_results_json(RESULTS_DIR / "condition_a_urbansound8k.json")
    p_ns = load_and_index(RESULTS_DIR / "p_nospeech_urbansound8k.json", "clip_id")

    clip_ids = []
    transcriptions = []
    per_clip = []
    num_skipped = 0

    for item in cond_a:
        cid = item["clip_id"]
        p = p_ns[cid]["p_no_speech"]
        triggered = p >= tau

        if triggered:
            text = ""
            num_skipped += 1
        else:
            text = item["transcription"]

        clip_ids.append(cid)
        transcriptions.append(text)
        per_clip.append({
            "clip_id": cid,
            "transcription": text,
            "p_no_speech": p,
            "triggered": triggered,
            "original_transcription": item["transcription"],
        })

    metrics = compute_hallucination_rate(clip_ids, transcriptions)

    save_results_json(per_clip, RESULTS_DIR / "skip_only_urbansound8k.json")

    return {
        "hallucination_rate": metrics["hallucination_rate"],
        "hallucination_rate_pct": round(metrics["hallucination_rate"] * 100, 2),
        "num_hallucinated": metrics["num_hallucinated"],
        "total_clips": metrics["total_clips"],
        "num_skipped": num_skipped,
        "frac_skipped": round(num_skipped / metrics["total_clips"], 6),
    }


def run_skip_only_librispeech(split: str, tau: float = DEFAULT_TAU) -> dict:
    split_label = split.replace(".", "_").replace("-", "_")
    cond_a = load_results_json(RESULTS_DIR / f"condition_a_librispeech_{split_label}.json")
    p_ns = load_and_index(RESULTS_DIR / f"p_nospeech_librispeech_{split_label}.json", "utt_id")

    references = []
    hypotheses = []
    per_utt = []
    num_skipped = 0

    for item in cond_a:
        uid = item["utt_id"]
        p = p_ns[uid]["p_no_speech"]
        triggered = p >= tau

        if triggered:
            hyp = ""
            num_skipped += 1
        else:
            hyp = item["hypothesis"]

        references.append(item["reference"])
        hypotheses.append(hyp)
        per_utt.append({
            "utt_id": uid,
            "reference": item["reference"],
            "hypothesis": hyp,
            "p_no_speech": p,
            "triggered": triggered,
            "original_hypothesis": item["hypothesis"],
        })

    metrics = compute_wer(references, hypotheses)

    save_results_json(per_utt, RESULTS_DIR / f"skip_only_librispeech_{split_label}.json")

    return {
        "wer_percent": metrics["wer_percent"],
        "num_utterances": metrics["num_utterances"],
        "num_skipped": num_skipped,
        "frac_skipped": round(num_skipped / metrics["num_utterances"], 6),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    tau = args.tau

    print(f"=== Skip-Only Ablation (tau={tau}) ===\n")

    print("--- UrbanSound8K ---")
    us8k = run_skip_only_urbansound(tau=tau)
    print(f"  Hallucination rate: {us8k['hallucination_rate_pct']}%")
    print(f"  Skipped: {us8k['num_skipped']}/{us8k['total_clips']} ({us8k['frac_skipped']:.4f})")

    print("\n--- LibriSpeech test-clean ---")
    ls_clean = run_skip_only_librispeech("test_clean", tau=tau)
    print(f"  WER: {ls_clean['wer_percent']}%")
    print(f"  Skipped: {ls_clean['num_skipped']}/{ls_clean['num_utterances']} ({ls_clean['frac_skipped']:.4f})")

    print("\n--- LibriSpeech test-other ---")
    ls_other = run_skip_only_librispeech("test_other", tau=tau)
    print(f"  WER: {ls_other['wer_percent']}%")
    print(f"  Skipped: {ls_other['num_skipped']}/{ls_other['num_utterances']} ({ls_other['frac_skipped']:.4f})")

    summary = {
        "condition": f"D (Skip-Only, tau={tau})",
        "description": "When p_no_speech >= tau, output empty transcription; otherwise use Condition A default Whisper.",
        "tau": tau,
        "urbansound8k": us8k,
        "librispeech_test_clean": ls_clean,
        "librispeech_test_other": ls_other,
    }

    save_summary(summary, RESULTS_DIR / "skip_only_summary.json")
    print(f"\nSummary saved to {RESULTS_DIR / 'skip_only_summary.json'}")


if __name__ == "__main__":
    main()
