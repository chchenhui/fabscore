"""Compute tau-robustness table for SCHM suppress mode.
Loads pre-computed p_nospeech values and per-clip transcriptions from
Conditions A and B, then re-derives SCHM decisions at each threshold
tau in {0.5, 0.6, 0.7} plus boundary conditions (A=never mask, B=always mask).
No GPU needed -- entirely offline computation.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from schm.evaluation.wer_eval import compute_wer


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TAU_LIST = [0.5, 0.6, 0.7]


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def build_lookup(records, key_field):
    return {r[key_field]: r for r in records}


def compute_us8k_suppress(p_lookup, tau):
    total = len(p_lookup)
    triggered = sum(1 for v in p_lookup.values() if v["p_no_speech"] >= tau)
    halluc = total - triggered
    return {
        "hallucination_rate_pct": round(halluc / total * 100, 2),
        "num_hallucinated": halluc,
        "total_clips": total,
        "frac_triggered": round(triggered / total, 6),
        "num_triggered": triggered,
    }


def compute_ls_suppress(p_records, cond_a_records, tau):
    a_lookup = build_lookup(cond_a_records, "utt_id")
    refs, hyps = [], []
    triggered = 0
    for rec in p_records:
        uid = rec["utt_id"]
        a_rec = a_lookup[uid]
        ref = a_rec["reference"]
        if rec["p_no_speech"] >= tau:
            hyp = ""
            triggered += 1
        else:
            hyp = a_rec["hypothesis"]
        refs.append(ref)
        hyps.append(hyp)
    wer_result = compute_wer(refs, hyps)
    return {
        "wer_percent": wer_result["wer_percent"],
        "num_utterances": len(p_records),
        "frac_triggered": round(triggered / len(p_records), 6),
        "num_triggered": triggered,
    }


def compute_ls_mask(p_records, cond_a_records, cond_b_records, tau):
    a_lookup = build_lookup(cond_a_records, "utt_id")
    b_lookup = build_lookup(cond_b_records, "utt_id")
    refs, hyps = [], []
    triggered = 0
    for rec in p_records:
        uid = rec["utt_id"]
        a_rec = a_lookup[uid]
        ref = a_rec["reference"]
        if rec["p_no_speech"] >= tau:
            hyp = b_lookup[uid]["hypothesis"]
            triggered += 1
        else:
            hyp = a_rec["hypothesis"]
        refs.append(ref)
        hyps.append(hyp)
    wer_result = compute_wer(refs, hyps)
    return {
        "wer_percent": wer_result["wer_percent"],
        "num_utterances": len(p_records),
        "frac_triggered": round(triggered / len(p_records), 6),
        "num_triggered": triggered,
    }


def main():
    print("Loading data...")
    p_us8k = load_json("p_nospeech_urbansound8k.json")
    p_ls_clean = load_json("p_nospeech_librispeech_test_clean.json")
    p_ls_other = load_json("p_nospeech_librispeech_test_other.json")

    a_us8k = load_json("condition_a_urbansound8k.json")
    a_ls_clean = load_json("condition_a_librispeech_test_clean.json")
    a_ls_other = load_json("condition_a_librispeech_test_other.json")

    b_us8k = load_json("condition_b_urbansound8k.json")
    b_ls_clean = load_json("condition_b_librispeech_test_clean.json")
    b_ls_other = load_json("condition_b_librispeech_test_other.json")

    p_us8k_lookup = build_lookup(p_us8k, "clip_id")

    a_summary = load_json("condition_a_summary.json")
    b_summary = load_json("condition_b_summary.json")

    rows = []

    cond_a_row = {
        "tau": "inf",
        "label": "Condition A (never mask)",
        "mode": "baseline",
        "us8k_halluc_rate_pct": round(a_summary["urbansound8k"]["hallucination_rate"] * 100, 2),
        "us8k_frac_masked": 0.0,
        "ls_clean_wer_pct": a_summary["librispeech_test_clean"]["wer_percent"],
        "ls_clean_frac_masked": 0.0,
        "ls_other_wer_pct": a_summary["librispeech_test_other"]["wer_percent"],
        "ls_other_frac_masked": 0.0,
    }
    rows.append(cond_a_row)

    for tau in TAU_LIST:
        print(f"\n--- tau = {tau} ---")

        us8k_sup = compute_us8k_suppress(p_us8k_lookup, tau)
        print(f"  US8K suppress: halluc={us8k_sup['hallucination_rate_pct']}%, triggered={us8k_sup['num_triggered']}/{us8k_sup['total_clips']}")

        ls_clean_sup = compute_ls_suppress(p_ls_clean, a_ls_clean, tau)
        print(f"  LS-clean suppress: WER={ls_clean_sup['wer_percent']}%, triggered={ls_clean_sup['num_triggered']}/{ls_clean_sup['num_utterances']}")

        ls_other_sup = compute_ls_suppress(p_ls_other, a_ls_other, tau)
        print(f"  LS-other suppress: WER={ls_other_sup['wer_percent']}%, triggered={ls_other_sup['num_triggered']}/{ls_other_sup['num_utterances']}")

        ls_clean_mask = compute_ls_mask(p_ls_clean, a_ls_clean, b_ls_clean, tau)
        print(f"  LS-clean mask: WER={ls_clean_mask['wer_percent']}%, triggered={ls_clean_mask['num_triggered']}")

        ls_other_mask = compute_ls_mask(p_ls_other, a_ls_other, b_ls_other, tau)
        print(f"  LS-other mask: WER={ls_other_mask['wer_percent']}%, triggered={ls_other_mask['num_triggered']}")

        row = {
            "tau": tau,
            "label": f"SCHM suppress (tau={tau})",
            "mode": "suppress",
            "us8k_halluc_rate_pct": us8k_sup["hallucination_rate_pct"],
            "us8k_frac_masked": us8k_sup["frac_triggered"],
            "ls_clean_wer_pct": ls_clean_sup["wer_percent"],
            "ls_clean_frac_masked": ls_clean_sup["frac_triggered"],
            "ls_other_wer_pct": ls_other_sup["wer_percent"],
            "ls_other_frac_masked": ls_other_sup["frac_triggered"],
            "mask_mode_ls_clean_wer_pct": ls_clean_mask["wer_percent"],
            "mask_mode_ls_other_wer_pct": ls_other_mask["wer_percent"],
        }
        rows.append(row)

    cond_b_row = {
        "tau": 0,
        "label": "Condition B (always mask)",
        "mode": "baseline",
        "us8k_halluc_rate_pct": round(b_summary["urbansound8k"]["hallucination_rate"] * 100, 2),
        "us8k_frac_masked": 1.0,
        "ls_clean_wer_pct": b_summary["librispeech_test_clean"]["wer_percent"],
        "ls_clean_frac_masked": 1.0,
        "ls_other_wer_pct": b_summary["librispeech_test_other"]["wer_percent"],
        "ls_other_frac_masked": 1.0,
    }
    rows.append(cond_b_row)

    output = {
        "description": "Tau robustness analysis for SCHM (suppress mode). Rows sorted by tau descending (A=inf -> 0.7 -> 0.6 -> 0.5 -> B=0).",
        "columns": [
            "tau", "label", "us8k_halluc_rate_pct", "us8k_frac_masked",
            "ls_clean_wer_pct", "ls_clean_frac_masked",
            "ls_other_wer_pct", "ls_other_frac_masked"
        ],
        "rows": rows,
    }

    out_path = os.path.join(RESULTS_DIR, "tau_robustness.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Robustness Table (suppress mode) ===")
    print(f"{'tau':>6s} | {'US8K Halluc%':>12s} | {'US8K %mask':>10s} | {'LS-clean WER%':>13s} | {'LS-clean %mask':>14s} | {'LS-other WER%':>13s} | {'LS-other %mask':>14s}")
    print("-" * 100)
    for r in rows:
        tau_str = str(r["tau"])
        print(f"{tau_str:>6s} | {r['us8k_halluc_rate_pct']:>12.2f} | {r['us8k_frac_masked']:>10.4f} | {r['ls_clean_wer_pct']:>13.2f} | {r['ls_clean_frac_masked']:>14.4f} | {r['ls_other_wer_pct']:>13.2f} | {r['ls_other_frac_masked']:>14.4f}")


if __name__ == "__main__":
    main()
