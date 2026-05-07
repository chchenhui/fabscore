"""Phase-1 go/no-go diagnostic: cross-reference p_no_speech with Condition A
hallucination results on UrbanSound8K. Extended version with per-class breakdown,
sweep over tau thresholds {0.2..0.8}, and revised go/no-go criterion:
GO if there exists a tau where >= 20% of hallucinating clips trigger.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schm.evaluation.results_io import load_results_json

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

US8K_CLASSES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

TAU_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GO_CRITERION_FRAC = 0.20


def _extract_class_id(clip_id: str) -> int:
    parts = clip_id.replace(".wav", "").split("-")
    return int(parts[1])


def _distribution_stats(arr):
    if len(arr) == 0:
        return None
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
    }


def run_diagnostic():
    cond_a_path = RESULTS_DIR / "condition_a_urbansound8k.json"
    p_nospeech_path = RESULTS_DIR / "p_nospeech_urbansound8k.json"

    cond_a = load_results_json(cond_a_path)
    p_nospeech_data = load_results_json(p_nospeech_path)

    print(f"Loaded {len(cond_a)} Condition A results", flush=True)
    print(f"Loaded {len(p_nospeech_data)} p_no_speech results", flush=True)

    p_map = {r["clip_id"]: r["p_no_speech"] for r in p_nospeech_data}

    all_p = []
    hall_p = []
    non_hall_p = []
    class_hall_p = defaultdict(list)
    class_all_p = defaultdict(list)
    matched = 0

    for clip in cond_a:
        cid = clip["clip_id"]
        if cid not in p_map:
            continue
        matched += 1
        p_val = p_map[cid]
        class_id = _extract_class_id(cid)
        is_hall = len(clip["transcription"].strip()) > 0
        all_p.append(p_val)
        class_all_p[class_id].append(p_val)
        if is_hall:
            hall_p.append(p_val)
            class_hall_p[class_id].append(p_val)
        else:
            non_hall_p.append(p_val)

    print(f"Matched {matched} clips", flush=True)

    all_arr = np.array(all_p)
    hall_arr = np.array(hall_p)

    total_hall = len(hall_p)
    total_non_hall = len(non_hall_p)

    trigger_sweep = {}
    for tau in TAU_THRESHOLDS:
        n_above = int(np.sum(hall_arr > tau)) if total_hall > 0 else 0
        frac = n_above / total_hall if total_hall > 0 else 0.0
        trigger_sweep[f"tau_{tau:.1f}"] = {
            "tau": tau,
            "hall_clips_above": n_above,
            "fraction": round(frac, 4),
        }

    best_tau_for_go = None
    for tau in TAU_THRESHOLDS:
        key = f"tau_{tau:.1f}"
        if trigger_sweep[key]["fraction"] >= GO_CRITERION_FRAC:
            best_tau_for_go = tau
            break

    go_decision = best_tau_for_go is not None

    per_class_analysis = {}
    for class_id in sorted(US8K_CLASSES.keys()):
        class_name = US8K_CLASSES[class_id]
        c_all = np.array(class_all_p[class_id]) if class_id in class_all_p else np.array([])
        c_hall = np.array(class_hall_p[class_id]) if class_id in class_hall_p else np.array([])
        class_trigger = {}
        for tau in TAU_THRESHOLDS:
            n_above = int(np.sum(c_hall > tau)) if len(c_hall) > 0 else 0
            frac = n_above / len(c_hall) if len(c_hall) > 0 else 0.0
            class_trigger[f"tau_{tau:.1f}"] = {
                "hall_clips_above": n_above,
                "fraction": round(frac, 4),
            }
        per_class_analysis[class_name] = {
            "class_id": class_id,
            "total_clips": len(c_all),
            "hallucinating_clips": len(c_hall),
            "distribution": _distribution_stats(c_all),
            "trigger_sweep": class_trigger,
        }

    report = {
        "total_clips": matched,
        "total_hallucinating": total_hall,
        "total_non_hallucinating": total_non_hall,
        "hallucination_rate": total_hall / matched if matched > 0 else 0.0,
        "trigger_sweep": trigger_sweep,
        "distribution_all_clips": _distribution_stats(all_arr),
        "distribution_hallucinating_clips": _distribution_stats(hall_arr),
        "per_class_analysis": per_class_analysis,
        "go_no_go": {
            "criterion": f"exists tau in {TAU_THRESHOLDS} where >= {GO_CRITERION_FRAC*100:.0f}% of hallucinating clips trigger",
            "go_criterion_fraction": GO_CRITERION_FRAC,
            "best_tau_for_go": best_tau_for_go,
            "decision": "GO" if go_decision else "NO-GO",
        },
    }

    out_path = RESULTS_DIR / "phase1_diagnostic_extended.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nExtended diagnostic report saved to {out_path}", flush=True)

    print("\n=== Phase-1 Extended Diagnostic Report ===", flush=True)
    print(f"Total clips: {matched}", flush=True)
    print(f"Hallucinating clips: {total_hall} ({total_hall/matched*100:.1f}%)", flush=True)

    print(f"\n--- Trigger Sweep (all hallucinating clips) ---", flush=True)
    for tau in TAU_THRESHOLDS:
        key = f"tau_{tau:.1f}"
        info = trigger_sweep[key]
        print(f"  tau={tau:.1f}: {info['hall_clips_above']:5d} / {total_hall} = {info['fraction']*100:5.1f}%", flush=True)

    print(f"\n--- Per-Class p_no_speech Summary ---", flush=True)
    for class_id in sorted(US8K_CLASSES.keys()):
        class_name = US8K_CLASSES[class_id]
        info = per_class_analysis[class_name]
        dist = info["distribution"]
        if dist:
            t05 = info["trigger_sweep"]["tau_0.5"]["fraction"]
            t03 = info["trigger_sweep"]["tau_0.3"]["fraction"]
            print(f"  {class_name:20s} (n={info['total_clips']:4d}): "
                  f"mean={dist['mean']:.3f}, median={dist['median']:.3f}, "
                  f">0.3: {t03*100:5.1f}%, >0.5: {t05*100:5.1f}%", flush=True)

    print(f"\n--- Distribution ---", flush=True)
    dist = _distribution_stats(all_arr)
    print(f"  mean={dist['mean']:.4f}, median={dist['median']:.4f}, std={dist['std']:.4f}", flush=True)
    print(f"  percentiles: {dist['percentiles']}", flush=True)

    print(f"\n=== GO/NO-GO DECISION: {report['go_no_go']['decision']} ===", flush=True)
    if go_decision:
        print(f"  Trigger fires on >= {GO_CRITERION_FRAC*100:.0f}% at tau={best_tau_for_go:.1f}. "
              f"Proceed to full SCHM evaluation.", flush=True)
    else:
        print(f"  No tau threshold achieves >= {GO_CRITERION_FRAC*100:.0f}% trigger rate.", flush=True)

    return report


if __name__ == "__main__":
    run_diagnostic()
