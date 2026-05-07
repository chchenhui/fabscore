# Compute BF16/FP32 drift ratios for the FP32 oracle sanity check.
# Reads BF16 and FP32 results, matches by (model, seq_len, shift_pair),
# computes drift_bf16/drift_fp32 ratios, saves to sanity_check.json.

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

BF16_DIR = os.path.join(PROJECT_ROOT, "sinkcast", "results", "microbench", "bf16_flash")
FP32_DIR = os.path.join(PROJECT_ROOT, "sinkcast", "results", "microbench", "fp32_oracle")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def make_key(entry):
    return (entry["seq_len"], tuple(entry["shift_pair"]))


def compute_ratios(bf16_data, fp32_data):
    bf16_by_key = {make_key(e): e for e in bf16_data["entries"]}
    fp32_by_key = {make_key(e): e for e in fp32_data["entries"]}

    comparisons = []
    for key in sorted(fp32_by_key.keys()):
        if key not in bf16_by_key:
            continue
        bf16 = bf16_by_key[key]
        fp32 = fp32_by_key[key]

        max_drift_ratio = bf16["max_drift"] / fp32["max_drift"] if fp32["max_drift"] > 0 else float("inf")
        mean_drift_ratio = bf16["mean_drift"] / fp32["mean_drift"] if fp32["mean_drift"] > 0 else float("inf")

        d_logit_ratios = {}
        for j_str in fp32["d_logit"]:
            fp32_val = fp32["d_logit"][j_str]
            bf16_val = bf16["d_logit"].get(j_str, 0.0)
            d_logit_ratios[j_str] = bf16_val / fp32_val if fp32_val > 0 else float("inf")

        d_logit_sum_ratio = bf16["d_logit_sum"] / fp32["d_logit_sum"] if fp32["d_logit_sum"] > 0 else float("inf")

        comparisons.append({
            "seq_len": key[0],
            "shift_pair": list(key[1]),
            "bf16_max_drift": bf16["max_drift"],
            "fp32_max_drift": fp32["max_drift"],
            "max_drift_ratio": round(max_drift_ratio, 1),
            "bf16_mean_drift": bf16["mean_drift"],
            "fp32_mean_drift": fp32["mean_drift"],
            "mean_drift_ratio": round(mean_drift_ratio, 1),
            "d_logit_ratios": {k: round(v, 1) for k, v in d_logit_ratios.items()},
            "d_logit_sum_ratio": round(d_logit_sum_ratio, 1),
        })

    return comparisons


def main():
    models = [
        ("llama_3.1_8b", "llama-3.1-8b"),
        ("mistral_7b_v0.3", "mistral-7b-v0.3"),
    ]

    sanity = {"experiment": "FP32 Oracle Sanity Check", "models": {}}

    for filename, model_name in models:
        bf16_path = os.path.join(BF16_DIR, f"{filename}.json")
        fp32_path = os.path.join(FP32_DIR, f"{filename}.json")

        if not os.path.exists(bf16_path) or not os.path.exists(fp32_path):
            print(f"Skipping {model_name}: missing results")
            continue

        bf16_data = load_results(bf16_path)
        fp32_data = load_results(fp32_path)
        comparisons = compute_ratios(bf16_data, fp32_data)

        max_drift_ratios = [c["max_drift_ratio"] for c in comparisons]
        mean_drift_ratios = [c["mean_drift_ratio"] for c in comparisons]

        sanity["models"][model_name] = {
            "comparisons": comparisons,
            "summary": {
                "min_max_drift_ratio": min(max_drift_ratios),
                "max_max_drift_ratio": max(max_drift_ratios),
                "avg_max_drift_ratio": round(sum(max_drift_ratios) / len(max_drift_ratios), 1),
                "min_mean_drift_ratio": min(mean_drift_ratios),
                "max_mean_drift_ratio": max(mean_drift_ratios),
                "avg_mean_drift_ratio": round(sum(mean_drift_ratios) / len(mean_drift_ratios), 1),
            }
        }

        print(f"\n=== {model_name} ===")
        for c in comparisons:
            print(f"  T={c['seq_len']}, shift={c['shift_pair']}: "
                  f"max_drift ratio={c['max_drift_ratio']:.0f}x, "
                  f"mean_drift ratio={c['mean_drift_ratio']:.0f}x")
        s = sanity["models"][model_name]["summary"]
        print(f"  Summary: max_drift ratio range [{s['min_max_drift_ratio']:.0f}x, {s['max_max_drift_ratio']:.0f}x], "
              f"avg={s['avg_max_drift_ratio']:.0f}x")

    out_path = os.path.join(FP32_DIR, "sanity_check.json")
    with open(out_path, "w") as f:
        json.dump(sanity, f, indent=2)
    print(f"\nSanity check saved to {out_path}")


if __name__ == "__main__":
    main()
