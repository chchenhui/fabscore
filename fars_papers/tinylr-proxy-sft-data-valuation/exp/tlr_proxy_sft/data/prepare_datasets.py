"""Download 12 math-domain ODA datasets, sample 50k each, convert to Alpaca format,
run 10-gram contamination check against GSM8K/MATH-500, and save as JSON for LLaMA-Factory."""

import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 0
SAMPLE_SIZE = 50_000
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = [
    "AM-Thinking-v1-Distilled-math",
    "DeepMath-309K",
    "Maths-College",
    "OpenR1-Math",
    "QwQ-LongCoT-130K-math",
    "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard",
    "mathplus",
    "numinamath-cot",
    "numinamath1_5",
    "openmathinstruct-2",
    "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]

HF_REPO = "OpenDataArena/OpenDataArena-scored-data"


def ngrams(text: str, n: int) -> list[str]:
    tokens = text.lower().split()
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def load_benchmark_prompts() -> list[str]:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [row["question"] for row in gsm]
    prompts += [row["problem"] for row in math500]
    return prompts


def build_benchmark_ngram_set(prompts: list[str], n: int = 10) -> set[str]:
    s = set()
    for p in prompts:
        s.update(ngrams(p, n))
    return s


def compute_overlap(dataset_instructions: list[str], benchmark_ngrams: set[str], n: int = 10) -> float:
    if not benchmark_ngrams:
        return 0.0
    overlap_count = 0
    total_count = 0
    for instr in dataset_instructions:
        grams = ngrams(instr, n)
        total_count += len(grams)
        overlap_count += sum(1 for g in grams if g in benchmark_ngrams)
    return overlap_count / total_count if total_count > 0 else 0.0


def safe_name(subset: str) -> str:
    return subset.replace("/", "__").replace(" ", "_")


def main():
    print("Loading benchmark prompts for contamination check...")
    bench_prompts = load_benchmark_prompts()
    bench_ngrams = build_benchmark_ngram_set(bench_prompts, n=10)
    print(f"  Built {len(bench_ngrams)} unique 10-grams from {len(bench_prompts)} benchmark prompts")

    contamination_report = {}

    for subset in SUBSETS:
        name = safe_name(subset)
        out_path = PROCESSED_DIR / f"{name}.json"
        print(f"\n--- Processing {subset} ---")

        ds = load_dataset(HF_REPO, subset, split="train")
        total = len(ds)
        print(f"  Total rows: {total}")

        if total > SAMPLE_SIZE:
            rng = random.Random(SEED)
            indices = rng.sample(range(total), SAMPLE_SIZE)
            ds = ds.select(indices)
            print(f"  Sampled {SAMPLE_SIZE} rows")
        else:
            print(f"  Using all {total} rows (< {SAMPLE_SIZE})")

        records = []
        instructions = []
        for row in ds:
            instr = row["instruction"]
            output = row["output"]
            records.append({"instruction": instr, "input": "", "output": output})
            instructions.append(instr)

        overlap_frac = compute_overlap(instructions, bench_ngrams, n=10)
        contamination_report[name] = {
            "subset": subset,
            "num_samples": len(records),
            "overlap_10gram_fraction": round(overlap_frac, 6),
            "flagged": overlap_frac > 0.05,
        }
        flag_str = " ** FLAGGED (>5%) **" if overlap_frac > 0.05 else ""
        print(f"  10-gram overlap: {overlap_frac:.4%}{flag_str}")

        with open(out_path, "w") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"  Saved {len(records)} records to {out_path}")

    report_path = PROCESSED_DIR / "contamination_report.json"
    with open(report_path, "w") as f:
        json.dump(contamination_report, f, indent=2, ensure_ascii=False)
    print(f"\nContamination report saved to {report_path}")

    print("\n=== Summary ===")
    for name, info in contamination_report.items():
        flag = "FLAGGED" if info["flagged"] else "OK"
        print(f"  {name}: {info['num_samples']} samples, overlap={info['overlap_10gram_fraction']:.4%} [{flag}]")


if __name__ == "__main__":
    main()
