# FP32 oracle shift microbenchmark: measures D_logit(j) and output-logit drift
# under position shifts using FP32 eager attention (no BF16 anywhere).
# Establishes the oracle upper bound -- drift should be near-zero, confirming
# that BF16 precision (not model architecture) causes shift-induced drift.
# Restricted to T <= 2048 to avoid OOM with full FP32 attention matrices.

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sinkcast.baselines.fp32_oracle import fp32_forward, fp32_forward_with_hooks
from sinkcast.models.loader import load_model_and_tokenizer

SHIFT_PAIRS = [(0, 16), (0, 256), (0, 4096)]
SEQ_LENGTHS = [512, 1024, 2048]
KEY_INDICES = [0, 1, 2, 8, 64]
SEED = 42

SAMPLE_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
    "modern AI were planted by philosophers who attempted to describe the process of human thinking "
    "as the mechanical manipulation of symbols. This work culminated in the invention of the "
    "programmable digital computer in the 1940s, a machine based on the abstract essence of "
    "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
    "to begin seriously discussing the possibility of building an electronic brain. The field of AI "
    "research was founded at a workshop held on the campus of Dartmouth College, USA during the "
    "summer of 1956. Those who attended would become the leaders of AI research for decades. Many "
    "of them predicted that a machine as intelligent as a human being would exist in no more than "
    "a generation, and they were given millions of dollars to make this vision come true. Eventually, "
    "it became obvious that commercial developers and researchers had grossly underestimated the "
    "difficulty of the project. In 1974, in response to the criticism from James Lighthill and "
    "ongoing pressure from congress, the U.S. and British governments cut off exploratory research "
    "in AI. The next few years would later be called an AI winter, a period when obtaining funding "
    "for AI projects was difficult. In the early 1980s, AI research was revived by the commercial "
    "success of expert systems, a form of AI program that simulated the knowledge and analytical skills "
    "of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, "
    "Japan's fifth generation computer project inspired the U.S and British governments to restore "
    "funding for academic research. However, beginning with the collapse of the Lisp Machine market "
    "in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began. Many "
    "researchers began to doubt that the symbolic approach would ever be able to imitate all the "
    "processes of human cognition, especially perception, robotics, learning and pattern recognition. "
    "A number of researchers began to look into sub-symbolic approaches. Robotics researchers, such as "
    "Rodney Brooks, rejected symbolic AI and focused on the basic engineering problems that would allow "
    "robots to move, survive, and learn their environment. Interest in neural networks and connectionism "
    "was revived by Geoffrey Hinton, David Rumelhart and others in the middle of the 1980s. Soft "
    "computing tools were developed in the 80s, such as neural networks, fuzzy systems, Grey system "
    "theory, evolutionary computation and many tools drawn from statistics or mathematical optimization. "
    "AI gradually restored its reputation in the late 1990s and early 21st century by finding specific "
    "solutions to specific problems. The narrow focus allowed researchers to produce verifiable results, "
    "exploit more mathematical methods, and collaborate with other fields such as statistics, economics "
    "and mathematics. By 2000, solutions developed by AI researchers were being widely used, although in "
    "the 1990s they were rarely described as artificial intelligence. Faster computers, algorithmic "
    "improvements, and access to large amounts of data enabled advances in machine learning and "
    "perception; data-hungry deep learning methods started to dominate accuracy benchmarks around 2012. "
    "According to Bloomberg's Jack Clark, 2015 was a landmark year for artificial intelligence, with "
    "the number of software projects that use AI within Google increased from a sporadic usage in 2012 "
    "to more than 2,700 projects. Clark also presents factual data indicating that error rates in image "
    "processing tasks have fallen significantly since 2011. He attributes this to an increase in "
    "affordable neural networks, due to a rise in cloud computing infrastructure and to an increase in "
    "research tools and datasets. In a 2017 survey, one in five companies reported they had incorporated "
    "AI in some offerings or processes. The amount of research into AI, measured by total publications, "
    "increased by 50 percent in the years from 2015 through 2019. Numerous academic researchers became "
    "combative toward the firms who make AI. "
)


def prepare_input(tokenizer, seq_len: int) -> torch.Tensor:
    tokens = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)
    if len(tokens) < seq_len:
        repeats = math.ceil(seq_len / len(tokens))
        tokens = (tokens * repeats)[:seq_len]
    else:
        tokens = tokens[:seq_len]
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        tokens[0] = tokenizer.bos_token_id
    return torch.tensor([tokens], dtype=torch.long)


def compute_d_logit(
    attn_logits_1: dict, attn_logits_2: dict, seq_len: int
) -> dict:
    d_logit = {}
    for j in KEY_INDICES:
        total = 0.0
        count = 0
        for layer_idx in attn_logits_1:
            if j not in attn_logits_1[layer_idx] or j not in attn_logits_2[layer_idx]:
                continue
            a1 = attn_logits_1[layer_idx][j]
            a2 = attn_logits_2[layer_idx][j]
            diff = (a1 - a2).abs().sum().item()
            total += diff
            count += a1.shape[1]
        if count > 0:
            d_logit[j] = total / seq_len
        else:
            d_logit[j] = 0.0
    return d_logit


def compute_output_drift(logits1: torch.Tensor, logits2: torch.Tensor) -> dict:
    diff = (logits1.float() - logits2.float()).abs()
    max_drift = diff.max().item()
    mean_drift = diff.mean().item()
    return {"max_drift": max_drift, "mean_drift": mean_drift}


def run_fp32_shift_benchmark(
    model,
    tokenizer,
    model_name: str,
    seq_lengths: list = None,
    shift_pairs: list = None,
    output_dir: str = None,
    debug: bool = False,
):
    if seq_lengths is None:
        seq_lengths = SEQ_LENGTHS
    if shift_pairs is None:
        shift_pairs = SHIFT_PAIRS

    results = {
        "model": model_name,
        "method": "fp32_oracle",
        "seed": SEED,
        "key_indices": KEY_INDICES,
        "entries": [],
    }

    for seq_len in seq_lengths:
        input_ids = prepare_input(tokenizer, seq_len)
        valid_keys = [j for j in KEY_INDICES if j < seq_len]

        for delta1, delta2 in shift_pairs:
            print(f"  seq_len={seq_len}, shift=({delta1},{delta2})...")
            t0 = time.time()

            logits1, attn1 = fp32_forward_with_hooks(
                model, input_ids, position_offset=delta1, key_indices=valid_keys
            )
            logits2, attn2 = fp32_forward_with_hooks(
                model, input_ids, position_offset=delta2, key_indices=valid_keys
            )

            d_logit = compute_d_logit(attn1, attn2, seq_len)
            drift = compute_output_drift(logits1, logits2)

            d_logit_sum = sum(d_logit.values())
            j0_fraction = d_logit.get(0, 0.0) / d_logit_sum if d_logit_sum > 0 else 0.0

            entry = {
                "seq_len": seq_len,
                "shift_pair": [delta1, delta2],
                "d_logit": {str(k): v for k, v in d_logit.items()},
                "d_logit_sum": d_logit_sum,
                "j0_fraction": j0_fraction,
                "max_drift": drift["max_drift"],
                "mean_drift": drift["mean_drift"],
                "time_sec": round(time.time() - t0, 2),
            }
            results["entries"].append(entry)
            print(f"    D_logit(0)={d_logit.get(0,0):.6f}, j0_frac={j0_fraction:.4f}, "
                  f"max_drift={drift['max_drift']:.6f}, mean_drift={drift['mean_drift']:.8f}, "
                  f"time={entry['time_sec']}s")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_").replace("-", "_")
        out_path = os.path.join(output_dir, f"{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FP32 Oracle shift microbenchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Model alias: llama-3.1-8b or mistral-7b-v0.3")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS)
    parser.add_argument("--shift-pairs", type=str, default=None,
                        help="Comma-separated shift pairs, e.g. '0:16,0:256'")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "sinkcast", "results",
                                             "microbench", "fp32_oracle"))
    parser.add_argument("--debug", action="store_true",
                        help="Run minimal config for debugging")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    shift_pairs = SHIFT_PAIRS
    if args.shift_pairs:
        shift_pairs = []
        for pair in args.shift_pairs.split(","):
            d1, d2 = pair.split(":")
            shift_pairs.append((int(d1), int(d2)))

    seq_lengths = args.seq_lengths
    if args.debug:
        seq_lengths = [seq_lengths[0]]
        shift_pairs = [shift_pairs[0]]

    print(f"Loading model: {args.model} in FP32 with eager attention...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    print(f"Model loaded. Seq lengths: {seq_lengths}, Shift pairs: {shift_pairs}")

    results = run_fp32_shift_benchmark(
        model, tokenizer, args.model,
        seq_lengths=seq_lengths,
        shift_pairs=shift_pairs,
        output_dir=args.output_dir,
        debug=args.debug,
    )

    print("Done.")
    return results


if __name__ == "__main__":
    main()
