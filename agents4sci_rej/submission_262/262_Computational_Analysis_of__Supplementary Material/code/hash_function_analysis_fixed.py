"""
Corrected cryptographic hash function analysis.

Fixes vs original:
- Uses perf_counter_ns for reliable timing, aggregates over batches
- Normalizes avalanche by actual digest bit-length per algorithm
- Counts collisions only among distinct inputs; structured data salted to avoid duplicates
- Uses BLAKE2b with digest_size=32 (256 bits) for apples-to-apples comparison
- Runs identical test vectors across algorithms for fairness
"""

from __future__ import annotations

import hashlib
import os
import os as _os
import json
import math
import random
import statistics
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


_seed = int(_os.getenv("SEED", "42"))
RNG = random.Random(_seed)
NP_RNG = np.random.default_rng(_seed)


@dataclass
class AlgoSpec:
    name: str
    digest_bits: int


ALGORITHMS: Dict[str, Tuple[AlgoSpec, callable]] = {
    "MD5": (AlgoSpec("MD5", 128), hashlib.md5),
    "SHA-256": (AlgoSpec("SHA-256", 256), hashlib.sha256),
    "SHA3-256": (AlgoSpec("SHA3-256", 256), hashlib.sha3_256),
    # Wrap blake2b to fix digest_size=32 (256 bits)
    "BLAKE2b-256": (AlgoSpec("BLAKE2b-256", 256), lambda d: hashlib.blake2b(d, digest_size=32)),
}


def generate_test_vectors(input_sizes: List[int], input_types: List[str], per_type_count: int) -> Dict[Tuple[int, str], List[bytes]]:
    """Generate identical test vectors for all algorithms.

    - random: cryptographically random bytes
    - structured: repeating patterns with index-based salt to avoid duplicates
    - edge: all-zero, all-ones, low-entropy patterns salted to be distinct
    """
    vectors: Dict[Tuple[int, str], List[bytes]] = {}
    for size in input_sizes:
        for itype in input_types:
            cur: List[bytes] = []
            if itype == "random":
                for _ in range(per_type_count):
                    cur.append(NP_RNG.integers(0, 256, size, dtype=np.uint8).tobytes())
            elif itype == "structured":
                base_patterns = [b"ABCD" * (size // 4), b"1234" * (size // 4), (b"\xAA\x55" * (size // 2))[:size]]
                for i in range(per_type_count):
                    p = base_patterns[i % len(base_patterns)]
                    salt = i.to_bytes(4, "big")
                    data = (salt + p)[:size]
                    if len(data) < size:
                        data = (data + p)[:size]
                    cur.append(data)
            elif itype == "edge":
                bases = [bytes([0]) * size, bytes([255]) * size, bytes([0x0F]) * size, bytes([0xF0]) * size]
                for i in range(per_type_count):
                    b = bases[i % len(bases)]
                    salt = RNG.randbytes(min(8, max(1, size))) if hasattr(RNG, "randbytes") else os.urandom(min(8, max(1, size)))
                    data = (salt + b)[:size]
                    if len(data) < size:
                        data = (data + b)[:size]
                    cur.append(data)
            else:
                raise ValueError(f"Unknown input type: {itype}")
            vectors[(size, itype)] = cur
    return vectors


def hash_bytes(factory: callable, data: bytes) -> bytes:
    return factory(data).digest()


def time_hashing(factory: callable, dataset: List[bytes], repeat: int = 5) -> Tuple[float, float]:
    """Return (avg_MBps, avg_ms_per_hash). Times the full dataset, repeated, using perf_counter_ns.
    """
    total_bytes = sum(len(d) for d in dataset)
    num_hashes = len(dataset)
    elapsed_ns: List[int] = []
    for _ in range(repeat):
        t0 = perf_counter_ns()
        for d in dataset:
            _ = hash_bytes(factory, d)
        t1 = perf_counter_ns()
        elapsed_ns.append(t1 - t0)
    avg_ns = statistics.mean(elapsed_ns)
    avg_s = avg_ns / 1e9
    mbps = (total_bytes / (1024 * 1024)) / avg_s if avg_s > 0 else float("inf")
    ms_per_hash = (avg_s / num_hashes) * 1000.0
    return mbps, ms_per_hash


def bit_diff_count(a: bytes, b: bytes) -> int:
    diff = 0
    for xb, yb in zip(a, b):
        z = xb ^ yb
        diff += z.bit_count()
    return diff


def avalanche_score(factory: callable, digest_bits: int, data: bytes) -> float:
    """Flip one random bit in input and measure differing output bits normalized by digest_bits."""
    original = hash_bytes(factory, data)
    if len(data) == 0:
        return 0.0
    bit_pos = RNG.randrange(len(data) * 8)
    byte_index, bit_index = divmod(bit_pos, 8)
    flipped = bytearray(data)
    flipped[byte_index] ^= (1 << bit_index)
    changed = hash_bytes(factory, bytes(flipped))
    diff_bits = bit_diff_count(original, changed)
    return diff_bits / float(digest_bits)


def per_bit_entropy(digests: List[bytes], digest_bits: int) -> float:
    if not digests:
        return 0.0
    digest_bytes = math.ceil(digest_bits / 8)
    # Compute probability of bit=1 across samples
    ones = np.zeros(digest_bits, dtype=np.float64)
    for dg in digests:
        bits = np.unpackbits(np.frombuffer(dg[:digest_bytes], dtype=np.uint8))
        ones += bits[:digest_bits]
    p = ones / len(digests)
    # Shannon entropy per bit, mean over bits
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12))
    return float(np.nanmean(H))


def measure_algorithm(spec: AlgoSpec, factory: callable, vectors: Dict[Tuple[int, str], List[bytes]]) -> Dict:
    results = {
        "algorithm": spec.name,
        "digest_bits": spec.digest_bits,
        "by_config": [],
    }
    # For collision rate and entropy we accumulate digests across all configs
    all_digests: List[bytes] = []
    for (size, itype), dataset in vectors.items():
        mbps, ms = time_hashing(factory, dataset)
        # Avalanche: sample a subset for speed
        sample = dataset[: min(50, len(dataset))]
        aval = [avalanche_score(factory, spec.digest_bits, d) for d in sample]
        results["by_config"].append({
            "input_size": size,
            "input_type": itype,
            "throughput_MBps": mbps,
            "ms_per_hash": ms,
            "avalanche_mean": statistics.mean(aval) if aval else 0.0,
            "avalanche_median": statistics.median(aval) if aval else 0.0,
        })
        # Collect digests for uniqueness/collision checks
        for d in dataset:
            all_digests.append(hash_bytes(factory, d))
    # Distinct-input collision rate: collisions among distinct inputs only
    # Since vectors were salted to avoid duplicates, this should be ~0
    seen = set()
    collisions = 0
    for dg in all_digests:
        if dg in seen:
            collisions += 1
        else:
            seen.add(dg)
    collision_rate = collisions / max(1, len(all_digests))
    entropy = per_bit_entropy(all_digests[: min(2000, len(all_digests))], spec.digest_bits)
    results["collision_rate"] = collision_rate
    results["bit_entropy"] = entropy
    # Aggregate throughput
    tps = [c["throughput_MBps"] for c in results["by_config"]]
    results["throughput_avg_MBps"] = statistics.mean(tps) if tps else 0.0
    return results


def plot_performance(all_results: List[Dict], out_png: str, out_pdf: str) -> None:
    sns.set_theme(style="whitegrid")
    # Build a compact performance figure
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=150)
    for r in all_results:
        xs = []
        ys = []
        for c in sorted(r["by_config"], key=lambda x: (x["input_size"], x["input_type"])):
            if c["input_type"] == "random":
                xs.append(c["input_size"] / 1024)
                ys.append(c["throughput_MBps"])
        ax.plot(xs, ys, marker="o", label=r["algorithm"])
    ax.set_xscale("log")
    ax.set_xlabel("Input size (KB)")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Hash Throughput vs Input Size (random data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_security(all_results: List[Dict], out_png: str, out_pdf: str) -> None:
    sns.set_theme(style="whitegrid")
    algos = [r["algorithm"] for r in all_results]
    aval = [statistics.mean([c["avalanche_mean"] for c in r["by_config"]]) for r in all_results]
    coll = [r["collision_rate"] for r in all_results]
    ent = [r["bit_entropy"] for r in all_results]

    x = np.arange(len(algos))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=150)
    ax.bar(x - width, aval, width, label="Avalanche (mean)")
    ax.bar(x, coll, width, label="Collision rate")
    ax.bar(x + width, ent, width, label="Bit entropy (avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title("Security Metrics Summary")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def run_and_save(results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    input_sizes = [1024, 16 * 1024, 256 * 1024, 1024 * 1024]  # 1KB, 16KB, 256KB, 1MB
    input_types = ["random", "structured", "edge"]
    per_type_count = 50
    vectors = generate_test_vectors(input_sizes, input_types, per_type_count)

    all_results: List[Dict] = []
    for key, (spec, factory) in ALGORITHMS.items():
        all_results.append(measure_algorithm(spec, factory, vectors))

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics_fixed.json")
    with open(metrics_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    # Write textual summary
    summary_path = os.path.join(results_dir, "experiment_summary_fixed.txt")
    with open(summary_path, "w") as f:
        f.write("Corrected Cryptographic Hash Function Analysis - Summary\n")
        f.write("========================================================\n\n")
        for r in all_results:
            f.write(f"Algorithm: {r['algorithm']} (digest {r['digest_bits']} bits)\n")
            f.write(f"  Avg throughput: {r['throughput_avg_MBps']:.2f} MB/s\n")
            f.write(f"  Collision rate: {r['collision_rate']:.6f}\n")
            f.write(f"  Bit entropy:   {r['bit_entropy']:.3f}\n")
            av_mean = statistics.mean([c["avalanche_mean"] for c in r["by_config"]])
            f.write(f"  Avalanche mean: {av_mean:.3f}\n\n")

    # Plots (overwrite existing filenames used by the paper)
    plot_performance(all_results, os.path.join(fig_dir, "hash_analysis.png"), os.path.join(fig_dir, "hash_analysis.pdf"))
    plot_security(all_results, os.path.join(fig_dir, "security_analysis.png"), os.path.join(fig_dir, "security_analysis.pdf"))


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_dir = os.path.join(root, "results")
    run_and_save(out_dir)




