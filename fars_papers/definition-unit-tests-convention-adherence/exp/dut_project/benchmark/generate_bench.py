"""Orchestrate benchmark generation across all convention families.

Generates ~300 items (100 per family), shuffles, and exports to JSONL.
Usage: python -m dut_project.benchmark.generate_bench [--output-dir data/]
"""

import argparse
import json
import os
import random
from pathlib import Path

from dut_project.benchmark.conventions.convolution import generate_items as gen_conv
from dut_project.benchmark.conventions.asymptotics import generate_items as gen_asym
from dut_project.benchmark.conventions.completeness import generate_items as gen_comp


def generate_all(items_per_family: int = 100, seed: int = 42) -> list[dict]:
    conv_items = gen_conv(items_per_family, seed=seed)
    asym_items = gen_asym(items_per_family, seed=seed + 1)
    comp_items = gen_comp(items_per_family, seed=seed + 2)

    all_items = conv_items + asym_items + comp_items

    for idx, item in enumerate(all_items):
        item["item_id"] = idx

    rng = random.Random(seed)
    rng.shuffle(all_items)

    for idx, item in enumerate(all_items):
        item["item_id"] = idx

    return all_items


def export_jsonl(items: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            serializable = {k: v for k, v in item.items() if k != "metadata"}
            serializable["metadata"] = {}
            for mk, mv in item.get("metadata", {}).items():
                if isinstance(mv, dict):
                    serializable["metadata"][mk] = {str(k2): v2 for k2, v2 in mv.items()}
                else:
                    serializable["metadata"][mk] = mv
            f.write(json.dumps(serializable) + "\n")


def export_metadata(items: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    family_counts = {}
    for item in items:
        fam = item["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    metadata = {
        "total_items": len(items),
        "family_counts": family_counts,
        "families": sorted(family_counts.keys()),
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate ErdosConventionsBench")
    parser.add_argument("--output-dir", default="dut_project/data", help="Output directory")
    parser.add_argument("--items-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = generate_all(args.items_per_family, args.seed)
    bench_path = os.path.join(args.output_dir, "erdos_conventions_bench.jsonl")
    meta_path = os.path.join(args.output_dir, "bench_metadata.json")

    export_jsonl(items, bench_path)
    export_metadata(items, meta_path)

    print(f"Generated {len(items)} items -> {bench_path}")
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
