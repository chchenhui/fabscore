"""Validate generated ErdosConventionsBench items.

Checks: (1) ground_truth_glossary != ground_truth_alternate for every item,
(2) discriminative checks differ under both conventions,
(3) neutral checks agree under both conventions,
(4) balanced family distribution (~100 per family).

Usage: python -m dut_project.benchmark.validate_bench [--input data/erdos_conventions_bench.jsonl]
"""

import argparse
import json
import sys


def load_bench(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def validate(items: list[dict]) -> tuple[bool, list[str]]:
    errors = []
    family_counts: dict[str, int] = {}

    for i, item in enumerate(items):
        fam = item.get("family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1

        if item["ground_truth_glossary"] == item["ground_truth_alternate"]:
            errors.append(f"Item {i} ({fam}): main ground truths are identical")

        for j, dc in enumerate(item.get("discriminative_checks", [])):
            if dc["answer_glossary"] == dc["answer_alternate"]:
                errors.append(
                    f"Item {i} ({fam}): discriminative_check[{j}] has identical answers"
                )

        for j, nc in enumerate(item.get("neutral_checks", [])):
            if nc["answer_glossary"] != nc["answer_alternate"]:
                errors.append(
                    f"Item {i} ({fam}): neutral_check[{j}] has differing answers"
                )

        if len(item.get("discriminative_checks", [])) != 3:
            errors.append(
                f"Item {i} ({fam}): expected 3 discriminative checks, got "
                f"{len(item.get('discriminative_checks', []))}"
            )
        if len(item.get("neutral_checks", [])) != 3:
            errors.append(
                f"Item {i} ({fam}): expected 3 neutral checks, got "
                f"{len(item.get('neutral_checks', []))}"
            )

    print(f"Total items: {len(items)}")
    print(f"Family distribution: {family_counts}")

    for fam, count in family_counts.items():
        if count < 50:
            errors.append(f"Family '{fam}' has only {count} items (expected ~100)")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate ErdosConventionsBench")
    parser.add_argument(
        "--input",
        default="dut_project/data/erdos_conventions_bench.jsonl",
        help="Path to benchmark JSONL",
    )
    args = parser.parse_args()

    items = load_bench(args.input)
    passed, errors = validate(items)

    if passed:
        print("VALIDATION PASSED: All checks OK")
    else:
        print(f"VALIDATION FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
