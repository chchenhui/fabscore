from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


FABRICATION_LABELS = {
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute human confirmed fabrications, precision score, and label accuracy "
            "from FabScore human review CSV files."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        default=[
            "FabScore Human Eval - ClaudeCode.csv",
            "FabScore Human Eval - Codex.csv",
        ],
        help="One or more human review CSV files to analyze.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def get_human_final_verdict(row: dict[str, str]) -> str:
    agree = normalize_text(row.get("Do you agree? (Yes/No)"))
    machine_verdict = normalize_text(row.get("Verdict"))
    human_override = normalize_text(row.get("Your Verdict"))

    if agree.lower() == "yes":
        return machine_verdict
    if agree.lower() == "no":
        return human_override
    return ""


def format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def analyze_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_rows = len(rows)
    human_confirmed_count = 0
    label_match_count = 0
    fabrication_breakdown: Counter[str] = Counter()
    invalid_no_rows: list[int] = []

    for row_index, row in enumerate(rows, start=2):
        machine_verdict = normalize_text(row.get("Verdict"))
        human_final = get_human_final_verdict(row)
        agree = normalize_text(row.get("Do you agree? (Yes/No)"))
        human_override = normalize_text(row.get("Your Verdict"))

        if agree.lower() == "no" and not human_override:
            invalid_no_rows.append(row_index)

        if human_final in FABRICATION_LABELS:
            human_confirmed_count += 1
            fabrication_breakdown[human_final] += 1
            if human_final == machine_verdict:
                label_match_count += 1

    if invalid_no_rows:
        raise ValueError(
            "Found rows with 'Do you agree? (Yes/No)' = 'No' but empty 'Your Verdict' "
            f"in {path}: lines {', '.join(str(line) for line in invalid_no_rows)}"
        )

    return {
        "path": path,
        "total_rows": total_rows,
        "human_confirmed_count": human_confirmed_count,
        "fabrication_breakdown": fabrication_breakdown,
        "precision_percent": format_percent(human_confirmed_count, total_rows),
        "label_match_count": label_match_count,
        "label_accuracy_percent": format_percent(label_match_count, human_confirmed_count),
    }


def print_report(result: dict[str, object]) -> None:
    path = result["path"]
    total_rows = result["total_rows"]
    human_confirmed_count = result["human_confirmed_count"]
    fabrication_breakdown = result["fabrication_breakdown"]
    label_match_count = result["label_match_count"]

    print(f"File: {path}")
    print(f"  Total rows: {total_rows}")
    print(f"  Human confirmed fabrications: {human_confirmed_count}")
    print("  Breakdown:")
    for label in sorted(FABRICATION_LABELS):
        print(f"    {label}: {fabrication_breakdown.get(label, 0)}")
    print(
        "  Precision score: "
        f"{human_confirmed_count}/{total_rows} ({result['precision_percent']})"
    )
    print(
        "  Label accuracy: "
        f"{label_match_count}/{human_confirmed_count} ({result['label_accuracy_percent']})"
    )
    print()


def main() -> None:
    args = parse_args()
    results = []

    for csv_path in args.csv_files:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        results.append(analyze_csv(path))

    for result in results:
        print_report(result)


if __name__ == "__main__":
    main()