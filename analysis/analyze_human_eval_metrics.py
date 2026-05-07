from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HALLUCINATION_LABELS = {
    "data fabrication",
    "experiment fabrication",
    "result fabrication",
}

FABRICATED_VERDICTS = {
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
}

SUMMARY_SECTIONS = ("tables", "figures", "results_section")

VERDICT_ORDER = (
    "Verified",
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
    "No Code Files",
    "Insufficient Evidence",
    "Error",
)

NON_HALLUCINATION_LABELS = {
    "verified",
    "insufficient evidence",
    "no code files",
}


@dataclass
class Metrics:
    total_claims: int = 0
    detected_hallucinations: int = 0
    human_confirmed_hallucinations: int = 0
    correct_labels_within_confirmed: int = 0
    confirmed_by_direct_agreement: int = 0
    confirmed_by_relabelled_hallucination: int = 0
    rejected_as_non_hallucination: int = 0

    @property
    def precision_score(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.human_confirmed_hallucinations / self.total_claims

    @property
    def detection_precision(self) -> float:
        if self.detected_hallucinations == 0:
            return 0.0
        return self.human_confirmed_hallucinations / self.detected_hallucinations

    @property
    def label_accuracy(self) -> float:
        if self.human_confirmed_hallucinations == 0:
            return 0.0
        return self.correct_labels_within_confirmed / self.human_confirmed_hallucinations


def normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def is_detected_hallucination(label: str) -> bool:
    return normalize_label(label) in HALLUCINATION_LABELS


def is_human_confirmed_hallucination(agree: str, reviewer_label: str) -> bool:
    normalized_agree = normalize_label(agree)
    normalized_reviewer_label = normalize_label(reviewer_label)

    if normalized_agree == "yes":
        return True

    if normalized_agree == "no" and normalized_reviewer_label in HALLUCINATION_LABELS:
        return True

    if normalized_agree == "no" and normalized_reviewer_label in NON_HALLUCINATION_LABELS:
        return False

    return False


def compute_metrics(csv_path: Path) -> Metrics:
    metrics = Metrics()

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics.total_claims += 1

            detected_label = row.get("Verdict", "")
            if not is_detected_hallucination(detected_label):
                continue

            metrics.detected_hallucinations += 1

            agree = row.get("Do you agree? (Yes/No)", "")
            reviewer_label = row.get("Your Verdict", "")
            normalized_agree = normalize_label(agree)
            normalized_reviewer_label = normalize_label(reviewer_label)

            if is_human_confirmed_hallucination(agree, reviewer_label):
                metrics.human_confirmed_hallucinations += 1
                if normalized_agree == "yes":
                    metrics.correct_labels_within_confirmed += 1
                    metrics.confirmed_by_direct_agreement += 1
                else:
                    metrics.confirmed_by_relabelled_hallucination += 1
            elif normalized_agree == "no" and normalized_reviewer_label in NON_HALLUCINATION_LABELS:
                metrics.rejected_as_non_hallucination += 1

    return metrics


def format_percentage(value: float) -> str:
    return f"{value:.2%}"


def render_report(csv_path: Path, metrics: Metrics) -> str:
    return "\n".join(
        [
            f"File: {csv_path}",
            f"  Total claims: {metrics.total_claims}",
            f"  Detected hallucinations (fabrications or no code files): {metrics.detected_hallucinations}",
            f"  Human confirmed hallucinations: {metrics.human_confirmed_hallucinations}",
            f"    Confirmed by agree=yes: {metrics.confirmed_by_direct_agreement}",
            f"    Confirmed by agree=no but reviewer kept a hallucination label: {metrics.confirmed_by_relabelled_hallucination}",
            f"    Rejected as verified/insufficient evidence: {metrics.rejected_as_non_hallucination}",
            f"  Precision score (confirmed / total claims): {format_percentage(metrics.precision_score)}",
            f"  Detection precision (confirmed / detected hallucinations): {format_percentage(metrics.detection_precision)}",
            f"  Label accuracy (agree=yes among confirmed hallucinations): {format_percentage(metrics.label_accuracy)}",
        ]
    )


def combine_metrics(items: Iterable[Metrics]) -> Metrics:
    combined = Metrics()
    for item in items:
        combined.total_claims += item.total_claims
        combined.detected_hallucinations += item.detected_hallucinations
        combined.human_confirmed_hallucinations += item.human_confirmed_hallucinations
        combined.correct_labels_within_confirmed += item.correct_labels_within_confirmed
        combined.confirmed_by_direct_agreement += item.confirmed_by_direct_agreement
        combined.confirmed_by_relabelled_hallucination += item.confirmed_by_relabelled_hallucination
        combined.rejected_as_non_hallucination += item.rejected_as_non_hallucination
    return combined


def resolve_summary_input(path: Path) -> Path:
    if path.is_dir():
        for name in ("fs_summary.json", "fs_summary_fabrications.json", "fs_summary_fabricated.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No summary JSON found under {path}")

    if path.exists():
        return path

    raise FileNotFoundError(path)


def build_filtered_summary(summary_data: dict) -> dict:
    output = dict(summary_data)
    filtered_sections: dict[str, list[dict]] = {}
    filtered_claims: list[dict] = []

    for section in SUMMARY_SECTIONS:
        items = summary_data.get(section, [])
        if not isinstance(items, list):
            raise TypeError(f"Expected list for section '{section}', got {type(items).__name__}")

        filtered = [item for item in items if isinstance(item, dict) and item.get("verdict") in FABRICATED_VERDICTS]
        filtered_sections[section] = filtered
        filtered_claims.extend(filtered)

    for section, filtered in filtered_sections.items():
        output[section] = filtered

    verdict_breakdown = {verdict: 0 for verdict in VERDICT_ORDER}
    verdict_breakdown["Other"] = 0
    for item in filtered_claims:
        verdict = item.get("verdict")
        if verdict in verdict_breakdown:
            verdict_breakdown[verdict] += 1
        else:
            verdict_breakdown["Other"] += 1

    output["filtered_summary"] = {
        "total_claims": len(filtered_claims),
        "tables": len(filtered_sections["tables"]),
        "figures": len(filtered_sections["figures"]),
        "results_section": len(filtered_sections["results_section"]),
        "verdict_breakdown": verdict_breakdown,
    }
    return output


def generate_fabricated_summary(path: Path) -> tuple[Path, Path, int]:
    source_path = resolve_summary_input(path)
    with source_path.open(encoding="utf-8") as handle:
        summary_data = json.load(handle)

    if not isinstance(summary_data, dict):
        raise TypeError(f"Expected object at top level in {source_path}, got {type(summary_data).__name__}")

    fabricated_summary = build_filtered_summary(summary_data)
    output_path = source_path.with_name("fs_summary_fabricated.json")
    output_path.write_text(json.dumps(fabricated_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return source_path, output_path, fabricated_summary["filtered_summary"]["total_claims"]


def parse_generate_paths(task_roots: Iterable[Path], judges: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for task_root in task_roots:
        for judge in judges:
            paths.append(task_root / judge)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute human-evaluation metrics for FabScore hallucination review CSV files."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="CSV files to analyze, or summary directories/files to generate fabricated summaries from.")
    parser.add_argument(
        "--generate-fabricated",
        action="store_true",
        help="Generate fs_summary_fabricated.json from task summary files instead of analyzing CSV metrics.",
    )
    parser.add_argument(
        "--task-root",
        dest="task_roots",
        action="append",
        type=Path,
        default=[],
        help="Task root directory. When used with --generate-fabricated, the script will look under each task root for the requested judge directories.",
    )
    parser.add_argument(
        "--judge-dir",
        dest="judge_dirs",
        action="append",
        default=[],
        help="Judge directory name under each task root, for example fabscore_claude or fabscore_codex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate_fabricated:
        target_paths = list(args.paths)
        if args.task_roots:
            judge_dirs = args.judge_dirs or ["fabscore_claude", "fabscore_codex"]
            target_paths.extend(parse_generate_paths(args.task_roots, judge_dirs))
        if not target_paths:
            raise SystemExit("No summary paths provided. Pass directories/files as positional args or use --task-root.")

        reports: list[str] = []
        for target_path in target_paths:
            source_path, output_path, claim_count = generate_fabricated_summary(target_path)
            reports.append(f"Generated {output_path} from {source_path} ({claim_count} fabricated claims)")
        print("\n".join(reports))
        return

    if not args.paths:
        raise SystemExit("No CSV paths provided.")

    reports: list[str] = []
    all_metrics: list[Metrics] = []

    for csv_path in args.paths:
        metrics = compute_metrics(csv_path)
        all_metrics.append(metrics)
        reports.append(render_report(csv_path, metrics))

    if len(all_metrics) > 1:
        reports.append(render_report(Path("ALL_FILES"), combine_metrics(all_metrics)))

    print("\n\n".join(reports))


if __name__ == "__main__":
    main()