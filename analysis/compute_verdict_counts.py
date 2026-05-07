from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


VERDICT_ORDER = [
    "Verified",
    "Insufficient Evidence",
    "No Code Files",
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
]

SUMMARY_KEY_MAP = {
    "Verified": "verified",
    "Insufficient Evidence": "insufficient_evidence",
    "No Code Files": "no_code_files",
    "Data Fabrication": "data_fabrication",
    "Experiment Fabrication": "experiment_fabrication",
    "Result Fabrication": "result_fabrication",
}


def load_verdict_counts(summary_path: Path) -> dict[str, int]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing or invalid 'summary' dict in {summary_path}")

    counts = {
        verdict: int(summary.get(summary_key, 0))
        for verdict, summary_key in SUMMARY_KEY_MAP.items()
    }
    return counts


def ordered_counts(counts: dict[str, int]) -> dict[str, int]:
    result = {verdict: int(counts.get(verdict, 0)) for verdict in VERDICT_ORDER}
    result["total_claims"] = int(sum(result.values()))
    return result


def infer_task_name(root_dir: Path, summary_path: Path) -> str:
    rel_parts = summary_path.relative_to(root_dir).parts

    if len(rel_parts) >= 3 and rel_parts[-2].startswith("fabscore_"):
        return rel_parts[-3]

    if rel_parts:
        return rel_parts[0]

    return summary_path.parent.name


def discover_fs_summary_paths(root_dir: Path, judge_type: str | None) -> Iterable[Path]:
    pattern = "**/fs_summary.json"
    for summary_path in sorted(root_dir.glob(pattern)):
        if judge_type is None:
            yield summary_path
            continue

        judge_dir = summary_path.parent.name
        if judge_dir == f"fabscore_{judge_type}":
            yield summary_path


def collect_counts(
    root_dir: Path,
    judge_type: str | None,
) -> tuple[dict[str, int], dict[str, dict[str, int]], int]:
    aggregate_counts = {verdict: 0 for verdict in VERDICT_ORDER}
    per_task_counts: dict[str, dict[str, int]] = {}
    summary_count = 0

    for summary_path in discover_fs_summary_paths(root_dir, judge_type):
        summary_count += 1
        task_name = infer_task_name(root_dir, summary_path)
        counts = load_verdict_counts(summary_path)

        for verdict in VERDICT_ORDER:
            aggregate_counts[verdict] += counts.get(verdict, 0)

        task_counts = per_task_counts.setdefault(
            task_name,
            {verdict: 0 for verdict in VERDICT_ORDER},
        )
        for verdict in VERDICT_ORDER:
            task_counts[verdict] += counts.get(verdict, 0)

    return aggregate_counts, per_task_counts, summary_count


def build_output(
    root_dir: Path,
    aggregate_counts: dict[str, int],
    per_task_counts: dict[str, dict[str, int]],
    summary_count: int,
    include_per_task: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "final": ordered_counts(aggregate_counts),
    }

    payload["meta_data"] = {
        "root_dir": str(root_dir),
        "task_count": len(per_task_counts),
        "fs_summary_count": summary_count,
    }

    if include_per_task:
        payload["per_task"] = {
            task_name: ordered_counts(counts)
            for task_name, counts in sorted(per_task_counts.items())
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count claim verdict categories from fs_summary.json files under a root directory."
    )
    parser.add_argument("root_dir", help="Root directory containing task folders.")
    parser.add_argument(
        "--judge-type",
        default="claude",
        help="Optional judge filter, e.g. claude or codex. Only counts fabscore_<judge_type>/fs_summary.json.",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Include a per-task breakdown in the JSON output.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    aggregate_counts, per_task_counts, summary_count = collect_counts(root_dir, args.judge_type)

    output = build_output(
        root_dir=root_dir,
        aggregate_counts=aggregate_counts,
        per_task_counts=per_task_counts,
        summary_count=summary_count,
        include_per_task=args.per_task,
    )

    print(json.dumps(output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()