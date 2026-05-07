import argparse
import csv
import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Optional, Tuple


CLAIM_BUCKETS = ("tables", "figures", "results_section")
CSV_HEADERS = [
    "Claim ID",
    "Claim",
    "Task Name",
    "Verdict",
    "Do you agree? (Yes/No)",
    "Your Verdict",
]


def _iter_summary_paths(
    root: str,
    summary_filename: str,
    summary_dir_name: Optional[str] = None,
) -> Iterable[str]:
    for current_root, _, files in os.walk(root):
        if summary_filename in files:
            path = osp.join(current_root, summary_filename)
            if summary_dir_name and osp.basename(osp.dirname(path)) != summary_dir_name:
                continue
            yield path


def _task_name_from_path(path: str) -> str:
    parts = osp.normpath(path).split(os.sep)
    if "aiscientist_papers" in parts:
        idx = parts.index("aiscientist_papers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return osp.basename(osp.dirname(osp.dirname(path)))


def _claim_sort_key(row: Dict[str, str]) -> Tuple[str, int, str]:
    claim_id_raw = row.get("Claim ID", "")
    try:
        claim_id = int(claim_id_raw)
    except (TypeError, ValueError):
        claim_id = 10**9
    task_name = row.get("Task Name", "")
    task_group = 1 if task_name.startswith("submission_") else 0
    return (task_group, task_name, claim_id, row.get("Claim", ""))


def _row_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row.get("Task Name", ""), row.get("Claim ID", ""))


def _load_existing_annotations(csv_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not osp.exists(csv_path):
        return {}

    annotations: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations[_row_key(row)] = {
                "Do you agree? (Yes/No)": row.get("Do you agree? (Yes/No)", ""),
                "Your Verdict": row.get("Your Verdict", ""),
            }
    return annotations


def _collect_rows(
    root: str,
    summary_filename: str,
    summary_dir_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(_iter_summary_paths(root, summary_filename, summary_dir_name)):
        task_name = _task_name_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for bucket in CLAIM_BUCKETS:
            for item in data.get(bucket, []):
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "Claim ID": str(item.get("claim_index", "")),
                        "Claim": str(item.get("claim", "")),
                        "Task Name": task_name,
                        "Verdict": str(item.get("verdict", "")),
                        "Do you agree? (Yes/No)": "",
                        "Your Verdict": "",
                    }
                )
    rows.sort(key=_claim_sort_key)
    return rows


def _merge_annotations(
    rows: List[Dict[str, str]],
    annotations: Dict[Tuple[str, str], Dict[str, str]],
) -> List[Dict[str, str]]:
    for row in rows:
        existing = annotations.get(_row_key(row))
        if not existing:
            continue
        row["Do you agree? (Yes/No)"] = existing.get("Do you agree? (Yes/No)", "")
        row["Your Verdict"] = existing.get("Your Verdict", "")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync FabScore Human Eval CSV from filtered fs_summary JSON files."
    )
    parser.add_argument(
        "--root",
        default="aiscientist_papers",
        help="Root directory to scan for filtered fs_summary JSON files.",
    )
    parser.add_argument(
        "--csv",
        default="human_reviews/FabScore Human Eval.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--summary-filename",
        default="fs_summary_fabricated.json",
        help="Filtered summary JSON filename to collect from each task directory.",
    )
    parser.add_argument(
        "--annotations-csv",
        default=None,
        help="Optional existing CSV file from which to copy annotation columns.",
    )
    parser.add_argument(
        "--summary-dir-name",
        default=None,
        help="Only scan summary files whose parent directory matches this name, e.g. fabscore_codex.",
    )
    parser.add_argument(
        "--no-preserve-annotations",
        action="store_true",
        help="Do not reuse existing annotation columns from any CSV source.",
    )
    args = parser.parse_args()

    root = osp.abspath(args.root)
    csv_path = osp.abspath(args.csv)
    annotations_csv_path: Optional[str] = (
        osp.abspath(args.annotations_csv) if args.annotations_csv else None
    )

    if not osp.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")

    rows = _collect_rows(root, args.summary_filename, args.summary_dir_name)
    annotations = {}
    if not args.no_preserve_annotations:
        annotations_source = annotations_csv_path or csv_path
        annotations = _load_existing_annotations(annotations_source)
    rows = _merge_annotations(rows, annotations)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"WROTE {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
