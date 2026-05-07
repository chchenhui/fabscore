"""Utility for saving and loading per-clip experiment results to JSON/CSV.
Supports fields: clip_id, transcription, is_hallucinated, p_no_speech,
reference, wer (per-utterance), and arbitrary extra fields.
"""

import csv
import json
from pathlib import Path
from typing import Optional


def save_results_json(results: list[dict], path: str | Path) -> None:
    """Save a list of per-clip result dicts to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results_json(path: str | Path) -> list[dict]:
    """Load per-clip results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_results_csv(results: list[dict], path: str | Path) -> None:
    """Save a list of per-clip result dicts to a CSV file."""
    if not results:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_results_csv(path: str | Path) -> list[dict]:
    """Load per-clip results from a CSV file."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_summary(summary: dict, path: str | Path) -> None:
    """Save an experiment summary dict (e.g. overall metrics) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def load_summary(path: str | Path) -> dict:
    """Load an experiment summary from JSON."""
    with open(path) as f:
        return json.load(f)
