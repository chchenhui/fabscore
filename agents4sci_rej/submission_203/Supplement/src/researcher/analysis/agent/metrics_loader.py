from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping


def locate_metric_files(metrics_dir: str, metrics: List[str], scene_catalog: Mapping[str, Any]) -> Dict[str, List[str]]:
    """
    Discover metric files under metrics_dir for the requested metric names.

    Returns a mapping metric_name -> list of file paths.
    We use a simple heuristic: gather all files under metrics_dir whose filename
    contains the metric name and ends with .json. scene_catalog may be used by
    upstream code; we keep it for API compatibility but do not consume it here.
    """
    result: Dict[str, List[str]] = {m: [] for m in metrics}
    for root, _dirs, files in os.walk(metrics_dir):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            for m in metrics:
                if m in fname:
                    result[m].append(fpath)
    return result


def load_and_normalize_metrics(files_map: Mapping[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Load rows from each metric's JSON files and normalize into a flat list of dicts.
    This is a minimal, generic loader that expects each file to be either:
    - list[dict], or
    - dict with a top-level key "rows" listing rows.

    Adds a field `metric` with the metric name for downstream grouping/filters.
    """
    rows: List[Dict[str, Any]] = []
    for metric, paths in files_map.items():
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for r in data:
                        if isinstance(r, dict):
                            nr = dict(r)
                            nr.setdefault("metric", metric)
                            rows.append(nr)
                elif isinstance(data, dict) and isinstance(data.get("rows"), list):
                    for r in data["rows"]:
                        if isinstance(r, dict):
                            nr = dict(r)
                            nr.setdefault("metric", metric)
                            rows.append(nr)
                else:
                    # Fallback: wrap entire object
                    rows.append({"metric": metric, "value": data})
            except Exception as e:  # pragma: no cover - surface error context to callers
                rows.append({"metric": metric, "load_error": str(e), "file": p})
    return rows


__all__ = [
    "locate_metric_files",
    "load_and_normalize_metrics",
]



