## Name

resource_centric_ppm_agents

## Title

Towards Resource-Centric Predictive Process Monitoring for Concurrent Business Processes

## Short Hypothesis

We contribute four elements:
C1 — Reproducible protocol. A leakage-safe, deterministic PPM pipeline (chronological case splits; train-only normalization; fixed seeds; artifact logging for figures/metrics).
C2 — Strong transparent baseline. A compact LSTM next-activity baseline on three public logs (BPI 2012, BPI 2017, Road Traffic) with ready-to-reuse splits and scripts.
C3 — Simulator blueprint & metrics. A complete, modular design for a resource-centric agent (per-resource policies + DES), with evaluation measures (global next event, workload MAPE, per-resource next-task precision) and ablations (FIFO vs learned).
C4 — Pitfalls & checklists. Concrete failure modes (duration pairing under partial lifecycles; policy back-off under imbalance) and a checklist for future resource-aware studies.

## Related Work

Case-centric PPM ignores shared-resource competition. We instead learn per-resource behavior and simulate concurrency.

## Abstract

We present a reproducible evaluation protocol for resource-aware predictive process monitoring (PPM), alongside a compact, transparent case-centric LSTM baseline evaluated on BPI 2012, BPI 2017, and Road Traffic logs with chronological splits. Our protocol fixes leakage controls, train-only normalization, figure and metric exports, and seeds for determinism. We also provide a fully specified but modular blueprint for a resource-centric agent (per-resource multinomial policies + lightweight discrete-event simulator), including metrics for global next event and resource-level workload—even though full end-to-end simulator results are not reported in this version. Baseline next-activity results are strong (Top-3 0.987–0.994; Top-1 0.757–0.833) and expose systematic confusions that motivate resource-aware context. We release code, splits, and plot artifacts to enable one-click replication and future comparisons. This paper is intended as a protocol + baseline + pitfalls report to accelerate trustworthy experiments on resource-aware PPM, rather than a claim of state-of-the-art accuracy.

## Experiments

- Data (offline): first load BPI_Challenge_2012.xes in input data folder and then BPI_Challenge_2017.xes; parse with PM4Py. Discover any of {BPI2012, BPI2017, Road Traffic}. Attributes: activity, timestamp, resource, lifecycle.
- Preprocess: map lifecycle to start/complete; pair start-complete for durations; if only complete, treat as instantaneous.
- Split: chronological 70/10/20 train/val/test by case start.
- State (replay): track each case's next activity; resource busy/idle; queue counts and oldest-wait per activity.
- Policies: per-resource multinomial logistic regression predicting next activity from {prev activity ID, 4-bin time-of-day, queue counts/oldest-wait}. Back off to a global model if few samples.
- Durations: log-normal per activity; fallback median if sparse.
- Simulator: when a resource frees, sample its next activity via policy; pick FIFO case among those needing it; sample duration; advance to next completion.
- Rollouts: N=30 Monte Carlo trajectories per prefix.
- Baseline: 1-layer LSTM (hidden≈64) on case sequences (activity ID, resource ID, Δt bin); iterative decoding for suffix. Train 5 epochs for a first pass.
- Metrics: next-activity Top-1/Top-3, suffix normalized Damerau-Levenshtein similarity, remaining-time MAE, global next-event accuracy, per-resource next-task precision & workload MAPE.
- Ablation: FIFO policy (no learned policy) with identical simulator, resource always selects the oldest-waiting eligible activity. Recompute metrics.

## Risk Factors And Limitations

- Hidden priorities/case attributes not modeled may limit fidelity.
- Replay-based per-case next-activity may miss complex branching.
- Long-horizon simulations can drift; small N keeps runtime low at the cost of variance.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
# ai_scientist/ideas/my_research_topic.py
# -----------------------------------------------------------------------------
# Robust XES discovery & loading for AI-Scientist/BPM-Scientist
# -----------------------------------------------------------------------------
# HOW THIS WORKS:
# - local input data under ./input. Therefore your logs are visible to the agent as: input/*.xes
# - This module ALWAYS prefers Path("input") (and CWD/input), then tries ./data
# - Supports .xes and .xes.gz and loads any of: BPI_Challenge_2012, BPI_Challenge_2017, Road_Traffic_Fine_Management_Process.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ---------- helpers: discovery ----------

def _has_xes(dirpath: Path) -> bool:
    """True if directory contains any .xes or .xes.gz files."""
    try:
        return dirpath.is_dir() and (any(dirpath.glob("*.xes")) or any(dirpath.glob("*.xes.gz")))
    except Exception:
        return False

def _resolve_data_dir() -> Path:
    """
    Resolution order (first hit wins):
      1) ./input (workspace copy of data) and CWD/input
      2) ./data and parent-walk fallbacks (also check parent/input)
      3) Common absolute fallbacks under /workspace
    """
    candidates: List[Path] = []

    # 1) workspace-local input/
    candidates += [Path("input").resolve(), (Path.cwd() / "input").resolve()]

    # 2) data/ in CWD and parent-walk (also try input/ in parents)
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidates.append((base / "data").resolve())
        candidates.append((base / "input").resolve())

    # 3) common absolute places in Sakana-style layouts
    candidates += [
        Path("/workspace/input"),
        Path("/workspace/data"),
        Path("/workspace/ai_scientist/data"),
        Path("/workspace/AI-Scientist-v2/data"),
        Path("/workspace/experiments/data"),
        Path("/workspace/ai_scientist/input"),
        Path("/workspace/experiments/input"),
    ]

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if _has_xes(p):
            print(f"[data] Using discovered data dir: {p}")
            return p

    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate a directory containing .xes files.\n"
        f"Checked:{tried}\n"
        "Tips:\n"
        "  • Ensure filenames include BPI 2012/2017 or 'Road_Traffic_Fine_Management_Process' for auto-match."
    )

def _first_match(d: Path, patterns: List[str]) -> Optional[Path]:
    """
    Return the first existing file in d matching any pattern (supports globs).
    Patterns like '*.xes*' allow both .xes and .xes.gz.
    """
    for pat in patterns:
        for p in d.glob(pat):
            if p.is_file():
                return p
    return None

# ---------- XES loading ----------

def xes_to_df(xes_path: Path) -> pd.DataFrame:
    """Load a .xes(.gz) with pm4py and return a tidy DataFrame."""
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except Exception as e:
        raise ImportError("pm4py is required. Install inside your venv: `pip install pm4py`.") from e

    print(f"[data] Loading XES: {xes_path}")
    log = xes_importer.apply(str(xes_path))
    rows = []
    for tr in log:
        case_id = tr.attributes.get("concept:name") or tr.attributes.get("case:concept:name")
        for e in tr:
            rows.append({
                "case_id": case_id,
                "activity": e.get("concept:name"),
                "lifecycle": e.get("lifecycle:transition", "complete"),
                "timestamp": e.get("time:timestamp"),
                "resource": e.get("org:resource", "System"),
            })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df[["case_id", "activity", "lifecycle", "timestamp", "resource"]]
    df = df.sort_values(["timestamp", "case_id"]).reset_index(drop=True)
    return df

# ---------- public API used by the agent ----------

def load_datasets() -> Dict[str, pd.DataFrame]:
    """
    Returns any subset of {'BPI2012','BPI2017','ROAD'} that exist.
    Does NOT fail if one is missing; it loads what it finds and prints diagnostics.
    """
    data_dir = _resolve_data_dir()
    available = sorted([p.name for p in list(data_dir.glob("*.xes")) + list(data_dir.glob("*.xes.gz"))])
    print(f"[data] Available in {data_dir}: {available}")

    # Patterns accept both strict and fuzzy names; '*.xes*' allows .xes or .xes.gz.
    patterns = {
        "BPI2012": ["BPI_Challenge_2012*.xes*", "BPI2012*.xes*", "*2012*.xes*"],
        "BPI2017": ["BPI_Challenge_2017*.xes*", "BPI2017*.xes*", "*2017*.xes*"],
        "ROAD":    ["Road_Traffic_Fine_Management_Process*.xes*", "*Traffic*Fine*.xes*", "*Traffic*.xes*"],
    }

    loaded: Dict[str, pd.DataFrame] = {}
    for key, pats in patterns.items():
        path = _first_match(data_dir, pats)
        if path is not None:
            try:
                loaded[key] = xes_to_df(path)
            except Exception as e:
                print(f"[warn] Failed to load {key} from {path}: {e}")
        else:
            print(f"[data] Not found for {key} (patterns {pats})")

    if not loaded:
        raise FileNotFoundError(
            f"No known XES files found in {data_dir}. "
            f"Found: {available}"
        )

    print(f"[data] Loaded datasets: {list(loaded.keys())}")
    return loaded

def pick_default_dataset(datasets: Dict[str, pd.DataFrame],
                         order: Tuple[str, ...] = ("BPI2017", "BPI2012", "ROAD")) -> Tuple[str, pd.DataFrame]:
    """Pick a default dataset by preference order."""
    for name in order:
        if name in datasets:
            return name, datasets[name]
    # Fallback: first any
    name = next(iter(datasets.keys()))
    return name, datasets[name]

# Optional smoke test when run standalone
if __name__ == "__main__":
    ds = load_datasets()
    name, df = pick_default_dataset(ds)
    print(f"[data] Using dataset: {name}, shape={df.shape}")
    print(df.head())

```

