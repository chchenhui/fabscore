import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .explainer_agent import ExplainerAgent
except Exception:
    # Fallback absolute import when running as a script
    from src.researcher.analysis.agent.explainer_agent import ExplainerAgent  # type: ignore


Number = float


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> Optional[Number]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _extract_step(value: Any) -> int:
    # Accept int or strings like "step_10" / "10"
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        m = re.search(r"\d+", str(value))
        if m:
            return int(m.group(0))
    except Exception:
        pass
    return 0


def _build_series_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, Number]]]:
    """Build per-group time series: {group: [(step, y), ...]}.

    Expects rows like {"group_name": str, "step": int/str, "data": number or {"value": number}}.
    Ignores rows lacking numeric y.
    """
    group_to_points: Dict[str, List[Tuple[int, Number]]] = {}
    for row in records:
        group = (
            row.get("group_name")
            or row.get("group")
            or row.get("experiment_group")
            or "group"
        )
        step = _extract_step(row.get("step"))
        y: Optional[Number] = None
        val = row.get("data")
        if isinstance(val, (int, float)):
            y = float(val)
        elif isinstance(val, dict):
            # common alternatives
            for k in ["value", "y", "score", "mean", "avg", "v"]:
                v = val.get(k)
                y = _safe_float(v)
                if y is not None:
                    break
        else:
            y = _safe_float(val)
        if y is None:
            continue
        group_to_points.setdefault(str(group), []).append((step, y))
    # sort by step
    for g, pts in group_to_points.items():
        pts.sort(key=lambda t: t[0])
    return group_to_points


def _highlight_final(values_by_group: Dict[str, List[Tuple[int, Number]]]) -> Dict[str, Any]:
    """Compute final-step highlights across groups."""
    finals: Dict[str, Number] = {}
    for g, pts in values_by_group.items():
        if not pts:
            continue
        finals[g] = pts[-1][1]
    if not finals:
        return {"note": "no_numeric_data"}
    max_group = max(finals, key=lambda k: finals[k])
    min_group = min(finals, key=lambda k: finals[k])
    max_v = finals[max_group]
    min_v = finals[min_group]
    return {
        "aggregation": "final",
        "groups": len(finals),
        "max": max_v,
        "max_group": max_group,
        "min": min_v,
        "min_group": min_group,
        "range": max_v - min_v,
    }


def _highlight_trajectory(values_by_group: Dict[str, List[Tuple[int, Number]]]) -> Dict[str, Any]:
    """Compute trajectory highlights: start->end delta averaged across groups and direction counts."""
    if not values_by_group:
        return {"note": "no_numeric_data"}
    deltas: List[Number] = []
    starts: List[Number] = []
    ends: List[Number] = []
    for pts in values_by_group.values():
        if len(pts) == 0:
            continue
        start = pts[0][1]
        end = pts[-1][1]
        starts.append(start)
        ends.append(end)
        deltas.append(end - start)
    if not deltas:
        return {"note": "no_numeric_data"}
    mean_delta = sum(deltas) / len(deltas)
    num_down = sum(1 for d in deltas if d < 0)
    num_up = sum(1 for d in deltas if d > 0)
    return {
        "aggregation": "trajectory_full_range",
        "groups": len(deltas),
        "mean_delta": mean_delta,
        "start_mean": (sum(starts) / len(starts)) if starts else None,
        "end_mean": (sum(ends) / len(ends)) if ends else None,
        "num_groups_decreasing": num_down,
        "num_groups_increasing": num_up,
    }


def _compose_numeric_sentence(high: Dict[str, Any], fig_kind: str) -> str:
    try:
        if fig_kind == "final" and high.get("aggregation") == "final":
            return (
                f"终值范围 [{high['min']:.4g}, {high['max']:.4g}] (Δ={high['range']:.4g}); "
                f"最高组={high['max_group']}, 最低组={high['min_group']}"
            )
        if fig_kind == "trajectory" and high.get("aggregation") == "trajectory_full_range":
            parts = []
            if high.get("start_mean") is not None and high.get("end_mean") is not None:
                parts.append(
                    f"均值由 {high['start_mean']:.4g} → {high['end_mean']:.4g}"
                )
            parts.append(f"平均Δ={high['mean_delta']:.4g}")
            parts.append(
                f"↓组数={high['num_groups_decreasing']}, ↑组数={high['num_groups_increasing']}"
            )
            return "; ".join(parts)
    except Exception:
        pass
    return "(无可用数值摘要)"


def _load_fig_spec(figures_dir: Path, idx: int) -> Optional[Dict[str, Any]]:
    path = figures_dir / f"fig{idx}_spec_used.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _resolve_processed_path(spec: Dict[str, Any], default_processed_dir: Path) -> Optional[Path]:
    cand = spec.get("_resolved_data_path") or spec.get("resolved_data_path")
    if cand:
        return Path(str(cand)).absolute()
    src = spec.get("source_reference")
    if isinstance(src, str):
        return (spec.get("processed_dir") and Path(spec["processed_dir"])) and Path(spec["processed_dir"]) / src or default_processed_dir / src
    return None


def _compute_highlights_for_spec(spec: Dict[str, Any], data_records: Any) -> Tuple[Dict[str, Any], str, str]:
    # Determine fig kind from spec title/id
    title = str(spec.get("title") or spec.get("id") or "Figure").lower()
    fig_kind = "final" if ("final" in title or spec.get("suggested_visualization_type") == "bar") else "trajectory"
    # Unwrap data to list of records
    records: List[Dict[str, Any]]
    if isinstance(data_records, list):
        records = data_records
    elif isinstance(data_records, dict):
        # Common containers in processed outputs
        for key in ["data", "records", "rows"]:
            if isinstance(data_records.get(key), list):
                records = data_records.get(key)  # type: ignore[assignment]
                break
        else:
            records = []
    else:
        records = []

    # Build per-group time series
    series = _build_series_from_records(records)
    # Compute highlights
    high = _highlight_final(series) if fig_kind == "final" else _highlight_trajectory(series)
    # Compose numeric sentence
    numeric_sentence = _compose_numeric_sentence(high, fig_kind)
    aggregation_policy = "final" if fig_kind == "final" else "trajectory_full_range"
    return high, numeric_sentence, aggregation_policy


def _ensure_numeric_in_text(text: str, fallback_sentence: str) -> str:
    if re.search(r"[-+]?\d+(?:\.\d+)?", text or ""):
        return text
    text = (text or "").strip()
    if text and not text.endswith("。") and not text.endswith("."):
        text += "。"
    return (text + " " + fallback_sentence).strip()


def _enforce_consistency(text: str, aggregation_policy: str) -> str:
    # Replace mentions of last-k/first-50 when aggregation is final, and vice versa avoid adding such claims
    if aggregation_policy == "final":
        text = re.sub(r"(last\s*\d+|最后\s*\d+|前\s*\d+|first\s*\d+)", "终值", text, flags=re.IGNORECASE)
        text = re.sub(r"(early window|early|初期|前期)", "全程", text, flags=re.IGNORECASE)
    # For trajectory, avoid claiming "终值" as the sole basis
    if aggregation_policy == "trajectory_full_range":
        text = re.sub(r"(仅基于终值|基于终值)", "基于完整轨迹", text)
    return text


def run_stage2(stage1_context: Dict[str, Any], paths: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Stage-2 Orchestrator

    - Build figure summaries from Stage-1 specs and processed data (no image pixels)
    - Call ExplainerAgent to generate structured analysis
    - Enforce per-figure numeric highlight and aggregation consistency
    - Persist to figure_analysis.json under the figures directory
    """
    cfg = cfg or {}

    project_name = (
        stage1_context.get("project_name")
        or stage1_context.get("project")
        or paths.get("project_name")
        or ""
    )
    figures_dir = Path(str(paths.get("figures_dir") or "")).absolute() if paths.get("figures_dir") else None
    processed_dir = Path(str(paths.get("data_processed_dir") or "")).absolute() if paths.get("data_processed_dir") else None

    # Fallback path resolution from conventional project structure
    if not figures_dir or not figures_dir.exists():
        if project_name:
            figures_dir = Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/analysis/data/processed/figures").absolute()
        else:
            raise FileNotFoundError("figures_dir not provided and project_name missing for default resolution")
    if not processed_dir or not processed_dir.exists():
        if project_name:
            processed_dir = Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/analysis/data/processed").absolute()
        else:
            raise FileNotFoundError("data_processed_dir not provided and project_name missing for default resolution")
    import time
    log_file = figures_dir / "stage2.log"
    def _log(level: str, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cleaned = str(msg).rstrip("\n")
        except Exception:
            cleaned = str(msg)
        line = f"{ts} | {level:<8} | stage2 - {cleaned}"
        try:
            print(line)
        except Exception:
            pass
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    _log("INFO", f"Stage-2 start: project_name={project_name}")
    _log("INFO", f"paths: figures_dir={figures_dir}, processed_dir={processed_dir}")

    # Collect specs
    specs: List[Tuple[int, Dict[str, Any]]] = []
    for i in [1, 2, 3]:
        spec = _load_fig_spec(figures_dir, i)
        if spec:
            specs.append((i, spec))
    _log("INFO", f"Found {len(specs)} figure specs")

    fig_summaries: List[Dict[str, Any]] = []
    numeric_highlights_by_id: Dict[str, Dict[str, Any]] = {}
    policies_by_idx: Dict[int, str] = {}

    for idx, spec in specs:
        fig_path = (figures_dir / f"fig{idx}.png").absolute()
        data_path = _resolve_processed_path(spec, processed_dir)
        _log("DEBUG", f"fig{idx}: resolved data_path={data_path}")
        records: Any = []
        if data_path and data_path.exists():
            try:
                records = _read_json(data_path)
            except Exception:
                _log("WARNING", f"fig{idx}: failed to read data from {data_path}, using empty records")
                records = []

        high, sentence, policy = _compute_highlights_for_spec(spec, records)
        _log("DEBUG", f"fig{idx}: aggregation_policy={policy}, numeric_sentence={sentence}")
        policies_by_idx[idx] = policy
        numeric_highlights_by_id[str(spec.get("id") or f"fig{idx}")] = high

        base_summary = (
            spec.get("why_this_figure")
            or spec.get("note")
            or spec.get("title")
            or ""
        )
        summary_text = _ensure_numeric_in_text(base_summary, sentence)

        fig_summaries.append(
            {
                "id": spec.get("id") or f"fig{idx}",
                "title": spec.get("title") or f"Figure {idx}",
                "figure_path": str(fig_path),
                "referenced_data": [spec.get("source_reference")] if spec.get("source_reference") else [],
                "summary": summary_text,
                "aggregation_policy": policy,
                "metrics": high,
            }
        )

    # Prepare workflow_state for ExplainerAgent
    workflow_state: Dict[str, Any] = dict(stage1_context)
    workflow_state.setdefault("project_name", project_name)
    workflow_state.setdefault("paths", {})
    workflow_state["paths"].update(
        {
            "figures_dir": str(figures_dir),
            "data_processed_dir": str(processed_dir),
        }
    )

    # Ensure agent writes to the figures directory
    os.environ["STAGE1_OUTPUTS_DIR"] = str(figures_dir)

    _log("INFO", "Calling ExplainerAgent.explain with figure summaries")
    agent = ExplainerAgent()
    text = agent.explain(fig_summaries, workflow_state)
    _log("INFO", f"ExplainerAgent returned {len(text)} chars")
    try:
        analysis = json.loads(text)
        _log("INFO", "Parsed ExplainerAgent output as JSON successfully")
    except Exception:
        _log("WARNING", "ExplainerAgent output not valid JSON; falling back to default analysis shell")
        # Fallback to an empty normalized shell
        analysis = {
            "project_name": project_name,
            "analysis_title": "Figure-grounded analysis and conclusions",
            "generated_at": "",
            "research_paradigm": "attribution_analysis",
            "research_question": stage1_context.get("research_question") or stage1_context.get("question"),
            "scenario_description": stage1_context.get("scenario_description") or "",
            "figures_analyzed": [],
            "groups_included": [],
            "key_findings": [],
            "metrics_citations": {},
            "supporting_evidence_notes": [],
            "limitations": [],
            "recommendations": [],
            "data_sources": [],
            "paths": {"figures_dir": str(figures_dir), "data_processed_dir": str(processed_dir)},
            "version": 2,
        }

    # Enforce per-figure numeric presence and consistency
    enriched_figs: List[Dict[str, Any]] = []
    for i, fig in enumerate(analysis.get("figures_analyzed", [])[: len(fig_summaries)]):
        try:
            policy = policies_by_idx.get(i + 1) or fig_summaries[i].get("aggregation_policy") or "trajectory_full_range"
            summary = str(fig.get("summary") or "")
            # Ensure numeric
            fallback_sentence = _compose_numeric_sentence(
                fig_summaries[i].get("metrics") or {},
                "final" if policy == "final" else "trajectory",
            )
            fixed = _ensure_numeric_in_text(summary, fallback_sentence)
            # Enforce policy consistency
            fixed = _enforce_consistency(fixed, policy)
            fig["summary"] = fixed
        except Exception:
            _log("WARNING", f"Post-process summary for figure index {i} failed; keeping original")
        enriched_figs.append(fig)
    if enriched_figs:
        analysis["figures_analyzed"] = enriched_figs
        _log("DEBUG", f"Post-processed summaries for {len(enriched_figs)} figures")

    # Write to figure_analysis.json
    out_path = figures_dir.parent / "figure_analysis.json"
    _log("INFO", f"Writing figure_analysis.json -> {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    _log("INFO", "Stage-2 completed")

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage-2 orchestrator: build figure_analysis.json")
    parser.add_argument("--project-name", dest="project_name", default="social_dynamics_combine")
    parser.add_argument("--figures-dir", dest="figures_dir", default=None)
    parser.add_argument("--processed-dir", dest="processed_dir", default=None)
    parser.add_argument("--stage1-context", dest="stage1_context", default=None, help="Path to stage1_context.json or workflow_state.json")
    args = parser.parse_args(argv)

    project_name = args.project_name
    figures_dir = Path(args.figures_dir).absolute() if args.figures_dir else Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/analysis/data/processed/figures").absolute()
    processed_dir = Path(args.processed_dir).absolute() if args.processed_dir else Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/analysis/data/processed").absolute()

    import time
    log_file = figures_dir / "stage2.log"
    def _log(level: str, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cleaned = str(msg).rstrip("\n")
        line = f"{ts} | {level:<8} | stage2 - {cleaned}"
        try:
            print(line)
        except Exception:
            pass
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    _log("INFO", f"CLI args parsed: project_name={project_name}")
    _log("INFO", f"CLI paths: figures_dir={figures_dir}, processed_dir={processed_dir}")

    # Load stage1 context/workflow
    ctx: Dict[str, Any] = {"project_name": project_name}
    if args.stage1_context:
        try:
            p = Path(args.stage1_context)
            if p.exists():
                ctx = _read_json(p)
                _log("INFO", f"Loaded stage1_context from {p}")
            else:
                _log("WARNING", f"stage1_context path not found: {p}")
        except Exception as e:
            _log("ERROR", f"Failed to load stage1_context: {e}")
    else:
        # Try default workflow_state.json under project
        default_wf = Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/workflow_state.json")
        try:
            if default_wf.exists():
                ctx = _read_json(default_wf)
                ctx.setdefault("project_name", project_name)
                _log("INFO", f"Loaded default workflow_state from {default_wf}")
            else:
                _log("WARNING", f"Default workflow_state not found at {default_wf}")
        except Exception as e:
            _log("ERROR", f"Failed to load default workflow_state: {e}")

    out = run_stage2(
        stage1_context=ctx,
        paths={"figures_dir": str(figures_dir), "data_processed_dir": str(processed_dir), "project_name": project_name},
        cfg={},
    )
    _log("INFO", f"Stage-2 output path: {out}")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


