import json
import os
import re
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .agent_client import SimpleChatLLM
except Exception:
    from src.researcher.analysis.agent.agent_client import SimpleChatLLM  # type: ignore


class ExplainerAgent:
    """
    Generate concise, structured explanations for three figures based on their
    summary metadata and the workflow_state context.

    - Input: fig_summaries (list[dict]), workflow_state (dict)
    - Output: JSON string with per-figure: phenomenon -> key_values -> conclusion
    - Side-effect: writes the JSON to fig_explanations.json under outputs dir
      resolved by env STAGE1_OUTPUTS_DIR or default ./outputs
    """

    def __init__(
        self,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        cfg_name = (
            config_name or os.environ.get("ONESIM_MODEL_NAME") or "openai-gpt4o"
        )
        cfg_path = (
            config_path
            or os.environ.get("ONESIM_MODEL_CONFIG")
            or "config/model_config.json"
        )
        self.llm = SimpleChatLLM(config_name=cfg_name, config_path=cfg_path)
        self.system_prompt = (
            system_prompt
            or (
                "You are a research explainer agent. Given workflow context and "
                "three figure summaries, produce a JSON-only analysis capturing, "
                "for each figure: the main phenomenon observed, key numeric values "
                "(as label/value pairs), and a concise conclusion tied to the "
                "research question. Do not include markdown or extra text."
            )
        )

    def explain(self, fig_summaries: List[Dict[str, Any]], workflow_state: Dict[str, Any]) -> str:
        out_dir = self._resolve_outputs_dir()
        import time
        log_file = out_dir / "explainer_agent.log"
        def _log(level: str, msg: str) -> None:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cleaned = str(msg).rstrip("\n")
            line = f"{ts} | {level:<8} | explainer - {cleaned}"
            try:
                print(line)
            except Exception:
                pass
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        _log("INFO", f"Explain start: outputs_dir={out_dir}")
        prompt = self._build_prompt(fig_summaries, workflow_state)
        _log("DEBUG", f"Prompt length={len(prompt)}")
        raw = self.llm.chat(user_query=prompt, system_prompt=self.system_prompt)
        _log("INFO", f"LLM chat returned length={len(raw)}")
        data = self._parse_json_strict(raw)
        if not data:
            _log("WARNING", "Initial parse failed; attempting repair prompt")
            repair_prompt = self._build_repair_prompt(fig_summaries, workflow_state, raw)
            raw = self.llm.chat(user_query=repair_prompt, system_prompt=self.system_prompt)
            _log("INFO", f"Repair chat returned length={len(raw)}")
            data = self._parse_json_strict(raw)

        if not data:
            _log("WARNING", "Repair parse failed; building fallback JSON")
            data = self._fallback_build(fig_summaries, workflow_state)

        # Normalize to example schema and key order
        data = self._normalize_to_figure_analysis_schema(data, fig_summaries, workflow_state)
        _log("DEBUG", "Normalized analysis schema")

        # Persist to outputs/fig_explanations.json
        out_path = self._resolve_outputs_dir() / "fig_explanations.json"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            _log("INFO", f"Wrote fig_explanations.json -> {out_path}")
        except Exception as e:
            _log("ERROR", f"Failed to write fig_explanations.json: {e}")
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _build_prompt(self, fig_summaries: List[Dict[str, Any]], workflow_state: Dict[str, Any]) -> str:
        rq = (
            workflow_state.get("research_question")
            or workflow_state.get("research_topic")
            or workflow_state.get("question")
        )
        project_name = (
            workflow_state.get("project_name")
            or workflow_state.get("project")
            or ""
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        guide: Dict[str, Any] = {
            "task": "Produce analysis JSON matching scripts/figure_analysis.json structure exactly.",
            "constraints": [
                "Output STRICT JSON only. No extra text or markdown.",
                "Top-level keys (and order if possible): project_name, analysis_title, generated_at, research_paradigm, research_question, scenario_description, figures_analyzed, groups_included, key_findings, metrics_citations, supporting_evidence_notes, limitations, recommendations, data_sources, paths, version.",
                "figures_analyzed: array of 3 objects with keys: name, file, summary, referenced_data (array of strings).",
                "Keep language concise, factual, grounded in provided workflow_state and figure summaries.",
                "If specific numeric values are not available, provide qualitative but non-trivial summaries consistent with the research question.",
            ],
            "defaults": {
                "analysis_title": "Figure-grounded analysis and conclusions",
                "research_paradigm": "attribution_analysis",
                "paths": {
                    "figures_dir": "analysis/figures",
                    "data_processed_dir": "analysis/data/processed"
                },
                "version": 2
            },
            "format_example": {
                "project_name": project_name,
                "analysis_title": "Figure-grounded analysis and conclusions",
                "generated_at": now,
                "research_paradigm": "attribution_analysis",
                "research_question": rq,
                "scenario_description": "One to two sentences summarizing the simulation scenario.",
                "figures_analyzed": [
                    {
                        "name": "Average Cultural Regions",
                        "file": "avg_cultural_regions.png",
                        "summary": "A short summary of what this figure shows and the pattern.",
                        "referenced_data": ["Number_of_Cultural_Regions_all_groups.json"]
                    }
                ],
                "groups_included": [],
                "key_findings": [
                    "Finding A stated as a concise bullet.",
                    "Finding B stated as a concise bullet."
                ],
                "metrics_citations": {},
                "supporting_evidence_notes": [],
                "limitations": [],
                "recommendations": [],
                "data_sources": [
                    "Number_of_Cultural_Regions_all_groups.json"
                ],
                "paths": {
                    "figures_dir": "analysis/figures",
                    "data_processed_dir": "analysis/data/processed"
                },
                "version": 2
            }
        }

        payload = {
            "project_name": project_name,
            "generated_at": now,
            "research_question": rq,
            "workflow_state_brief": self._brief_workflow(workflow_state),
            "fig_summaries": fig_summaries,
            "instructions": guide,
        }
        return (
            "Generate JSON only. No explanations outside JSON.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _build_repair_prompt(
        self,
        fig_summaries: List[Dict[str, Any]],
        workflow_state: Dict[str, Any],
        previous_output: str,
    ) -> str:
        base = self._build_prompt(fig_summaries, workflow_state)
        feedback = {
            "error": "Your previous output was not valid strict JSON or did not match the schema. Return ONLY a valid JSON object.",
            "previous_output": previous_output[:2000],
        }
        return base + "\n\n" + json.dumps(feedback, ensure_ascii=False, indent=2)

    def _parse_json_strict(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}\s*\Z", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _fallback_build(self, fig_summaries: List[Dict[str, Any]], workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        rq = (
            workflow_state.get("research_question")
            or workflow_state.get("research_topic")
            or workflow_state.get("question")
        )
        project_name = (
            workflow_state.get("project_name")
            or workflow_state.get("project")
            or ""
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        figures: List[Dict[str, Any]] = []
        data_sources_set = set()
        for fs in fig_summaries[:3]:
            file_name = None
            try:
                file_name = Path(fs.get("figure_path", "")).name if fs.get("figure_path") else None
            except Exception:
                file_name = None
            refs = self._infer_referenced(fs)
            for r in refs:
                data_sources_set.add(r)
            figures.append(
                {
                    "name": fs.get("title") or (fs.get("id") or "Figure").replace("_", " ").title(),
                    "file": file_name or (fs.get("id") or "figure") + ".png",
                    "summary": fs.get("summary")
                    or fs.get("phenomenon")
                    or "A concise description of the visible pattern consistent with the research question.",
                    "referenced_data": refs,
                }
            )

        return {
            "project_name": project_name,
            "analysis_title": "Figure-grounded analysis and conclusions",
            "generated_at": now,
            "research_paradigm": "attribution_analysis",
            "research_question": rq,
            "scenario_description": workflow_state.get("scenario_description")
            or "A brief description of the simulation scenario and variables under study.",
            "figures_analyzed": figures,
            "groups_included": workflow_state.get("groups_included") or [],
            "key_findings": workflow_state.get("key_findings") or [],
            "metrics_citations": workflow_state.get("metrics_citations") or {},
            "supporting_evidence_notes": workflow_state.get("supporting_evidence_notes") or [],
            "limitations": workflow_state.get("limitations") or [],
            "recommendations": workflow_state.get("recommendations") or [],
            "data_sources": sorted(list(data_sources_set)) if data_sources_set else [],
            "paths": {
                "figures_dir": "analysis/figures",
                "data_processed_dir": "analysis/data/processed",
            },
            "version": 2,
        }

    def _normalize_to_figure_analysis_schema(
        self,
        data: Dict[str, Any],
        fig_summaries: List[Dict[str, Any]],
        workflow_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize the model output to match scripts/figure_analysis.json fields and order."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def as_list(value, default):
            if value is None:
                return list(default)
            if isinstance(value, list):
                return value
            return [value]

        def as_dict(value, default):
            if isinstance(value, dict):
                return value
            return dict(default)

        # Convert alternative shapes
        figures_analyzed: List[Dict[str, Any]] = []
        data_sources_from_figs = set()
        if isinstance(data.get("figures_analyzed"), list):
            for item in data["figures_analyzed"][:3]:
                name = item.get("name") or item.get("title") or "Figure"
                file_name = item.get("file") or item.get("figure") or None
                refs = as_list(item.get("referenced_data"), [])
                for r in refs:
                    if isinstance(r, str):
                        data_sources_from_figs.add(r)
                figures_analyzed.append(
                    {
                        "name": name,
                        "file": file_name or f"{name.lower().replace(' ', '_')}.png",
                        "summary": item.get("summary")
                        or item.get("phenomenon")
                        or item.get("conclusion")
                        or "",
                        "referenced_data": refs,
                    }
                )
        elif isinstance(data.get("figure_explanations"), list):
            # Map legacy schema -> required schema
            for item in data["figure_explanations"][:3]:
                name = item.get("title") or (item.get("figure_id") or "Figure").replace("_", " ").title()
                file_name = (item.get("figure_id") or name).lower().replace(" ", "_") + ".png"
                refs = as_list(item.get("referenced_data"), [])
                for r in refs:
                    if isinstance(r, str):
                        data_sources_from_figs.add(r)
                figures_analyzed.append(
                    {
                        "name": name,
                        "file": file_name,
                        "summary": item.get("summary")
                        or item.get("phenomenon")
                        or item.get("conclusion")
                        or "",
                        "referenced_data": refs,
                    }
                )
        else:
            # Build from fig_summaries if nothing provided
            for fs in fig_summaries[:3]:
                try:
                    file_name = Path(fs.get("figure_path", "")).name if fs.get("figure_path") else (fs.get("id") or "figure") + ".png"
                except Exception:
                    file_name = (fs.get("id") or "figure") + ".png"
                refs = self._infer_referenced(fs)
                for r in refs:
                    data_sources_from_figs.add(r)
                figures_analyzed.append(
                    {
                        "name": fs.get("title") or (fs.get("id") or "Figure").replace("_", " ").title(),
                        "file": file_name,
                        "summary": fs.get("summary")
                        or fs.get("phenomenon")
                        or "",
                        "referenced_data": refs,
                    }
                )

        # Compose normalized output with required key order
        normalized: Dict[str, Any] = {}
        normalized["project_name"] = data.get("project_name") or workflow_state.get("project_name") or workflow_state.get("project") or ""
        normalized["analysis_title"] = data.get("analysis_title") or "Figure-grounded analysis and conclusions"
        normalized["generated_at"] = data.get("generated_at") or now
        normalized["research_paradigm"] = data.get("research_paradigm") or "attribution_analysis"
        normalized["research_question"] = data.get("research_question") or workflow_state.get("research_question") or workflow_state.get("research_topic") or workflow_state.get("question")
        normalized["scenario_description"] = data.get("scenario_description") or "A brief description of the simulation scenario and variables under study."
        normalized["figures_analyzed"] = figures_analyzed
        normalized["groups_included"] = as_list(data.get("groups_included"), [])
        normalized["key_findings"] = as_list(data.get("key_findings"), [])
        normalized["metrics_citations"] = as_dict(data.get("metrics_citations"), {})
        normalized["supporting_evidence_notes"] = as_list(data.get("supporting_evidence_notes"), [])
        normalized["limitations"] = as_list(data.get("limitations"), [])
        normalized["recommendations"] = as_list(data.get("recommendations"), [])

        # Merge data_sources from payload and figures
        payload_sources = set([s for s in as_list(data.get("data_sources"), []) if isinstance(s, str)])
        all_sources = sorted(list(payload_sources.union(data_sources_from_figs)))
        normalized["data_sources"] = all_sources

        paths = data.get("paths") or {"figures_dir": "analysis/figures", "data_processed_dir": "analysis/data/processed"}
        normalized["paths"] = {
            "figures_dir": paths.get("figures_dir", "analysis/figures"),
            "data_processed_dir": paths.get("data_processed_dir", "analysis/data/processed"),
        }
        normalized["version"] = data.get("version") or 2
        return normalized

    def _infer_referenced(self, fs: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for k in ["source_reference", "referenced_data", "data_file", "dataset"]:
            v = fs.get(k)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, list):
                out.extend([str(x) for x in v])
        return sorted(list({*out}))

    def _brief_workflow(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "project_name",
            "status",
            "steps_completed",
            "experiment_design",
            "workflow_version",
        ]
        brief: Dict[str, Any] = {}
        for k in keys:
            v = workflow_state.get(k)
            if v is not None:
                brief[k] = v if isinstance(v, (str, int, float)) else str(v)[:400]
        return brief

    def _resolve_outputs_dir(self) -> Path:
        p = os.environ.get("STAGE1_OUTPUTS_DIR", "outputs")
        return Path(p).absolute()


def explain(fig_summaries: List[Dict[str, Any]], workflow_state: Dict[str, Any]) -> str:
    agent = ExplainerAgent()
    return agent.explain(fig_summaries, workflow_state)


__all__ = ["ExplainerAgent", "explain"]


def _default_fig_dir(project_name: str) -> Path:
    return Path(f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/analysis/data/processed/figures").absolute()


def _build_fig_summaries_from_dir(fig_dir: Path, prefer_files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    figs: List[Dict[str, Any]] = []
    candidates: List[Path] = []
    if prefer_files:
        for name in prefer_files:
            p = fig_dir / name
            if p.exists() and p.is_file():
                candidates.append(p)
    else:
        for name in ["fig1.png", "fig2.png", "fig3.png"]:
            p = fig_dir / name
            if p.exists() and p.is_file():
                candidates.append(p)
    for p in candidates[:3]:
        stem = p.stem
        figs.append(
            {
                "id": stem,
                "title": stem.replace("_", " ").title(),
                "figure_path": str(p),
                "referenced_data": [],
            }
        )
    return figs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ExplainerAgent test runner")
    parser.add_argument("--project-name", default="social_dynamics_combine", help="Project name under /projects/{project-name}")
    parser.add_argument("--fig-path", default=None, help="Path to a figure (.png). If omitted, will search fig1.png~fig3.png under the project's figures directory")
    parser.add_argument("--workflow-state", dest="workflow_state", default=None, help="Path to workflow_state.json; defaults to projects/{name}/workflow_state.json")
    parser.add_argument("--outputs-dir", dest="outputs_dir", default=None, help="Outputs directory to write fig_explanations.json; defaults to the project's figures directory if omitted")

    args = parser.parse_args(argv)
    project_name = args.project_name
    # If a specific fig path is provided, prefer that single file; otherwise, scan for fig1..fig3
    if args.fig_path:
        fig_path = args.fig_path
        fig_dir = Path(fig_path).parent
        prefer = [Path(fig_path).name]
    else:
        fig_path = None
        fig_dir = _default_fig_dir(project_name)
        prefer = None
    fig_summaries = _build_fig_summaries_from_dir(fig_dir, prefer_files=prefer)
    if not fig_summaries:
        sys.stderr.write(f"No figures found under {fig_dir}\n")
        return 2

    # Load or synthesize workflow_state
    workflow_state_path = args.workflow_state or f"/data/wujinchao/test/YuLan-OneSim-Dev/projects/{project_name}/workflow_state.json"
    workflow_state: Dict[str, Any]
    try:
        p = Path(workflow_state_path)
        if p.exists():
            workflow_state = json.loads(p.read_text(encoding="utf-8"))
        else:
            workflow_state = {"project_name": project_name}
    except Exception:
        workflow_state = {"project_name": project_name}

    # Ensure outputs dir env for writer: default to the project's figures directory when not provided
    try:
        desired_outputs_dir = args.outputs_dir or str(Path(fig_dir).absolute())
        os.environ["STAGE1_OUTPUTS_DIR"] = str(Path(desired_outputs_dir).absolute())
    except Exception:
        pass

    text = explain(fig_summaries, workflow_state)
    sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
