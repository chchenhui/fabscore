import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Union

try:
    # 优先包内相对导入（作为模块运行时）
    from .agent_client import SimpleChatLLM
except Exception:
    # 作为脚本直接运行时，尝试同目录导入
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    try:
        from agent_client import SimpleChatLLM  # type: ignore
    except Exception:
        # 再回退：将项目根加入 sys.path，以支持 'src.researcher...' 形式
        PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from src.researcher.analysis.agent.agent_client import SimpleChatLLM  # type: ignore


VisualizationSpec = Dict[str, Any]


class AnalyzerAgent:
    def __init__(
        self,
        model_config_name: Optional[str] = None,
        model_config_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_config_name = (
            model_config_name
            or os.environ.get("ONESIM_MODEL_NAME")
            or "openai-gpt4o"
        )
        self.model_config_path = (
            model_config_path
            or os.environ.get("ONESIM_MODEL_CONFIG", "config/model_config.json")
        )
        self.system_prompt = (
            system_prompt
            or "".join(
                [
                    "You are a senior research visualization planner. ",
                    "Given a simulation scene specification, a research question, and available processed metrics, ",
                    "propose the three most insightful visualizations. Output strict JSON only.",
                ]
            )
        )
        self.llm = SimpleChatLLM(
            config_name=self.model_config_name, config_path=self.model_config_path
        )

    def propose_figures(
        self,
        scene_info: Union[str, Dict[str, Any]],
        workflow_state: Union[str, Dict[str, Any]],
        index: Union[str, Dict[str, Any], None] = None,
        max_results: int = 3,
        max_retries: int = 2,
    ) -> List[VisualizationSpec]:
        scene = self._ensure_json_object(scene_info, label="scene_info")
        workflow = self._ensure_json_object(workflow_state, label="workflow_state")
        index_summary = self._summarize_index(index)
        processed_catalog = (
            self._build_processed_catalog(index_summary.get("details", {}).get("path"))
            if index_summary.get("type") == "directory"
            else None
        )

        prompt = self._build_prompt(scene, workflow, index_summary, max_results, processed_catalog)
        last_raw: Optional[str] = None

        for attempt in range(max_retries + 1):
            raw = self.llm.chat(user_query=prompt, system_prompt=self.system_prompt)
            last_raw = raw
            specs = self._parse_llm_json(raw)
            ok, errors = self._validate_specs(specs, max_results)
            if ok:
                # Side-effect: if processed dir provided, persist figure plan JSONs for downstream
                try:
                    self._maybe_save_specs(index_summary, specs[:max_results])
                except Exception:
                    pass
                return specs[:max_results]

            # build a repair prompt with explicit error feedback
            prompt = self._build_repair_prompt(
                scene=scene,
                workflow=workflow,
                index_summary=index_summary,
                max_results=max_results,
                previous_output=raw,
                errors=errors,
                attempt=attempt + 1,
            )

        raise ValueError(
            f"LLM 未能在 {max_retries + 1} 次尝试内生成有效的 VisualizationSpec。最后一次原始输出: {last_raw}"
        )

    def _ensure_json_object(
        self, obj: Union[str, Dict[str, Any]], label: str
    ) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            # If path exists, load file
            if os.path.exists(obj):
                with open(obj, "r", encoding="utf-8") as f:
                    return json.load(f)
            # Otherwise try to parse as JSON string
            try:
                return json.loads(obj)
            except Exception as e:
                raise ValueError(f"{label} must be dict, path, or JSON string: {e}")
        raise ValueError(f"{label} must be dict, path string, or JSON string.")

    def _summarize_index(
        self, index: Union[str, Dict[str, Any], None]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"type": None, "details": None}
        if index is None:
            return summary

        if isinstance(index, dict):
            keys = list(index.keys())
            sample_keys = keys[:20]
            summary["type"] = "dict"
            summary["details"] = {
                "key_count": len(keys),
                "sample_keys": sample_keys,
            }
            return summary

        if isinstance(index, str):
            if os.path.isdir(index):
                # List top-level files only (avoid heavy IO)
                try:
                    entries = sorted(os.listdir(index))
                except Exception:
                    entries = []
                files = [e for e in entries if os.path.isfile(os.path.join(index, e))]
                dirs = [e for e in entries if os.path.isdir(os.path.join(index, e))]
                summary["type"] = "directory"
                summary["details"] = {
                    "path": index,
                    "file_count": len(files),
                    "dir_count": len(dirs),
                    "sample_files": files[:30],
                    "sample_dirs": dirs[:10],
                }
                return summary

            if os.path.isfile(index):
                summary["type"] = "file"
                summary["details"] = {
                    "path": index,
                    "size_bytes": os.path.getsize(index),
                }
                return summary

        summary["type"] = type(index).__name__
        summary["details"] = str(index)[:500]
        return summary

    def _build_prompt(
        self,
        scene: Dict[str, Any],
        workflow: Dict[str, Any],
        index_summary: Dict[str, Any],
        max_results: int,
        processed_catalog: Optional[Dict[str, Any]] = None,
    ) -> str:
        research_question = workflow.get("research_question") or workflow.get(
            "research_topic"
        )
        metrics = scene.get("odd_protocol", {}).get("metrics") or scene.get("metrics")
        if not isinstance(metrics, list):
            metrics = []

        metrics_brief: List[Dict[str, Any]] = []
        for m in metrics:
            if not isinstance(m, dict):
                continue
            metrics_brief.append(
                {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "visualization_type": m.get("visualization_type"),
                    "function_name": m.get("function_name"),
                    "description": m.get("description"),
                    "update_interval": m.get("update_interval"),
                }
            )

        scene_overview = scene.get("odd_protocol", {}).get("overview", {})
        scene_design = scene.get("odd_protocol", {}).get("design_concepts", {})

        guide = {
            "task": "Select exactly three figures that best explain the research question using available metrics and processed outputs.",
            "output_schema": {
                "id": "string (short identifier)",
                "title": "string (clear human-friendly title)",
                "data_source_category": "one of: agent|environment|processed|simulation_summary|custom",
                "source_reference": "string (which metric/function/file/folder it relies on)",
                "group_by_fields": [
                    "array of strings (dimensions or fields used to group data)"
                ],
                "aggregation": {
                    "method": "string (e.g., count|mean|sum|proportion|entropy|region_count)",
                    "field": "string or null (field to aggregate)",
                    "note": "string (any caveats)",
                },
                "suggested_visualization_type": "string (bar|line|area|heatmap|scatter|network)",
                "why_this_figure": "string (concise rationale tied to research question)",
            },
            "constraints": [
                "Return JSON array ONLY, with exactly three items.",
                "Prefer metrics whose visualization_type aligns with the intent.",
                "If processed outputs exist, consider aggregations across experiment groups.",
                "Be specific about grouping (e.g., by experiment group, openness, interaction_range).",
                "For time-evolving metrics (e.g., simulation outputs across steps), prefer line charts with 'step' (or 'time') as the x-axis; ensure the x-axis is sorted ascending and include clear legends when multiple groups/series exist.",
                "Label axes explicitly (x='Step' or 'Time', y=metric name), enable grid lines, and use integer tick locator on the x-axis; rotate tick labels if dense.",
                "If the metric is normalized/bounded in [0,1] (e.g., indices/proportions), constrain y-limits to [0,1] with evenly spaced ticks to improve interpretability.",
                "Use a consistent color palette; when multiple groups exist, prefer placing the legend outside the plotting area to avoid occlusion (e.g., upper-right with bbox_to_anchor).",
                "Promote diversity across the three figures: avoid producing three highly similar charts; consider combining line (trend), bar (category comparison / end-state), and heatmap (interaction/condition matrix) where appropriate.",
                "Axes must be semantically meaningful (units/categories/ranges); consider log-scale y when values span orders of magnitude.",
                "When time series are long/dense, allow readable presentation via downsampling (plot every N steps) or smoothing (rolling mean) while preserving the key trends.",
                "If groups are numerous, limit to top_k salient groups (e.g., by final value or variance) and clearly state the selection logic in the spec's note.",
                "Keep annotations minimal and purposeful (e.g., mark key steps or thresholds) without cluttering the chart.",
            ],
        }

        prompt_obj = {
            "scene_overview": scene_overview,
            "scene_design_concepts": scene_design,
            "metrics_catalog": metrics_brief,
            "workflow_research_question": research_question,
            "index_summary": index_summary,
            "processed_catalog": processed_catalog,
            "instructions": guide,
            "max_results": max_results,
        }
        return (
            "请根据以下场景说明、研究问题与可用指标，产出最能说明问题的三张图的规划。"
            "严格输出 JSON 数组（3 个对象），并遵循给定 schema。\n\n"
            + json.dumps(prompt_obj, ensure_ascii=False, indent=2)
        )

    def _build_processed_catalog(self, dir_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not dir_path or not os.path.isdir(dir_path):
            return None
        datasets: List[Dict[str, Any]] = []
        try:
            entries = sorted(os.listdir(dir_path))
        except Exception:
            entries = []
        for name in entries:
            if not name.lower().endswith(".json"):
                continue
            # skip figure plan outputs to avoid feedback loop
            if name.startswith("figures_analysis_combine"):
                continue
            fpath = os.path.join(dir_path, name)
            if not os.path.isfile(fpath):
                continue
            try:
                size_b = os.path.getsize(fpath)
            except Exception:
                size_b = 0
            summary: Dict[str, Any] = {
                "filename": name,
                "size_bytes": size_b,
                "category": None,
                "entry_count": None,
                "sample_fields": None,
                "nested_data_shape": None,
                "time_field": None,
                "group_field": None,
            }
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # Expect either {file_info, data: list} or list
                if isinstance(obj, dict):
                    fi = obj.get("file_info") or {}
                    if isinstance(fi, dict):
                        summary["category"] = fi.get("category") or None
                    data = obj.get("data")
                else:
                    data = obj
                rows = data if isinstance(data, list) else []
                summary["entry_count"] = len(rows)
                # sample head
                head = rows[: min(10, len(rows))]
                # collect top-level keys
                fields: Dict[str, int] = {}
                for r in head:
                    if isinstance(r, dict):
                        for k in r.keys():
                            fields[k] = fields.get(k, 0) + 1
                if fields:
                    summary["sample_fields"] = sorted(list(fields.keys()))
                # detect common fields
                for tf in ("step", "time", "t", "round"):
                    if tf in fields:
                        summary["time_field"] = tf
                        break
                for gf in ("group_name", "group", "experiment_group"):
                    if gf in fields:
                        summary["group_field"] = gf
                        break
                # detect nested data shape
                nested_shape = None
                for r in head:
                    if isinstance(r, dict) and isinstance(r.get("data"), dict):
                        dd = r.get("data")
                        if isinstance(dd, dict) and "xAxis" in dd and "series" in dd:
                            nested_shape = "distribution(xAxis,series)"
                            break
                    if isinstance(r, dict) and (isinstance(r.get("data"), (int, float, str)) or r.get("data") is None):
                        nested_shape = nested_shape or "scalar"
                summary["nested_data_shape"] = nested_shape
            except Exception:
                # keep minimal info
                pass
            datasets.append(summary)
        # Also provide a lightweight name->filename mapping to encourage exact references
        name_map: Dict[str, str] = {}
        for ds in datasets:
            fname = ds.get("filename") or ""
            base = os.path.splitext(fname)[0]
            cat = (ds.get("category") or "").strip()
            if cat:
                name_map[cat] = fname
            name_map[base] = fname
        catalog = {
            "path": dir_path,
            "datasets": datasets,
            "name_to_file": name_map,
        }
        return catalog

    def _maybe_save_specs(self, index_summary: Dict[str, Any], specs: List[VisualizationSpec]) -> None:
        try:
            details = index_summary.get("details") if isinstance(index_summary, dict) else None
            dir_path = details.get("path") if isinstance(details, dict) else None
            if not dir_path or not os.path.isdir(dir_path):
                return
            # write to figures_analysis_combine*.json in processed dir
            out_cn = os.path.join(dir_path, "figures_analysis_combine.json")
            out_en = os.path.join(dir_path, "figures_analysis_combine_en.json")
            with open(out_cn, "w", encoding="utf-8") as f:
                json.dump(specs, f, ensure_ascii=False, indent=2)
            # For now, duplicate to _en; upstream can translate if needed
            with open(out_en, "w", encoding="utf-8") as f:
                json.dump(specs, f, ensure_ascii=False, indent=2)
        except Exception:
            # best-effort only
            pass

    def _parse_llm_json(self, raw: str) -> List[VisualizationSpec]:
        if not raw:
            return []
        # First attempt: direct JSON
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass

        # Second: extract JSON array substring
        array_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except Exception:
                pass

        # Third: try to replace single quotes and trailing commas
        cleaned = raw.replace("'", '"')
        cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            return []
        return []

    def _validate_specs(
        self, specs: Optional[List[Dict[str, Any]]], max_results: int
    ) -> (bool, List[str]):
        errors: List[str] = []
        if not isinstance(specs, list):
            return False, ["输出不是 JSON 数组"]
        if len(specs) != max_results:
            errors.append(f"需要 {max_results} 个对象，实际为 {len(specs)}")

        allowed_source = {"agent", "environment", "processed", "simulation_summary", "custom"}
        allowed_vis = {"bar", "line", "area", "heatmap", "scatter", "network"}

        for i, item in enumerate(specs):
            if not isinstance(item, dict):
                errors.append(f"第 {i} 个元素不是对象")
                continue
            for key in [
                "id",
                "title",
                "data_source_category",
                "source_reference",
                "group_by_fields",
                "aggregation",
                "suggested_visualization_type",
                "why_this_figure",
            ]:
                if key not in item:
                    errors.append(f"第 {i} 个对象缺少字段: {key}")

            if item.get("data_source_category") not in allowed_source:
                errors.append(
                    f"第 {i} 个对象 data_source_category 非法: {item.get('data_source_category')}"
                )
            if item.get("suggested_visualization_type") not in allowed_vis:
                errors.append(
                    f"第 {i} 个对象 suggested_visualization_type 非法: {item.get('suggested_visualization_type')}"
                )
            if not isinstance(item.get("group_by_fields"), list):
                errors.append(f"第 {i} 个对象 group_by_fields 需要为数组")
            agg = item.get("aggregation")
            if not isinstance(agg, dict) or "method" not in agg:
                errors.append(f"第 {i} 个对象 aggregation 无效，需包含 method")
        return len(errors) == 0, errors

    def _build_repair_prompt(
        self,
        scene: Dict[str, Any],
        workflow: Dict[str, Any],
        index_summary: Dict[str, Any],
        max_results: int,
        previous_output: str,
        errors: List[str],
        attempt: int,
    ) -> str:
        base = self._build_prompt(scene, workflow, index_summary, max_results)
        feedback = {
            "attempt": attempt,
            "previous_output": previous_output,
            "errors": errors,
            "instruction": "请严格修复以上问题，仅输出 JSON 数组（恰好3个对象），不要任何额外文字。",
        }
        return base + "\n\n修复提示：\n" + json.dumps(feedback, ensure_ascii=False, indent=2)


# Convenience function for external callers
def propose_figures(
    scene_info: Union[str, Dict[str, Any]],
    workflow_state: Union[str, Dict[str, Any]],
    index: Union[str, Dict[str, Any], None] = None,
    model_config_name: Optional[str] = None,
    model_config_path: Optional[str] = None,
) -> List[VisualizationSpec]:
    agent = AnalyzerAgent(
        model_config_name=model_config_name, model_config_path=model_config_path
    )
    return agent.propose_figures(scene_info, workflow_state, index)


if __name__ == "__main__":
    # Lightweight manual check (no disk writes)
    sample_scene_path = os.environ.get(
        "SCENE_INFO_PATH",
        "/data/wujinchao/test/YuLan-OneSim-Dev/projects/social_dynamics_combine/scene_info(1).json",
    )
    sample_workflow_path = os.environ.get(
        "WORKFLOW_STATE_PATH",
        "/data/wujinchao/test/YuLan-OneSim-Dev/projects/social_dynamics_combine/workflow_state.json",
    )
    sample_index_dir = os.environ.get(
        "PROCESSED_DIR",
        "/data/wujinchao/test/YuLan-OneSim-Dev/projects/social_dynamics_combine/analysis/data/processed",
    )

    agent = AnalyzerAgent()
    try:
        results = agent.propose_figures(sample_scene_path, sample_workflow_path, sample_index_dir)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Failed to propose figures: {e}")


