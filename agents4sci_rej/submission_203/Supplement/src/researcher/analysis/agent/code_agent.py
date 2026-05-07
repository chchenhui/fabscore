import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_client import SimpleChatLLM


def _sanitize_code(text: str) -> str:
    """Remove markdown code fences like ```python and ``` from model outputs.
    Keep inner content intact and strip leading/trailing whitespace.
    """
    try:
        lines = text.splitlines()
    except Exception:
        return text
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned


class PlotCodeAgent:
    """
    High-level agent that leverages an LLM to generate and patch
    Python plotting scripts from a VisualizationSpec dict.

    The agent communicates natural-language prompts to the model and expects
    raw Python code (not JSON) in return. It is designed to be extensible:
    - Pluggable model configuration via config_name/config_path
    - Prompt templates are overridable through constructor args
    - Easy to extend with more channels (e.g., critique, explanation)
    """

    def __init__(
        self,
        config_name: str = "openai-gpt4o",
        config_path: str = "config/model_config.json",
        system_prompt: Optional[str] = None,
        gen_template: Optional[str] = None,
        patch_template: Optional[str] = None,
    ) -> None:
        # 允许通过环境变量覆盖模型选择，便于统一用 ONESIM_MODEL_NAME/ONESIM_MODEL_CONFIG 控制
        cfg_name = os.environ.get("ONESIM_MODEL_NAME", config_name)
        cfg_path = os.environ.get("ONESIM_MODEL_CONFIG", config_path)
        self.llm = SimpleChatLLM(config_name=cfg_name, config_path=cfg_path)
        self.system_prompt = (
            system_prompt
            or """
You are PlotCodeAgent, an expert Python data visualization developer.
Your job is to generate runnable Python plotting scripts from a given VisualizationSpec.
Output ONLY Python code. Do not wrap code in markdown fences. Do not return JSON.
Prefer matplotlib or plotly depending on the spec, and import required libraries.
Make the script self-contained, including minimal example data adaptation if needed.
Validate column names from the spec and fail gracefully with comments if ambiguous.
For simulation or time-evolving metrics, treat 'step' (or 'time') as the default x-axis for line charts when available. Ensure x is sorted ascending, use readable integer ticks, and include a legend for grouped series.
Always call plt.tight_layout() and plt.show() at the end so that headless runners can capture the figure.
When plotting line charts, prefer figsize=(12, 6), label x as 'Step' (or 'Time') and y as the metric name, enable grid (alpha≈0.3), and use a consistent color palette.
If the metric appears bounded in [0, 1] (e.g., indices, proportions, rates), set y-limits to [0, 1] and use evenly spaced y-ticks (e.g., 0.0, 0.2, …, 1.0).
Use integer tick locator for the x-axis; rotate tick labels slightly if overcrowded.
For multiple groups/conditions, map groups to distinct hues and optionally vary linestyle or markers; place legend outside the top-right via bbox_to_anchor to avoid occluding data.
Use moderate linewidth (~1.8) and small markers (~4) at reasonable intervals to improve readability without clutter.
Favor diverse visualization types when appropriate (line, bar, area, heatmap, scatter). Choose chart types that best match data semantics and the analytical question.
Ensure axes carry meaningful domain semantics (units, categories, ranges). Consider log scale for the y-axis when data span multiple orders of magnitude.
For dense time series, support controlled downsampling: allow a step_stride (e.g., plot every Nth step) or compute a rolling mean (configurable window) for smoother trends.
If many groups exist, restrict to top_k most salient groups (by final magnitude or variance) and clearly state the selection logic in code comments.
When helpful, annotate key events/thresholds and highlight endpoints; avoid clutter by limiting text annotations.

Data loading policy:
- If the spec includes '_resolved_data_path', load that path directly.
- Else, if 'source_reference' looks like a filename (e.g., '*.json'), try resolving:
  1) spec.get('processed_dir')/source_reference if available; otherwise
  2) os.environ.get('STAGE1_PROCESSED_DIR')/source_reference if the env var is set.
- Handle JSON structured as {"data": list} or a top-level list of records; if rows contain nested 'data' per record, adapt appropriately in code.

Labeling requirements (very important):
- Legend/group labels MUST reflect experimental conditions (e.g., openness, interaction_range, group_name). Never use generic labels like 'Group A', 'Group B', or 'dimension'.
- If multiple condition fields exist (e.g., openness and interaction_range), compose labels such as "openness=medium, interaction_range=third_order".
- X-axis label can be simple (e.g., 'Step' or 'Time' or a category field). Ensure x is numeric-sorted for steps/time even if strings like 'step_10'.
- Y-axis label MUST explicitly state what is measured and how (e.g., 'Global Polarization Index (mean)', 'Count of regions', 'Sum of adoption_rate'). Infer from spec.aggregation.method/field and/or metric/category names.
- Provide a clear legend title using the second grouping field name when applicable (e.g., 'interaction_range').
""".strip()
        )

        self.gen_template = (
            gen_template
            or """
Given the following VisualizationSpec (a JSON-like dict string), generate a complete, runnable Python plotting script.

Requirements:
- Output raw Python, no markdown fences, no JSON.
- Include all necessary imports.
- Handle missing/ambiguous fields robustly with reasonable defaults and comments.
- Prefer loading real data: if '_resolved_data_path' is present, load it; otherwise if 'source_reference' exists, try resolving using spec.get('processed_dir') or the environment variable 'STAGE1_PROCESSED_DIR'. If no dataset can be found, create a small sample DataFrame that matches the fields.
- Follow best practices for clarity and readability.
 - For line plots/time-series from simulations, prefer using a column named 'step' (or 'time') as the x-axis if present; sort by x ascending before plotting; when multiple groups exist, map them to color/hue and add a legend.
 - Prefer grid lines for readability, set axis labels and a clear title, call plt.tight_layout() then plt.show() at the end.
 - Use figsize≈(12, 6); set x-label 'Step' (or 'Time'), y-label as the metric; apply integer tick locator on x and rotate labels if dense; enable grid(alpha≈0.3).
 - If the metric is normalized/bounded [0,1], set ylim(0,1) and evenly spaced y-ticks; otherwise infer sensible y-limits from data without clipping extremes.
 - For multiple groups, prefer distinct hues and optionally linestyle/markers; place legend outside upper-right (bbox_to_anchor) to avoid overlap with curves.
 - Choose linewidth≈1.8 and markersize≈4; avoid over-plotting by marking every N steps when dense (e.g., markevery=5 or 10).
 - Prefer diverse plot types based on intent and data semantics (line/area for trends, bar for category comparisons, heatmap for matrix-like interactions, scatter for relationships).
 - Axes should use meaningful domain units/categories; consider log-scale y when values span orders of magnitude.
 - For long or dense series, add optional step_stride (plot every N-th sample) and/or rolling mean (rolling_window) to smooth noise; document choices.
 - If groups are numerous, select top_k representative groups (by final value, average, variance, or domain ranking) to maintain readability; mention selection rule in code comments.
 - Keep annotations minimal yet informative: optional vertical lines for key steps, endpoints labels, or threshold lines.

Data normalization for x-axis ordering:
- If the x-axis dimension corresponds to steps/time and arrives as strings (e.g., 'step_1', 'step_10', 'step_2'), parse the numeric portion and sort by numeric value to avoid lexical misordering.
- Ensure the plotted x values are strictly ascending numerically; set an integer tick locator when appropriate.
- Implement this explicitly in code:
  1) Define a helper: extract_num(s) -> int that uses regex r"\\d+"; returns int(digits) if found; else falls back to int(s) when possible, otherwise 0.
  2) Identify x field among ['step','time','t','round'] (case-insensitive).
  3) If using pandas: create a new column '<x>_num' = df[x].apply(extract_num); sort_values(by=['<x>_num', ...]) before plotting; use the numeric column for ordering while displaying original labels.
  4) If using plain lists/dicts: sort the data by extract_num(item[x]).
  5) For Matplotlib, set x-axis to integer locator (MaxNLocator(integer=True)) and rotate labels slightly if crowded.

Aggregation and field usage (critical):
- Use spec.aggregation.field as the y-value source. Do NOT invent new column names.
- If spec.aggregation.method is one of ['mean','sum','count'], apply it over the provided field; if it's another descriptive label (e.g., 'region_count'), still compute using the provided field and treat the method as descriptive for labeling only.
- Many processed rows follow a common schema with keys like ['group_name', 'step', 'data']:
  - When field == 'data' and the value is a scalar per row, aggregate that directly by step/group.
  - When row['data'] is a dict with 'xAxis' and 'series', adapt into a tabular structure before plotting (explode categories to rows or plot distribution appropriately).

Labeling policy (enforce via code logic and comments):
- Derive the legend/group labels from experimental condition fields present in the data/spec. Check, in order: ['openness', 'interaction_range', 'group_name', 'group', 'experiment_group']. For multiple, combine as 'field=value' pairs joined by ', '.
- For x-axis: label with the actual field name ('Step'/'Time' if step/time), otherwise the category dimension.
- For y-axis: if spec.aggregation.method == 'mean' or 'sum' and has a 'field', label as 'Mean <field>' or 'Sum of <field>'. If method == 'count', label as 'Count of <entity>' where <entity> is inferred from the metric/category or filename. Avoid vague labels.
- Avoid generic placeholders like 'Group A/B' or 'dimension'. Use concrete condition values from the dataset rows.

VisualizationSpec:
{viz_spec}
""".strip()
        )

        self.patch_template = (
            patch_template
            or '''
You are given an existing Python plotting script that failed to run. Produce a corrected script.

Provide ONLY the full corrected Python script (no explanations, no markdown).
Make minimal changes necessary to fix the error while preserving intended visualization.
If needed, add imports or replace APIs to match installed libraries.

Original Script:
"""
{broken_code}
"""

Error Summary:
"""
{error}
"""
'''.strip()
        )

    def generate_plot_code(self, viz_spec: Dict[str, Any]) -> str:
        prompt = self.gen_template.format(viz_spec=json.dumps(viz_spec, ensure_ascii=False, indent=2))
        raw = self.llm.chat(user_query=prompt, system_prompt=self.system_prompt)
        return _sanitize_code(raw)

    def patch_plot_code(self, broken_code: str, error: str) -> str:
        prompt = self.patch_template.format(broken_code=broken_code, error=error)
        raw = self.llm.chat(user_query=prompt, system_prompt=self.system_prompt)
        return _sanitize_code(raw)


# Module-level convenience functions expected by the caller
_default_agent: Optional[PlotCodeAgent] = None


def _get_default_agent() -> PlotCodeAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = PlotCodeAgent()
    return _default_agent


def gen_plot_code(viz_spec: Dict[str, Any]) -> str:
    """Generate a Python plotting script string from a VisualizationSpec dict."""
    agent = _get_default_agent()
    return agent.generate_plot_code(viz_spec)


def patch_code(broken_code: str, error: str) -> str:
    """Patch a broken plotting script using an error summary, returning corrected code."""
    agent = _get_default_agent()
    return agent.patch_plot_code(broken_code, error)


__all__ = [
    "PlotCodeAgent",
    "gen_plot_code",
    "patch_code",
]



# --------------------------- CLI and Utilities ---------------------------
DEMO_SPECS: Dict[str, Dict[str, Any]] = {
    "num_regions_openness_flow": {
        "id": "num_regions_openness_flow",
        "title": "Number of Cultural Regions by Openness and Information Flow",
        "data_source_category": "processed",
        "source_reference": "Number_of_Cultural_Regions_all_groups.json",
        "group_by_fields": ["openness", "interaction_range"],
        "aggregation": {"method": "count", "field": "region_count"},
        "suggested_visualization_type": "line",
        "why_this_figure": "Directly shows how varying openness and information flow parameters affects cultural region formation, addressing the research question's core.",
    },
    "polarization_openness_flow": {
        "id": "polarization_openness_flow",
        "title": "Global Polarization Index by Openness and Information Flow",
        "data_source_category": "processed",
        "source_reference": "Global_Polarization_Index_all_groups.json",
        "group_by_fields": ["openness", "interaction_range"],
        "aggregation": {"method": "mean", "field": "polarization_value"},
        "suggested_visualization_type": "line",
        "why_this_figure": "Quantifies fragmentation levels (distinct regions) normalized by population, revealing how openness and information flow jointly shape polarization.",
    },
    "region_size_distribution": {
        "id": "region_size_distribution",
        "title": "Region Size Distribution by Openness and Information Flow",
        "data_source_category": "processed",
        "source_reference": "Region_Size_Distribution_all_groups.json",
        "group_by_fields": ["openness", "interaction_range"],
        "aggregation": {"method": "distribution", "field": "region_size"},
        "suggested_visualization_type": "bar",
        "why_this_figure": "Shows how openness and information flow affect regional homogenization (large regions) vs fragmentation (small regions), complementing region count metrics.",
    },
}


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _resolve_viz_spec(
    *, spec_json: Optional[str], spec_file: Optional[str], demo_id: Optional[str]
) -> Dict[str, Any]:
    if spec_json:
        return json.loads(spec_json)
    if spec_file:
        return _load_json_file(Path(spec_file))
    if demo_id:
        if demo_id not in DEMO_SPECS:
            raise ValueError(f"Unknown demo_id: {demo_id}. Available: {list(DEMO_SPECS)}")
        return DEMO_SPECS[demo_id]
    raise ValueError("One of --spec-json, --spec-file, or --demo-id is required")


def _build_agent_from_args(args: argparse.Namespace) -> PlotCodeAgent:
    return PlotCodeAgent(config_name=args.config_name, config_path=args.config_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PlotCodeAgent CLI")
    parser.add_argument(
        "--config-name",
        default="openai-gpt4o",
        help="Model config name defined in config/model_config.json",
    )
    parser.add_argument(
        "--config-path",
        default="config/model_config.json",
        help="Path to model configuration JSON",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # gen subcommand
    gen_p = subparsers.add_parser("gen", help="Generate plotting code from a VisualizationSpec")
    gen_p.add_argument("--spec-json", help="VisualizationSpec as JSON string")
    gen_p.add_argument("--spec-file", help="Path to VisualizationSpec JSON file")
    gen_p.add_argument("--demo-id", help=f"One of: {list(DEMO_SPECS.keys())}")
    gen_p.add_argument("--out", help="Path to save generated .py code; prints to stdout if omitted")

    # patch subcommand
    patch_p = subparsers.add_parser("patch", help="Patch a broken plotting script using an error summary")
    patch_p.add_argument("--broken-file", required=True, help="Path to the broken .py script")
    patch_p.add_argument("--error-text", help="Inline error summary text")
    patch_p.add_argument("--error-file", help="Path to a file containing the error summary")
    patch_p.add_argument("--out", help="Path to save corrected .py code; prints to stdout if omitted")

    # test subcommand
    test_p = subparsers.add_parser("test", help="Generate code for demo specs and save to a directory")
    test_p.add_argument(
        "--out-dir",
        default="outputs/code_agent_demos",
        help="Directory to write generated demo scripts",
    )
    test_p.add_argument(
        "--demo-ids",
        nargs="*",
        help="Subset of demo ids to run; defaults to all",
    )

    args = parser.parse_args(argv)
    agent = _build_agent_from_args(args)

    if args.command == "gen":
        viz_spec = _resolve_viz_spec(
            spec_json=getattr(args, "spec_json", None),
            spec_file=getattr(args, "spec_file", None),
            demo_id=getattr(args, "demo_id", None),
        )
        code = agent.generate_plot_code(viz_spec)
        if args.out:
            _save_text(Path(args.out), code)
        else:
            sys.stdout.write(code)
            if not code.endswith("\n"):
                sys.stdout.write("\n")
        return 0

    if args.command == "patch":
        broken_path = Path(args.broken_file)
        with broken_path.open("r", encoding="utf-8") as f:
            broken_code = f.read()
        error_text = getattr(args, "error_text", None)
        if not error_text and getattr(args, "error_file", None):
            error_text = Path(args.error_file).read_text(encoding="utf-8")
        if not error_text:
            raise ValueError("Either --error-text or --error-file must be provided")
        fixed = agent.patch_plot_code(broken_code, error_text)
        if args.out:
            _save_text(Path(args.out), fixed)
        else:
            sys.stdout.write(fixed)
            if not fixed.endswith("\n"):
                sys.stdout.write("\n")
        return 0

    if args.command == "test":
        out_dir = Path(args.out_dir)
        demo_ids = args.demo_ids or list(DEMO_SPECS.keys())
        exit_code = 0
        for demo_id in demo_ids:
            if demo_id not in DEMO_SPECS:
                sys.stderr.write(f"Unknown demo id: {demo_id}\n")
                exit_code = 2
                continue
            spec = DEMO_SPECS[demo_id]
            try:
                code = agent.generate_plot_code(spec)
                _save_text(out_dir / f"{demo_id}.py", code)
            except Exception as exc:  # noqa: BLE001 - surface errors to user
                sys.stderr.write(f"Failed to generate for {demo_id}: {exc}\n")
                exit_code = 1
        return exit_code

    # Should not reach here
    return 1


def test_main() -> None:
    """Lightweight test runner that mimics `python -m ... code_agent test`."""
    _ = main(["test"])  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
