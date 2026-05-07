import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# 兼容包导入（优先绝对导入，失败时使用相对）
try:
    from researcher.analysis.agent.agent_client import SimpleChatLLM
except Exception:
    try:
        from .agent_client import SimpleChatLLM  # 当作为包运行时
    except Exception as e:
        raise RuntimeError(f"Failed to import SimpleChatLLM: {e}")

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _find_project_root() -> Path:
    """
    推断仓库根目录：当前文件位于 src/researcher/analysis/agent 下，
    根目录是向上 4 层（agent -> analysis -> researcher -> src -> ROOT）
    """
    here = Path(__file__).resolve()
    # parents[0]=agent, [1]=analysis, [2]=researcher, [3]=src, [4]=ROOT
    return here.parents[4]

def _resolve_project_dir(project_arg: str) -> Path:
    """
    支持传入：
    - 'social_dynamics'
    - 'projects/social_dynamics'
    - 绝对路径
    """
    if not project_arg:
        raise ValueError("project 参数不能为空")

    if os.path.isabs(project_arg) and os.path.isdir(project_arg):
        return Path(project_arg)

    root = _find_project_root()

    # 候选路径
    candidates = [
        root / project_arg,  # e.g., ROOT/social_dynamics
        root / Path(project_arg.strip("/")),  # e.g., ROOT/projects/social_dynamics
        root / "projects" / project_arg,  # e.g., ROOT/projects/social_dynamics
    ]
    for c in candidates:
        if c.is_dir():
            return c

    # 最后兜底：如果传入 'projects/social_dynamics' 但真实在 'social_dynamics'
    if "projects/" in project_arg:
        maybe = root / project_arg.split("projects/", 1)[1]
        if maybe.is_dir():
            return maybe

    raise FileNotFoundError(f"无法解析项目目录: {project_arg}")

def _extract_groups(project_summary: Optional[Dict[str, Any]], workflow_state: Optional[Dict[str, Any]]) -> List[str]:
    groups: List[str] = []
    # 优先从 experiment_design.group_details
    try:
        details = (project_summary or {}).get("experiment_design", {}).get("group_details", [])
        for d in details:
            gid = d.get("group_id")
            if isinstance(gid, str):
                groups.append(gid)
    except Exception:
        pass

    # 次选从 workflow_state.simulation_results.simulation_results.simulation_details 的键
    if not groups and isinstance(workflow_state, dict):
        try:
            sim_details = workflow_state.get("simulation_results", {}).get("simulation_results", {}).get("simulation_details", {})
            if isinstance(sim_details, dict):
                groups = list(sim_details.keys())
        except Exception:
            pass

    return groups

def _extract_dependent_variable(scene_info: Optional[Dict[str, Any]], research_question: Optional[str]) -> str:
    # 优先 scene_info.dependent_variable
    try:
        dv = (scene_info or {}).get("dependent_variable")
        if isinstance(dv, str) and dv.strip():
            return dv.strip()
    except Exception:
        pass

    # 从研究问题中简单解析（示例中为 "number of cultural regions"）
    if isinstance(research_question, str):
        # 常见度量短语捕获
        m = re.search(r"number of [A-Za-z ]+regions", research_question, flags=re.IGNORECASE)
        if m:
            return m.group(0)

        # 备选：Cultural Homogeneity Index
        if "homogeneity" in research_question.lower():
            return "Cultural Homogeneity Index"

    # 最后兜底
    return "number of cultural regions"

def _extract_independent_variables(research_question: Optional[str]) -> List[str]:
    vars_: List[str] = []
    if isinstance(research_question, str):
        # 抓取 "degree of X" 模式
        for m in re.finditer(r"degree of ([A-Za-z ]+)", research_question, flags=re.IGNORECASE):
            phrase = m.group(0).strip()
            if phrase not in vars_:
                vars_.append(phrase)

        # 额外补全：如果提到了 "information flow" 但没带 "degree of"
        if "information flow" in research_question.lower() and not any("information flow" in v.lower() for v in vars_):
            vars_.append("degree of information flow")

    # 兜底：至少有一个
    if not vars_:
        vars_.append("degree of openness")
    return vars_

def _find_data_file(project_dir: Path) -> Optional[str]:
    """
    优先返回“处理后的分组数据”的相对路径，示例：
    analysis/data/processed/Cultural_Homogeneity_Index_all_groups.json
    如果没有找到，则返回 None（不再回退到 data_analysis.json）。
    """
    root = _find_project_root()

    # 首选：collection_summary_all_groups.json（若存在）
    coll_path = project_dir / "analysis" / "data" / "collection_summary_all_groups.json"
    if coll_path.exists():
        try:
            payload = _read_json(coll_path) or {}
            data_file = payload.get("data_file")
            if isinstance(data_file, str) and data_file.strip():
                return data_file.strip()
        except Exception:
            pass

    # 备选：常见的 processed 路径
    candidates = [
        project_dir / "analysis" / "data" / "processed" / "Cultural_Homogeneity_Index_all_groups.json",
        project_dir / "analysis" / "data" / "processed" / "Cultural_Homogeneity_Index.json",
        project_dir / "analysis" / "data" / "Cultural_Homogeneity_Index_all_groups.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return str(p.relative_to(root))
            except Exception:
                return str(p)

    # 通配搜索 *_all_groups.json
    try:
        for p in (project_dir / "analysis").rglob("*_all_groups.json"):
            if p.is_file():
                try:
                    return str(p.relative_to(root))
                except Exception:
                    return str(p)
    except Exception:
        pass

    # 不再回退到 data_analysis.json，返回 None
    return None

def build_context(project_dir: Path) -> Dict[str, Any]:
    root = _find_project_root()

    # 读取文件
    workflow_state = _read_json(project_dir / "workflow_state.json") or {}
    project_summary = _read_json(project_dir / "project_summary.json") or {}
    scene_info = _read_json(project_dir / "analysis" / "data" / "scene_info.json")  # 可能不存在

    research_paradigm = workflow_state.get("research_paradigm")
    research_question = workflow_state.get("research_question")

    groups = _extract_groups(project_summary, workflow_state)
    dependent_variable = _extract_dependent_variable(scene_info, research_question)
    data_file = _find_data_file(project_dir)

    # category：若 scene_info 有指标说明则用其名称，否则用 dependent_variable
    category = None
    try:
        category = (scene_info or {}).get("metric_name") or (scene_info or {}).get("category")
    except Exception:
        category = None
    if not category:
        category = dependent_variable

    # 构造 context（按你示例），并补充 dependent_variable
    context: Dict[str, Any] = {
        "research_paradigm": research_paradigm,
        "research_question": research_question,
        "category": category,
        "data_file": data_file,
        "groups": groups,
        "dependent_variable": dependent_variable,  # 新增
    }
    return context

def compose_system_prompt() -> str:
    return (
        "Return ONLY valid JSON with keys:\n"
        "- model_name (statistical test to apply, e.g., ANOVA, t-test, regression)\n"
        "- data_path (relative path to input file)\n"
        "- research_purpose (from research_question)\n"
        "- groups (list of experimental groups)\n"
        "- dependent_variable (metric under study)\n"
        "- independent_variables (list of factors from research_question)\n"
        "- methods (AT LEAST 3). Each method MUST include:\n"
        "    - name (e.g., one_way_anova, pairwise_t_tests, ols_regression)\n"
        "    - apply_to (which subset of data to use; reference the data_path and specific groups)\n"
        "    - params (key parameters, e.g., alpha, correction, formula)\n"
        "    - description (a brief description of what this method examines in context)\n"
        "If 'data_file' is present in the input context, use it as 'data_path'. If not, propose a reasonable processed file path like 'analysis/data/processed/Cultural_Homogeneity_Index_all_groups.json'."
    )

def plan_analysis_request(context: Dict[str, Any], llm: SimpleChatLLM, temperature: float = 0.7) -> Dict[str, Any]:
    system_prompt = compose_system_prompt()
    # user_query：把 context json.dumps 传进去
    user_query = json.dumps(context, ensure_ascii=False)

    # 调用 LLM
    try:
        analysis_request = llm.chat_json(
            user_query=user_query,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        # 轻量校验：必须包含必要键，且 methods 至少 3 个
        required_keys = {"model_name", "data_path", "research_purpose", "groups", "dependent_variable", "independent_variables", "methods"}
        if not (isinstance(analysis_request, dict) and required_keys.issubset(analysis_request.keys())):
            raise ValueError("LLM 返回 JSON 缺少必要键，进入回退策略。")
        methods = analysis_request.get("methods")
        if not (isinstance(methods, list) and len(methods) >= 3):
            raise ValueError("LLM 返回的 methods 少于 3 项，进入回退策略。")
        return analysis_request
    except Exception:
        # 回退策略：基于 context 构造更详细的 3+ 方法
        rq = context.get("research_question") or ""
        indeps = _extract_independent_variables(rq)
        groups = context.get("groups") or []
        dv = context.get("dependent_variable") or (context.get("category") or "metric")
        data_file = context.get("data_file")
        # 推荐的典型 processed 路径（若 data_file 未提供时作为提示性 apply_to）
        suggested_path = "analysis/data/processed/Cultural_Homogeneity_Index_all_groups.json"

        # 简单选择总体模型：组数>2 用 ANOVA，否则 t-test；若无组信息则回归
        if isinstance(groups, list) and len(groups) > 2:
            model_name = "ANOVA"
        elif isinstance(groups, list) and len(groups) == 2:
            model_name = "t-test"
        else:
            model_name = "regression"

        methods = [
            {
                "name": "one_way_anova",
                "apply_to": {
                    "data": (data_file or suggested_path),
                    "groups": groups,
                    "target": dv
                },
                "params": {
                    "alpha": 0.05,
                    "assumption": "independent groups; approximate normality or large-sample robustness"
                },
                "description": "Compare the dependent variable across all experimental groups to test if group means differ significantly."
            },
            {
                "name": "pairwise_t_tests",
                "apply_to": {
                    "data": (data_file or suggested_path),
                    "groups": groups,
                    "target": dv
                },
                "params": {
                    "correction": "bonferroni",
                    "alpha": 0.05,
                    "test_type": "two-sided",
                },
                "description": "Conduct pairwise comparisons between groups for the dependent variable with multiple-testing correction."
            },
            {
                "name": "ols_regression",
                "apply_to": {
                    "data": (data_file or suggested_path),
                    "groups": groups,
                    "target": dv
                },
                "params": {
                    "formula": f"{dv} ~ " + " + ".join(indeps) if indeps else f"{dv} ~ 1",
                    "robust_se": True
                },
                "description": "Model the relationship between the dependent variable and independent factors using linear regression."
            }
        ]

        # 可选地增加一个非参数相关性（如果独立变量中包含有序度量）
        if indeps:
            methods.append(
                {
                    "name": "spearman_correlation",
                    "apply_to": {
                        "data": (data_file or suggested_path),
                        "groups": groups,
                        "target": dv
                    },
                    "params": {
                        "variables": indeps,
                        "alpha": 0.05
                    },
                    "description": "Assess monotonic relationships between the dependent variable and independent variables."
                }
            )

        return {
            "model_name": model_name,
            "data_path": (data_file or ""),  # 若无法确定，则留空字符串（不再回退到 data_analysis.json）
            "research_purpose": rq,
            "groups": groups,
            "dependent_variable": dv,
            "independent_variables": indeps,
            "methods": methods,
        }

def save_analysis_request(project_dir: Path, analysis_request: Dict[str, Any], output_path: Optional[str] = None) -> Path:
    """
    默认保存到：{project_dir}/analysis/data/analysis_request.json
    若指定 output_path 则使用该路径（相对或绝对）。
    """
    if output_path:
        out = Path(output_path)
        if not out.is_absolute():
            out = _find_project_root() / out
    else:
        out = project_dir / "analysis" / "data" / "analysis_request.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(analysis_request, f, ensure_ascii=False, indent=2)
    return out

def main():
    parser = argparse.ArgumentParser(description="Stage3 Planner Agent: 生成分析请求 JSON（analysis_request.json）")
    parser.add_argument("--project", type=str, required=True, help="项目名或路径（如 social_dynamics 或 projects/social_dynamics）")
    parser.add_argument("--config-name", type=str, default="openai-gpt4o", help="LLM 配置名")
    parser.add_argument("--config-path", type=str, default="config/model_config.json", help="LLM 配置文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认保存到项目的 analysis/data/analysis_request.json）")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM 温度")
    args = parser.parse_args()

    project_dir = _resolve_project_dir(args.project)

    # 构造 context
    context = build_context(project_dir)

    # 将 research_question 中的因素解析后补全给 LLM（作为 context 的辅助）
    # 这部分可以不改 context 的原始字段，直接供 LLM参考或回退使用
    # 这里保持最小偏置：仅在回退时使用
    # 初始化 LLM
    llm = SimpleChatLLM(config_name=args.config_name, config_path=args.config_path)

    # 生成 analysis_request
    analysis_request = plan_analysis_request(context, llm, temperature=args.temperature)

    # 保存输出
    out_path = save_analysis_request(project_dir, analysis_request, output_path=args.output)
    print(f"分析请求已保存: {out_path}")

if __name__ == "__main__":
    main()