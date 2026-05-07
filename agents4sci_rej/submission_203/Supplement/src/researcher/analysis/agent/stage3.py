#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage3 runner（重写版）：
- 输入：仅 project_name
- 模型：默认使用 config-name=openai-gpt4o（对应 gpt-4o）
- 流程：
  1) 规划器（Planner）：根据项目上下文生成 analysis_request.json
     输出到：/projects/{project_name}/analysis/analysis_request.json
  2) 执行器（Executor）：读取 analysis_request.json 执行数据分析
     输出到：/projects/social_dynamics/analysis/data_analysis.json

说明：
- 完全移除 EnhancedStatAgentLLMAdapter 相关依赖，避免导入时的 SyntaxError 问题。
- 执行器在解析数据路径时，会优先基于项目目录（base_dir）解析相对路径，如 "analysis/data/processed/..."。
"""

from pathlib import Path
import sys
import argparse
import os


def _repo_root() -> Path:
    """
    当前文件位于 src/researcher/analysis/agent/stage3.py
    仓库根目录为向上 4 层：agent -> analysis -> researcher -> src -> ROOT
    """
    return Path(__file__).resolve().parents[4]


def _resolve_project_dir(project_name: str) -> Path:
    """
    解析项目目录：
    - 支持传入绝对路径
    - 支持 'projects/{project_name}'
    - 也兼容仓库根下的 '{project_name}'（如存在）
    """
    repo = _repo_root()
    candidates = []

    p = Path(project_name)
    if p.is_absolute() and p.exists():
        candidates.append(p)

    candidates.append(repo / "projects" / project_name)
    candidates.append(repo / project_name)

    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()

    tried = [str(c) for c in candidates]
    raise FileNotFoundError(f"Cannot locate project '{project_name}'. Tried: {tried}")


def stage3(project_name: str, config_name: str = "openai-gpt4o") -> None:
    """
    执行 Stage3（Planner + Executor）
    """
    # 确保可以导入 src/*
    repo = _repo_root()
    src_path = repo / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # 显式设置模型配置路径
    os.environ.setdefault("ONESIM_MODEL_CONFIG_PATH", str(repo / "config" / "model_config.json"))

    # 导入规划器与执行器
    from researcher.analysis.agent.stage3_planner_agent import (
        build_context,
        plan_analysis_request,
        save_analysis_request,
    )
    from researcher.analysis.agent.request_analysis_agent import RequestAnalysisAgent
    from researcher.analysis.agent.agent_client import SimpleChatLLM

    # 解析项目目录
    project_dir = _resolve_project_dir(project_name)
    print(f"[stage3] Using project dir: {project_dir}")

    # 1) 构造分析请求（Planner）
    context = build_context(project_dir)
    llm = SimpleChatLLM(config_name=config_name, config_path=str(repo / "config" / "model_config.json"))
    analysis_request = plan_analysis_request(context, llm, temperature=0.7)

    # 保存到项目的 analysis/analysis_request.json（注意不是 analysis/data/analysis_request.json）
    out_req_path = repo / "projects" / project_name / "analysis" / "analysis_request.json"
    saved_req_path = save_analysis_request(project_dir, analysis_request, output_path=str(out_req_path))
    print(f"[stage3] Analysis request saved to: {saved_req_path}")

    # 2) 执行分析（Executor）固定输出到 social_dynamics/analysis/data_analysis.json
    agent = RequestAnalysisAgent()
    out_analysis_path = repo / "projects" / project_name / "analysis" / "data_analysis.json"
    res = agent.run(
        request_json_path=str(saved_req_path),
        output_json_path=str(out_analysis_path),
        base_dir=str(project_dir),  # 传入项目目录，让相对路径（如 analysis/data/processed/...）正确解析到该项目
    )
    print(f"[stage3] Analysis results written to: {out_analysis_path}")
    # 可选：打印方法状态摘要
    try:
        statuses = [
            {"method": r.get("method"), "status": r.get("status"), "warnings": r.get("warnings")}
            for r in res.get("methods_results", [])
        ]
        print(f"[stage3] Methods summary: {statuses}")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 runner: planner + request executor")
    parser.add_argument("project_name", help="Project name under projects/ or absolute path")
    args = parser.parse_args()
    stage3(args.project_name)


if __name__ == "__main__":
    main()