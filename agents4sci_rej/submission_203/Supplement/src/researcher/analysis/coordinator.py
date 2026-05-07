#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OneSim Researcher Analysis Coordinator (E2E Pipeline)

按顺序执行四个阶段：
1) collect_scene_metrics_simple: 收集各组/各步的指标（每步每类仅保留最新文件）
2) stage1: 自动选图、生成绘图代码并运行得到三张 PNG（保存到 figures 目录）
3) stage2: 基于图与处理后的数据生成 figure_analysis.json
4) stage3: 使用 EnhancedStatAgentLLMAdapter 输出 data_analysis.json（严格 JSON）

输入仅需一个 project_name，项目路径解析逻辑与 stage3.py 保持一致。
"""

from pathlib import Path
import sys
import os
import json
import argparse
from typing import Dict, Any, Optional, List


def _repo_root() -> Path:
    # 当前文件位于 src/researcher/analysis/coordinator.py
    # 仓库根目录为 parents[4]
    return Path(__file__).resolve().parents[4]


def _resolve_project_dir(project_name: str) -> Path:
    repo = _repo_root()
    cand: List[Path] = []

    p = Path(project_name)
    if p.is_absolute():
        cand.append(p)

    # relative candidate: projects/{project_name}
    cand.append(repo / "projects" / project_name)

    for c in cand:
        if c.exists() and c.is_dir():
            return c.resolve()

    tried = [str(c) for c in cand]
    raise FileNotFoundError(f"Cannot locate project '{project_name}'. Expected under: {tried}")


def coordinate_analysis(project_name: str, config_name: str = "openai-gpt4o", latest_runs: int = 1) -> Dict[str, Any]:
    """
    统一协调器：顺序执行四个阶段。
    - project_name: 项目名（projects/{project_name}）
    - config_name: LLM 配置名（用于 stage3，默认 openai-gpt4o）
    - latest_runs: 收集阶段每组保留最近 N 次运行（默认 1）

    返回各阶段产物的路径摘要。
    """
    # 1) 解析仓库与项目路径；确保 src 在 sys.path
    repo = _repo_root()
    src_path = repo / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from loguru import logger

    project_dir = _resolve_project_dir(project_name)
    logger.info(f"[coordinator] Using project dir: {project_dir}")

    # 统一路径
    groups_dir = project_dir / "groups"
    analysis_dir = project_dir / "analysis"
    data_dir = analysis_dir / "data"
    processed_dir = data_dir / "processed"
    figures_dir = analysis_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 将 config_name 和统一的模型配置路径传递给下游
    os.environ["ONESIM_MODEL_NAME"] = config_name
    os.environ["ONESIM_MODEL_CONFIG"] = "config/model_config.json"
    os.environ["ONESIM_MODEL_CONFIG_PATH"] = "config/model_config.json"

    # 2) 收集数据（仅保留每步每类最新文件；支持最新 N 次运行）
    logger.info("[coordinator] Stage 1/4: collect_scene_metrics_simple -> processed JSON")
    from researcher.analysis.agent.collect_scene_metrics_simple import SimpleSceneMetricsCollector

    collector = SimpleSceneMetricsCollector(
        groups_base_path=str(groups_dir),
        output_dir=str(processed_dir),
        latest_runs=int(latest_runs),
    )
    collector.save()  # 产出 *_all_groups.json 和 collection_summary_all_groups.json
    logger.info("[coordinator] Data collection completed")

    # 3) 生成三张图（保存到 figures_dir）
    logger.info("[coordinator] Stage 2/4: stage1 -> generate figures (PNG)")
    from researcher.analysis.agent.stage1 import run_stage1

    scene_info_path = project_dir/ "base_scenario" / "scene_info.json"
    workflow_state_path = project_dir / "workflow_state.json"

    # 如果缺失仍继续（stage1 内部有回退逻辑）
    if not scene_info_path.exists():
        logger.warning(f"[coordinator] scene_info not found: {scene_info_path}")
    if not workflow_state_path.exists():
        logger.warning(f"[coordinator] workflow_state not found: {workflow_state_path}")

    stage1_paths = {
        "scene_info": str(scene_info_path),
        "workflow_state": str(workflow_state_path),
        "processed_dir": str(processed_dir),
        "outputs_dir": str(figures_dir),  # 将 PNG 放在 figures 目录，方便后续 stage2
    }
    fig_paths = run_stage1(stage1_paths, cfg=None)
    logger.info(f"[coordinator] Stage1 generated figures: {[str(p) for p in fig_paths]}")

    # 4) 基于图与数据生成 figure_analysis.json
    logger.info("[coordinator] Stage 3/4: stage2 -> figure_analysis.json")
    from researcher.analysis.agent.stage2 import run_stage2

    # 加载 workflow_state 作为 stage2 的上下文
    stage1_context: Dict[str, Any] = {"project_name": project_name}
    try:
        if workflow_state_path.exists():
            with workflow_state_path.open("r", encoding="utf-8") as f:
                stage1_context = json.load(f)
            stage1_context.setdefault("project_name", project_name)
    except Exception as e:
        logger.warning(f"[coordinator] failed to read workflow_state.json: {e}")

    figure_analysis_path = run_stage2(
        stage1_context=stage1_context,
        paths={
            "figures_dir": str(figures_dir),
            "data_processed_dir": str(processed_dir),
            "project_name": project_name,
        },
        cfg={},
    )
    logger.info(f"[coordinator] Stage2 output: {figure_analysis_path}")

    # 5) 使用 LLM 输出综合 data_analysis.json（严格 JSON）
    logger.info("[coordinator] Stage 4/4: stage3 -> data_analysis.json (LLM)")
    from researcher.analysis.agent.stage3 import stage3 as run_stage3

    # 执行 stage3（内部会打印保存路径）
    run_stage3(project_name, config_name=config_name)

    # 汇总返回
    summary = {
        "project_dir": str(project_dir),
        "paths": {
            "groups_dir": str(groups_dir),
            "processed_dir": str(processed_dir),
            "figures_dir": str(figures_dir),
        },
        "artifacts": {
            "figures": [str(p) for p in fig_paths],
            "figure_analysis_json": str(figure_analysis_path),
            # data_analysis.json 的精确路径由 stage3 内部保存与打印（analysis/data/data_analysis.json）
            "data_analysis_json_dir": str(data_dir),
        },
        "model_config": config_name,
    }
    logger.info("[coordinator] Pipeline completed successfully")
    return summary


def main():
    parser = argparse.ArgumentParser(description="OneSim Researcher: End-to-End analysis coordinator (project_name only).")
    parser.add_argument("project_name", help="Project name under projects/ or absolute path to the project directory")
    parser.add_argument("--config-name", default="openai-gpt4o", help="Model config name for stage3 (default: openai-gpt4o)")
    parser.add_argument("--latest-runs", type=int, default=1, help="Collect latest N runs per group (default: 1)")
    args = parser.parse_args()

    summary = coordinate_analysis(args.project_name, config_name=args.config_name, latest_runs=args.latest_runs)
    # 输出一个简要 JSON 摘要，便于在管线或脚本中使用
    try:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    main()