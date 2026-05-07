"""
Analysis Request Agent

- Reads analysis_request.json
- Loads each method's apply_to dataset (JSON, full)
- Normalizes data
- Runs analyses using tool_registry TOOLS where applicable
- Writes results to social_dynamics/analysis/data/analysis_results.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools

import pandas as pd

# statsmodels for OLS regression
import statsmodels.formula.api as smf

try:
    # Prefer absolute import within project structure
    from src.researcher.analysis.agent.tool_registry import TOOLS
except Exception:
    # Fallback for relative execution
    from researcher.analysis.agent.tool_registry import TOOLS  # type: ignore


def _find_project_root() -> Path:
    """
    推断仓库根目录：当前文件位于 src/researcher/analysis/agent 下，
    根目录是向上 4 层（agent -> analysis -> researcher -> src -> ROOT）
    """
    here = Path(__file__).resolve()
    return here.parents[4]


def _resolve_path(root: Path, p: str, base_dir: Optional[Path] = None) -> Path:
    """
    将 Windows/相对路径规范化为仓库内的绝对路径。
    """
    p_norm = p.replace("\\", "/")
    candidate = Path(p_norm)
    if candidate.is_absolute():
        return candidate
    # 如果是仓库根下的相对路径（如 'projects/...', 'src/...', 'analysis/...')，用 root 解析，避免与 base_dir 重复拼接
    if p_norm.startswith("projects/") or p_norm.startswith("src/") or p_norm.startswith("analysis/"):
        return (root / p_norm).resolve()
    # 否则若提供了 base_dir，则相对于 base_dir 解析
    if base_dir is not None:
        return (base_dir / p_norm).resolve()
    # 回退到仓库根目录
    return (root / p_norm).resolve()


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_group_column(df: pd.DataFrame) -> Optional[str]:
    """
    从 DataFrame 中推断分组列名：优先寻找包含 'group' 或 'openness' 的列。
    """
    candidates = [c for c in df.columns if "group" in c.lower() or "openness" in c.lower()]
    if candidates:
        return candidates[0]
    # 其次尝试典型列名
    for name in ["group", "openness_group"]:
        if name in df.columns:
            return name
    return None


def _to_dataframe(data: Any, dv_name_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    将任意 JSON 数据转换为 DataFrame。
    """
    meta: Dict[str, Any] = {"normalized": False, "shape": None, "columns": None, "detected_group_col": None}

    # 优先处理顶层包装结构：{"file_info": {...}, "data": [ {...}, ... ]}
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        inner = data["data"]
        if len(inner) == 0:
            df = pd.DataFrame()
        elif isinstance(inner[0], dict):
            df = pd.DataFrame(inner)
        else:
            # data 为 list[primitive]
            dv_col = dv_name_hint or "value"
            df = pd.DataFrame({dv_col: inner})
        meta["normalized"] = True
        meta["shape"] = list(df.shape)
        meta["columns"] = list(df.columns)
        meta["detected_group_col"] = _infer_group_column(df)
        return df, meta

    if isinstance(data, list):
        # 期望为 list[dict]
        if len(data) == 0:
            df = pd.DataFrame()
        elif isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            # list of primitives -> 包装成单列 DataFrame
            df = pd.DataFrame({"value": data})
        meta["normalized"] = True

    elif isinstance(data, dict):
        # 可能是 group -> list[dict] 或 group -> list[value]
        rows: List[Dict[str, Any]] = []
        all_keys: set = set()

        for g, items in data.items():
            if isinstance(items, list) and len(items) > 0:
                first = items[0]
                if isinstance(first, dict):
                    for it in items:
                        row = dict(it)
                        row["group"] = g
                        rows.append(row)
                        all_keys.update(row.keys())
                else:
                    # list[value]
                    dv_col = dv_name_hint or "value"
                    for v in items:
                        rows.append({"group": g, dv_col: v})
                    all_keys.update({"group", dv_col})
            else:
                # 空列表或非列表，跳过
                continue

        df = pd.DataFrame(rows)
        meta["normalized"] = True
    else:
        # 其他类型：转为单列
        df = pd.DataFrame({"value": [data]})
        meta["normalized"] = True

    meta["shape"] = list(df.shape)
    meta["columns"] = list(df.columns)
    meta["detected_group_col"] = _infer_group_column(df)
    return df, meta


def _extract_group_samples(
    df: pd.DataFrame,
    groups: List[str],
    dv_col: str,
    group_col: Optional[str],
) -> Dict[str, List[float]]:
    """
    按请求中给定的 groups 顺序，从 DataFrame 中提取每组的 DV 样本列表。
    """
    if group_col is None:
        # 无显式分组列，尝试将每个 group 名作为列名（不常见但兜底）
        samples = {}
        for g in groups:
            if g in df.columns:
                series = pd.to_numeric(df[g], errors="coerce").dropna()
                samples[g] = series.tolist()
    else:
        samples = {}
        for g in groups:
            sub = df[df[group_col] == g]
            series = pd.to_numeric(sub[dv_col], errors="coerce").dropna()
            samples[g] = series.tolist()
    return samples


class RequestAnalysisAgent:
    """
    执行 analysis_request.json 中的分析计划：
    - 使用 tool_registry 中的工具（如 one_way_anova、independent_t_test）
    - 对未提供的工具（如 OLS 回归、成对 t 检验）做稳健实现
    """

    def __init__(self) -> None:
        self.root = _find_project_root()

    def run(
        self,
        request_json_path: str = "social_dynamics/analysis/data/analysis_request.json",
        output_json_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        request_abs = _resolve_path(self.root, request_json_path)
        base_abs: Optional[Path] = _resolve_path(self.root, base_dir) if base_dir else None
        request_obj = _load_json(request_abs)

        results: Dict[str, Any] = {
            "request_path": str(request_abs),
            "loaded_request": request_obj is not None,
            "methods_results": [],
            "errors": [],
            "meta": {"base_dir": str(base_abs) if base_abs else None},
        }

        if request_obj is None:
            results["errors"].append(f"analysis_request.json 不存在或无法解析: {request_abs}")
            return results

        dv_human = request_obj.get("dependent_variable")
        indeps_human = request_obj.get("independent_variables") or []
        groups = request_obj.get("groups") or []
        data_path_hint = request_obj.get("data_path")

        for method in request_obj.get("methods", []):
            method_name_req = method.get("name")
            apply_to = method.get("apply_to") or data_path_hint
            params = method.get("params") or {}

            # 兼容 apply_to 为字典的情况（如 {"data": "...", "groups": [...], "target": "..." }）
            if isinstance(apply_to, dict):
                apply_to_path = apply_to.get("data") or data_path_hint or ""
            else:
                apply_to_path = apply_to or ""

            apply_abs = _resolve_path(self.root, apply_to_path, base_dir=base_abs)
            data_obj = _load_json(apply_abs)

            method_result: Dict[str, Any] = {
                "method": method_name_req,
                "apply_to": str(apply_abs),
                "loaded_data": data_obj is not None,
                "status": "ok",
                "outputs": {},
                "warnings": [],
            }

            if data_obj is None:
                method_result["status"] = "skipped"
                method_result["warnings"].append(f"数据文件不存在或无法解析: {apply_abs}")
                results["methods_results"].append(method_result)
                continue

            formula = params.get("formula")
            dv_col_from_formula = None
            if isinstance(formula, str) and "~" in formula:
                left, _ = formula.split("~", 1)
                dv_col_from_formula = left.strip()

            df, df_meta = _to_dataframe(data_obj, dv_name_hint=dv_col_from_formula or "value")
            method_result["outputs"]["data_meta"] = df_meta

            try:
                if method_name_req == "anova":
                    # 优先使用 'data' 列作为 DV
                    dv_col = "data" if "data" in df.columns else (dv_col_from_formula or (df.columns[0] if len(df.columns) > 0 else "value"))
                    group_col = df_meta["detected_group_col"]
                    if group_col is None:
                        method_result["status"] = "error"
                        method_result["warnings"].append("无法推断分组列，无法执行 ANOVA。")
                    else:
                        samples_map = _extract_group_samples(df, groups, dv_col, group_col)
                        samples = [samples_map[g] for g in groups if g in samples_map and len(samples_map[g]) > 0]
                        if len(samples) < 2:
                            method_result["status"] = "error"
                            method_result["warnings"].append("可用组少于 2 或样本为空，无法执行 ANOVA。")
                        else:
                            func = TOOLS["one_way_anova"]["func"]
                            res = func(*samples)
                            method_result["outputs"]["anova"] = {
                                "f_statistic": getattr(res, "statistic", None),
                                "p_value": getattr(res, "pvalue", None),
                                "detail": str(res),
                                "groups": groups,
                                "sample_sizes": {g: len(samples_map.get(g, [])) for g in groups},
                            }

                elif method_name_req == "pairwise_t_tests":
                    correction = (params.get("correction") or "").lower()
                    dv_col = "data" if "data" in df.columns else (dv_col_from_formula or (df.columns[0] if len(df.columns) > 0 else "value"))
                    group_col = df_meta["detected_group_col"]

                    if group_col is None:
                        method_result["status"] = "error"
                        method_result["warnings"].append("无法推断分组列，无法执行成对 t 检验。")
                    else:
                        samples_map = _extract_group_samples(df, groups, dv_col, group_col)
                        pairs = list(itertools.combinations(groups, 2))
                        pair_results: List[Dict[str, Any]] = []
                        raw_p_values: List[float] = []

                        for g1, g2 in pairs:
                            if len(samples_map.get(g1, [])) == 0 or len(samples_map.get(g2, [])) == 0:
                                pair_results.append({"pair": [g1, g2], "status": "skipped", "reason": "样本为空或缺失"})
                                continue
                            func = TOOLS["independent_t_test"]["func"]
                            res = func(samples_map[g1], samples_map[g2])
                            pval = getattr(res, "pvalue", None)
                            raw_p_values.append(pval if pval is not None else float("nan"))
                            pair_results.append({
                                "pair": [g1, g2],
                                "t_statistic": getattr(res, "statistic", None),
                                "p_value": pval,
                                "n1": len(samples_map[g1]),
                                "n2": len(samples_map[g2]),
                            })

                        adjusted: List[float] = raw_p_values[:]
                        if correction == "bonferroni":
                            m = len(raw_p_values) if raw_p_values else 1
                            adjusted = [min(p * m, 1.0) if p is not None else None for p in raw_p_values]
                        elif correction:
                            method_result["warnings"].append(f"不支持的校正方法: {correction}，已返回未校正 p 值。")

                        idx = 0
                        for pr in pair_results:
                            if pr.get("status") == "skipped":
                                continue
                            pr["p_value_adjusted"] = adjusted[idx] if idx < len(adjusted) else None
                            idx += 1

                        method_result["outputs"]["pairwise_t_tests"] = {"pairs": pair_results, "correction": correction or "none"}

                elif method_name_req == "ols_regression":
                    if not formula:
                        method_result["status"] = "error"
                        method_result["warnings"].append("ols_regression 缺少 formula 参数。")
                    else:
                        try:
                            model = smf.ols(formula=formula, data=df).fit()
                            method_result["outputs"]["ols_regression"] = {
                                "formula": formula,
                                "params": {k: float(v) for k, v in model.params.items()},
                                "pvalues": {k: float(v) for k, v in model.pvalues.items()},
                                "rsquared": float(getattr(model, "rsquared", float('nan'))),
                                "rsquared_adj": float(getattr(model, "rsquared_adj", float('nan'))),
                                "nobs": int(getattr(model, "nobs", 0)),
                                "fvalue": float(getattr(model, "fvalue", float('nan'))),
                                "f_pvalue": float(getattr(model, "f_pvalue", float('nan'))),
                            }
                        except Exception as e:
                            method_result["status"] = "error"
                            method_result["warnings"].append(f"OLS 拟合失败: {e}")

                else:
                    if method_name_req in TOOLS:
                        spec = TOOLS[method_name_req]
                        func = spec["func"]
                        if "data" in spec.get("args", []):
                            out = func(df, **{k: v for k, v in params.items() if k != "data"})
                        else:
                            group_col = _infer_group_column(df)
                            dv_col = "data" if "data" in df.columns else (dv_col_from_formula or (df.columns[0] if len(df.columns) > 0 else "value"))
                            samples_map = _extract_group_samples(df, groups, dv_col, group_col)
                            out = func(*[samples_map[g] for g in groups if g in samples_map])
                        method_result["outputs"]["raw"] = str(out)
                    else:
                        method_result["status"] = "skipped"
                        method_result["warnings"].append(f"不支持的方法: {method_name_req}。")
            except Exception as e:
                method_result["status"] = "error"
                method_result["warnings"].append(f"方法执行异常: {e}")

            results["methods_results"].append(method_result)

        # 输出路径：优先写入 base_dir 下的 analysis/data/analysis_results.json
        if output_json_path is None:
            default_rel = "analysis/data/analysis_results.json"
            output_abs = _resolve_path(self.root, default_rel, base_dir=base_abs) if base_abs else _resolve_path(self.root, f"social_dynamics/{default_rel}")
        else:
            output_abs = _resolve_path(self.root, output_json_path, base_dir=base_abs)

        output_abs.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_abs.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            results["meta"]["output_path"] = str(output_abs)
            results["meta"]["write_ok"] = True
        except Exception as e:
            results["errors"].append(f"结果文件写入失败: {e}")
            results["meta"]["output_path"] = str(output_abs)
            results["meta"]["write_ok"] = False

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run analysis based on analysis_request.json")
    parser.add_argument("--request", type=str, default="social_dynamics/analysis/data/analysis_request.json",
                        help="Path to analysis_request.json (absolute or relative)")
    parser.add_argument("--base", type=str, default=None,
                        help="Base project directory to resolve relative paths in apply_to (e.g., /data/.../projects/social_dynamics)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (absolute or relative). Default: <base>/analysis/data/analysis_results.json")

    args = parser.parse_args()

    agent = RequestAnalysisAgent()
    res = agent.run(request_json_path=args.request, output_json_path=args.output, base_dir=args.base)

    print(json.dumps({
        "output_path": res.get("meta", {}).get("output_path"),
        "errors": res.get("errors", []),
        "method_statuses": [
            {"method": r.get("method"), "status": r.get("status"), "warnings": r.get("warnings")}
            for r in res.get("methods_results", [])
        ],
    }, ensure_ascii=False, indent=2))