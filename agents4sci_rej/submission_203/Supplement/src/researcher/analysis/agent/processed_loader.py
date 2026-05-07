from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


def _read_processed_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data") if isinstance(obj, dict) else obj
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def _infer_time_field(fields: List[str]) -> Optional[str]:
    for tf in ("step", "time", "t", "round"):
        if tf in fields:
            return tf
    return None


def _to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 规范 step 为整数
    if "step" in df.columns:
        try:
            df["step"] = df["step"].astype(str).str.replace("step_", "").astype(int)
        except Exception:
            pass
    return df


def build_loader_from_processed_dir(processed_dir: str, prefer_metrics: Optional[List[str]] = None):
    """
    返回一个 loader(plan)->context 的可调用对象。
    - 按 plan["dependent_variables"] 或 prefer_metrics 选择需要的 *_all_groups.json 文件
    - 汇总成 DataFrame，并根据不同方法准备参数：
      * t/非参检验：从两个 group 抽取一个度量的样本向量 sample1/2
      * one_way/kruskal：从多个 group 作为 *samples
      * 时间序列：按 group 聚合为宽表，或提供原始 df
    """

    def loader(plan: Mapping[str, Any]) -> Dict[str, Any]:
        families = plan.get("method_families") if isinstance(plan, Mapping) else None
        deps = plan.get("dependent_variables") if isinstance(plan, Mapping) else None
        metrics: List[str] = []
        if isinstance(deps, list) and deps:
            metrics = [str(deps[0])]
        elif isinstance(prefer_metrics, list) and prefer_metrics:
            metrics = [str(prefer_metrics[0])]
        else:
            # 兜底：选任意一个 *_all_groups.json（除 figures）
            for name in sorted(os.listdir(processed_dir)):
                if name.endswith("_all_groups.json") and not name.startswith("figures_analysis_combine"):
                    metrics = [os.path.splitext(name)[0].replace("_all_groups", "")]
                    break

        # 找到匹配文件（名字可能带空格与下划线差异，使用 name_to_file 映射）
        name_to_file: Dict[str, str] = {}
        for fname in os.listdir(processed_dir):
            if not fname.endswith("_all_groups.json"):
                continue
            base = os.path.splitext(fname)[0].replace("_all_groups", "")
            name_to_file[base] = fname
            # 也放一份去掉空格/下划线变体
            name_to_file[base.replace(" ", "_")] = fname
            name_to_file[base.replace("_", " ")] = fname

        frames: List[pd.DataFrame] = []
        for m in metrics:
            fname = name_to_file.get(m) or name_to_file.get(m.replace(" ", "_")) or name_to_file.get(m.replace("_", " "))
            if not fname:
                continue
            rows = _read_processed_file(os.path.join(processed_dir, fname))
            if not rows:
                continue
            df_m = _to_dataframe(rows)
            df_m["metric"] = m
            frames.append(df_m)

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        context: Dict[str, Any] = {"data": df}

        # 推断 group 与 time 字段
        group_col = None
        time_col = None
        if not df.empty:
            cols = df.columns.tolist()
            if "group_name" in cols:
                group_col = "group_name"
            time_col = _infer_time_field(cols)

        # 为常见方法准备参数
        if not df.empty:
            # 选择一个 metric 的末步各组作为样本
            metric_name = metrics[0] if metrics else None
            sub = df[df["metric"] == metric_name] if metric_name else df
            if time_col and time_col in sub.columns:
                # 取最后一步
                last_t = sub[time_col].max()
                sub_last = sub[sub[time_col] == last_t]
            else:
                sub_last = sub

            # 如果 data 为标量，字段在列 "data"；若为嵌套分布，此列为 dict，需要另行处理
            if "data" in sub_last.columns and group_col in sub_last.columns:
                # 标量型：按组取值
                scalar_rows = sub_last[sub_last["data"].map(lambda x: isinstance(x, (int, float)))]
                if not scalar_rows.empty:
                    # 取两个不同组作为 t 检验示例
                    groups = scalar_rows[group_col].unique().tolist()
                    if len(groups) >= 2:
                        g1, g2 = groups[:2]
                        sample1 = scalar_rows[scalar_rows[group_col] == g1]["data"].astype(float).tolist()
                        sample2 = scalar_rows[scalar_rows[group_col] == g2]["data"].astype(float).tolist()
                        context.update({
                            "sample1": sample1,
                            "sample2": sample2,
                        })
                    # one-way/kruskal: *samples
                    samples = [scalar_rows[scalar_rows[group_col] == g]["data"].astype(float).tolist() for g in groups]
                    if len(samples) >= 3:
                        context.update({"samples": samples})

                # 分布型：data 为 {xAxis, series}
                dist_rows = sub_last[sub_last["data"].map(lambda x: isinstance(x, dict) and "series" in x)]
                if not dist_rows.empty:
                    # 将 series 的和或均值作为一个数值特征
                    dist_rows = dist_rows.copy()
                    dist_rows["dist_mean"] = dist_rows["data"].map(lambda d: float(sum(d.get("series", []))) / max(1, len(d.get("series", []))))
                    groups = dist_rows[group_col].unique().tolist() if group_col else []
                    if len(groups) >= 2:
                        g1, g2 = groups[:2]
                        sample1 = dist_rows[dist_rows[group_col] == g1]["dist_mean"].astype(float).tolist()
                        sample2 = dist_rows[dist_rows[group_col] == g2]["dist_mean"].astype(float).tolist()
                        context.update({
                            "sample1": sample1 or context.get("sample1"),
                            "sample2": sample2 or context.get("sample2"),
                        })

            # 时间序列：为相关/趋势等准备
            if time_col and group_col and "data" in df.columns:
                scalar_df = df[df["data"].map(lambda x: isinstance(x, (int, float)))][[time_col, group_col, "data"]].copy()
                if not scalar_df.empty:
                    # pivot 为宽表：行=时间，列=组
                    wide = scalar_df.pivot_table(index=time_col, columns=group_col, values="data", aggfunc="mean").sort_index()
                    # 取两列作为 x,y
                    cols = [c for c in wide.columns if wide[c].notna().any()]
                    if len(cols) >= 2:
                        context.update({
                            "x": wide[cols[0]].dropna().values.tolist(),
                            "y": wide[cols[1]].dropna().values.tolist(),
                        })
        return context

    return loader


__all__ = ["build_loader_from_processed_dir"]
