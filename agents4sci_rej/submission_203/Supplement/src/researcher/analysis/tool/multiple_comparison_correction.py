# -*- coding: utf-8 -*-
"""
Multiple Comparison Correction Tool

提供事后多重比较的标准流程：
- Tukey HSD（适用于方差齐性且样本量均衡的情形）
- 基于两两 t 检验的通用多重校正（bonferroni/holm/sidak/fdr_bh 等）

返回结构化的校正结果，便于后续记录和可视化。
"""

from typing import Dict, Any, List, Tuple
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

def multiple_comparison_correction(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    method: str = "bonferroni",
    equal_var: bool = True
) -> Dict[str, Any]:
    """
    参数:
        data: DataFrame
        group_col: 分组列名
        value_col: 因变量列名
        method: 'tukey' 或 multipletests 支持的方法（'bonferroni', 'holm', 'sidak', 'fdr_bh' 等）
        equal_var: 两两 t 检验是否假定方差齐性

    返回:
        若 method == 'tukey':
            {
                "method": "tukey",
                "alpha": 0.05,
                "results": [ {group1, group2, meandiff, p_adj, lower, upper, reject}, ... ]
            }
        否则:
            {
                "method": method,
                "pairs": [ {group1, group2, n1, n2, t_stat, p_raw, p_adj, reject}, ... ]
            }
    """
    groups = sorted([g for g in data[group_col].dropna().unique()])
    if not groups or len(groups) < 2:
        return {"method": method, "error": "分组数量不足，至少需要两个组以执行多重比较。"}

    if method.lower() == "tukey":
        # Tukey HSD
        tukey = pairwise_tukeyhsd(endog=data[value_col], groups=data[group_col], alpha=0.05)
        # 解析 summary 表
        tbl = tukey.summary()
        header = tbl.data[0]
        rows = tbl.data[1:]
        # 标准列名约定
        rename = {
            "group1": "group1", "group2": "group2",
            "meandiff": "meandiff", "p-adj": "p_adj",
            "lower": "lower", "upper": "upper", "reject": "reject"
        }
        # 构造结构化结果
        results = []
        for r in rows:
            row = {rename.get(h, h): r[i] for i, h in enumerate(header)}
            # 强制转换成基本类型
            for k in ("meandiff", "p_adj", "lower", "upper"):
                if k in row and isinstance(row[k], (np.floating, np.integer)):
                    row[k] = float(row[k])
            if "reject" in row and isinstance(row["reject"], (np.bool_,)):
                row["reject"] = bool(row["reject"])
            results.append(row)
        return {"method": "tukey", "alpha": 0.05, "results": results}

    # 通用两两 t 检验 + 多重校正
    pairs: List[Dict[str, Any]] = []
    raw_pvals: List[float] = []

    for g1, g2 in itertools.combinations(groups, 2):
        x = data.loc[data[group_col] == g1, value_col].dropna().values
        y = data.loc[data[group_col] == g2, value_col].dropna().values
        if len(x) == 0 or len(y) == 0:
            pairs.append({
                "group1": g1, "group2": g2,
                "n1": int(len(x)), "n2": int(len(y)),
                "t_stat": None, "p_raw": None,
                "p_adj": None, "reject": None
            })
            continue

        t_stat, p_raw = ttest_ind(x, y, equal_var=equal_var)
        pairs.append({
            "group1": g1, "group2": g2,
            "n1": int(len(x)), "n2": int(len(y)),
            "t_stat": float(t_stat),
            "p_raw": float(p_raw),
            "p_adj": None,
            "reject": None
        })
        raw_pvals.append(float(p_raw))

    if raw_pvals:
        reject, p_adj, _, _ = multipletests(raw_pvals, method=method)
        # 将校正结果填回
        k = 0
        for i in range(len(pairs)):
            if pairs[i]["p_raw"] is None:
                continue
            pairs[i]["p_adj"] = float(p_adj[k])
            pairs[i]["reject"] = bool(reject[k])
            k += 1

    return {"method": method, "pairs": pairs}