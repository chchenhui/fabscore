# -*- coding: utf-8 -*-
"""
Kruskal–Wallis test by group column

从 data[group_col] 的每个组提取 data[value_col]，执行 Kruskal–Wallis 检验。
返回统计量、p 值，以及每组的基本统计摘要。
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import kruskal

def kruskal_wallis_by_group(
    data: pd.DataFrame,
    group_col: str,
    value_col: str
) -> Dict[str, Any]:
    groups = sorted([g for g in data[group_col].dropna().unique()])
    samples: List[np.ndarray] = []
    for g in groups:
        vals = data.loc[data[group_col] == g, value_col].dropna().values.astype(float)
        samples.append(vals)

    if len(samples) < 2:
        return {"error": "分组数量不足，至少需要两个组。", "groups": groups}

    stat, pval = kruskal(*samples)

    # 每组摘要
    summaries = []
    for g, vals in zip(groups, samples):
        if len(vals) == 0:
            summaries.append({"group": g, "n": 0, "mean": None, "median": None, "std": None})
        else:
            summaries.append({
                "group": g,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })

    return {
        "test": "Kruskal–Wallis H-test",
        "H_stat": float(stat),
        "p_value": float(pval),
        "k_groups": len(groups),
        "group_summaries": summaries
    }