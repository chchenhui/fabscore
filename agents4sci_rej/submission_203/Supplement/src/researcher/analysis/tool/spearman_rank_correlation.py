# -*- coding: utf-8 -*-
"""
Spearman rank correlation for ordered trend

计算两列之间的 Spearman 秩相关系数，适用于评估有序（单调）趋势。
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def spearman_rank_correlation(
    data: pd.DataFrame,
    x: str,
    y: str
) -> Dict[str, Any]:
    df = data[[x, y]].dropna()
    if df.empty:
        return {"error": "数据为空或两列均为空。"}

    rho, pval = spearmanr(df[x].values.astype(float), df[y].values.astype(float))
    return {
        "rho": float(rho),
        "p_value": float(pval),
        "n": int(len(df)),
        "notes": ["Spearman 秩相关用于检验单调趋势关系（不要求线性）。"]
    }