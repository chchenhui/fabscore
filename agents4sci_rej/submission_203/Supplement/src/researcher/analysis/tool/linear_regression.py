# -*- coding: utf-8 -*-
"""
Linear Regression Tool (Alias of OLS)

提供易用的线性回归接口，基于现有 ols_regression + summarize_ols：
- 支持稳健标准误（HC0-HC3）
- 返回结构化摘要，便于记录与展示
"""

from typing import Dict, Any, Optional
import pandas as pd

from .ols_regression import ols_regression, summarize_ols

def linear_regression(
    data: pd.DataFrame,
    formula: str,
    robust: Optional[str] = "HC3",
    cluster_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    参数:
        data: DataFrame
        formula: 形如 'y ~ x1 + x2 + C(group)'
        robust: None 或 'HC0'/'HC1'/'HC2'/'HC3'
        cluster_col: 若提供，将使用聚类稳健协方差
    
    返回:
        dict: OLS 摘要（summary_text, coefficients, metrics, diagnostics）
    """
    result = ols_regression(data=data, formula=formula, robust=robust, cluster_col=cluster_col)
    summary = summarize_ols(result)
    summary["model"] = "OLS"
    summary["cov_type"] = robust if robust else "non-robust"
    return summary