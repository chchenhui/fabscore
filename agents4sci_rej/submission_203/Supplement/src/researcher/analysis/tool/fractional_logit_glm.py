# -*- coding: utf-8 -*-
"""
Fractional Logit GLM Tool

针对 (0,1) 区间比例型因变量的拟合：
- 使用 GLM Binomial + logit link
- 支持稳健标准误（HC0-HC3），默认 HC3
- 自动对边界值进行轻微裁剪避免数值问题
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def fractional_logit_glm(
    data: pd.DataFrame,
    formula: str,
    robust: Optional[str] = "HC3",
    clip_eps: float = 1e-6
) -> Dict[str, Any]:
    """
    参数:
        data: DataFrame
        formula: 形如 'y ~ x1 + x2 + C(group)'
        robust: None 或 'HC0'/'HC1'/'HC2'/'HC3'
        clip_eps: 对 y 的边界裁剪以避免 logit 数值问题

    返回:
        dict: GLM 摘要与指标
    """
    # 解析因变量名
    try:
        dv = formula.split("~")[0].strip()
    except Exception:
        dv = None

    df = data.copy()
    if dv and dv in df.columns:
        # 剪裁到 (eps, 1-eps)，避免 logit 无穷大
        df[dv] = np.clip(df[dv].astype(float), clip_eps, 1.0 - clip_eps)

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    result = model.fit(cov_type=robust) if robust else model.fit()

    # 系数摘要
    conf_int = result.conf_int()
    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    coef_summary = {}
    for name in params.index:
        ci_low, ci_high = conf_int.loc[name].values
        coef_summary[name] = {
            "coef": float(params[name]),
            "std_err": float(bse[name]),
            "p": float(pvalues[name]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    # 指标
    metrics = {
        "nobs": int(result.nobs),
        "aic": float(getattr(result, "aic", np.nan)),
        "deviance": float(getattr(result, "deviance", np.nan)),
        "pearson_chi2": float(getattr(result, "pearson_chi2", np.nan)),
        "llf": float(getattr(result, "llf", np.nan)),
    }

    return {
        "summary_text": str(result.summary()),
        "coefficients": coef_summary,
        "metrics": metrics,
        "model": "GLM (Binomial, logit)",
        "cov_type": robust if robust else "non-robust",
    }