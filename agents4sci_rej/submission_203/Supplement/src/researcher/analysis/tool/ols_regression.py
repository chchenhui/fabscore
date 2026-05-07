# -*- coding: utf-8 -*-
"""
OLS Regression Tool

提供基于公式接口的普通最小二乘回归（OLS），支持：
- 稳健标准误（HC0-HC3）
- 聚类稳健（cluster）
- 多重共线性评估（VIF）
- 常见诊断（Breusch-Pagan、White、Durbin-Watson、残差正态性）

示例：
    from researcher.analysis.tool import ols_regression, summarize_ols

    result = ols_regression(df, "y ~ x1 + x2", robust="HC3")
    summary = summarize_ols(result)
    print(summary["summary_text"])
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson


def ols_regression(
    data: pd.DataFrame,
    formula: str,
    robust: Optional[str] = None,
    cluster_col: Optional[str] = None,
    missing: str = "drop",
):
    """
    拟合 OLS 回归模型（基于公式）

    参数:
        data: Pandas DataFrame
        formula: 公式字符串，例如 'y ~ x1 + x2 + C(group)'
        robust: 稳健标准误选项，可为 None, 'HC0', 'HC1', 'HC2', 'HC3'
        cluster_col: 若提供，将使用聚类稳健标准误（cov_type='cluster'）
        missing: 缺失值处理方式，默认 'drop'

    返回:
        statsmodels 回归结果对象 (RegressionResultsWrapper)

    注意:
        - 公式中用到的列必须存在于 data 中
        - 分类变量可使用 C(col) 处理
    """
    model = smf.ols(formula=formula, data=data, missing=missing)

    # 选择协方差类型
    if cluster_col:
        cov_type = "cluster"
        cov_kwds = {"groups": data[cluster_col]}
        result = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    elif robust in {None, "HC0", "HC1", "HC2", "HC3"}:
        if robust is None:
            result = model.fit()
        else:
            result = model.fit(cov_type=robust)
    else:
        raise ValueError("robust 仅支持 None, 'HC0', 'HC1', 'HC2', 'HC3'。")

    return result


def summarize_ols(result) -> Dict:
    """
    提取 OLS 结果的关键摘要信息（便于序列化/记录）

    返回:
        dict，包括：
            - summary_text: 文本摘要
            - coefficients: 每个参数的系数/标准误/置信区间/t/p
            - metrics: R^2, adj R^2, F pvalue, nobs, AIC, BIC
            - diagnostics: Durbin-Watson, 残差正态性(Anderson-Darling)
    """
    conf_int = result.conf_int()
    params = result.params
    bse = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues

    coef_summary = {}
    for name in params.index:
        ci_low, ci_high = conf_int.loc[name].values
        coef_summary[name] = {
            "coef": float(params[name]),
            "std_err": float(bse[name]),
            "t": float(tvalues[name]),
            "p": float(pvalues[name]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    # 残差正态性（Anderson-Darling）
    ad_stat, ad_p = normal_ad(result.resid)

    summary = {
        "summary_text": str(result.summary()),
        "coefficients": coef_summary,
        "metrics": {
            "r_squared": float(result.rsquared),
            "adj_r_squared": float(result.rsquared_adj),
            "f_pvalue": float(result.f_pvalue) if result.f_pvalue is not None else None,
            "nobs": int(result.nobs),
            "aic": float(result.aic) if result.aic is not None else None,
            "bic": float(result.bic) if result.bic is not None else None,
        },
        "diagnostics": {
            "durbin_watson": float(durbin_watson(result.resid)),
            "residual_normality_ad_stat": float(ad_stat),
            "residual_normality_ad_p": float(ad_p),
        },
    }
    return summary


def compute_vif(data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    计算特征列的 VIF（方差膨胀因子），用于评估多重共线性

    参数:
        data: DataFrame
        feature_cols: 仅包含自变量特征列（不包含目标变量）

    返回:
        dict: {col -> VIF}
    """
    X = data[feature_cols].copy()

    # 添加常量项
    X_const = sm.add_constant(X, has_constant="add")

    vifs = {}
    # statsmodels 的 VIF 不包含常量项
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vifs[col] = float(variance_inflation_factor(X_const.values, i))
    return vifs


def breusch_pagan_test(result) -> Dict[str, float]:
    """
    Breusch-Pagan 异方差检验

    返回:
        dict 包含 Lagrange multiplier, p-value, f-value, f p-value
    """
    resid = result.resid
    # 使用原始设计矩阵（不含因变量）
    exog = result.model.exog
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, exog)
    return {
        "lm_stat": float(lm),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(fvalue),
        "f_pvalue": float(f_pvalue),
    }


def white_test(result) -> Dict[str, float]:
    """
    White 异方差检验
    """
    resid = result.resid
    exog = result.model.exog
    lm, lm_pvalue, fvalue, f_pvalue = het_white(resid, exog)
    return {
        "lm_stat": float(lm),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(fvalue),
        "f_pvalue": float(f_pvalue),
    }


def durbin_watson_stat(result) -> float:
    """
    Durbin-Watson 自相关统计量（残差）
    """
    return float(durbin_watson(result.resid))


def run_example():
    """
    简单示例：生成数据并拟合 OLS
    """
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "group": np.random.choice(["A", "B"], size=n)
    })
    # 目标变量：x1, x2, group 对 y 的影响
    df["y"] = 2.0 + 1.5 * df["x1"] - 0.8 * df["x2"] + (df["group"] == "B").astype(float) * 1.2 + np.random.normal(0, 1, n)

    # OLS 回归（含分类变量）
    result = ols_regression(df, "y ~ x1 + x2 + C(group)", robust="HC3")
    summary = summarize_ols(result)

    print(summary["summary_text"])

    # 计算 VIF（仅数值自变量）
    vifs = compute_vif(df, ["x1", "x2"])
    print("\nVIF:", vifs)

    # 诊断
    print("\nBreusch-Pagan:", breusch_pagan_test(result))
    print("White test:", white_test(result))
    print("Durbin-Watson:", durbin_watson_stat(result))


if __name__ == "__main__":
    run_example()