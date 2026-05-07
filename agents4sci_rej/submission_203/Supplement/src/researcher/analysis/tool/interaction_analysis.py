# -*- coding: utf-8 -*-
"""
Interaction Analysis Tool

基于公式接口执行交互效应分析：
- 使用 OLS + ANOVA（typ=2 或 typ=3）评估主效应与交互效应
- 自动提取交互项（形如 A:B）显著性与效应量（部分 eta^2）
- 返回可序列化的结果字典
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def _to_serializable_anova(df_anova: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df_anova.reset_index().rename(columns={"index": "term"})
    rows = []
    for _, r in df.iterrows():
        # 将 numpy 类型转换成内置类型以便 JSON 序列化
        row = {k: (float(v) if isinstance(v, (int, float)) else (None if pd.isna(v) else v))
               for k, v in r.items()}
        rows.append(row)
    return rows

def interaction_analysis(data: pd.DataFrame, formula: str, typ: int = 2) -> Dict[str, Any]:
    """
    参数:
        data: pandas DataFrame
        formula: 类似 'y ~ A + B + A:B' 或 'y ~ A * B' 的公式，支持 C(A) 指定分类
        typ: ANOVA 类型（2 或 3）

    返回:
        {
            "formula": str,
            "anova_table": [ {term, sum_sq, df, F, PR(>F)}, ... ],
            "interaction_terms": [ {term, pvalue, fvalue, eta_sq_partial}, ... ],
            "model_metrics": {rsquared, rsquared_adj, nobs, f_pvalue},
            "notes": [...]
        }
    """
    notes: List[str] = []
    try:
        model = smf.ols(formula=formula, data=data).fit()
    except Exception as e:
        return {
            "formula": formula,
            "error": f"OLS 拟合失败: {e}",
            "notes": ["请检查公式中的变量是否存在于数据列中，必要时使用 C(var) 指定分类变量。"]
        }

    try:
        anova_typ = 3 if typ == 3 else 2
        anova_table = sm.stats.anova_lm(model, typ=anova_typ)
    except Exception as e:
        return {
            "formula": formula,
            "error": f"ANOVA 计算失败: {e}",
            "notes": ["若包含分类变量，建议在公式中使用 C(var) 显式声明。"]
        }

    serial_anova = _to_serializable_anova(anova_table)

    # 计算 Residual sum_sq 以便部分 eta^2
    sum_sq_resid: Optional[float] = None
    if "Residual" in anova_table.index and "sum_sq" in anova_table.columns:
        try:
            sum_sq_resid = float(anova_table.loc["Residual", "sum_sq"])
        except Exception:
            sum_sq_resid = None

    interaction_terms = []
    for term in anova_table.index:
        if term == "Residual":
            continue
        # statsmodels 中交互项通常以 "A:B" 命名
        if ":" in term:
            pval = anova_table.loc[term, "PR(>F)"] if "PR(>F)" in anova_table.columns else None
            fval = anova_table.loc[term, "F"] if "F" in anova_table.columns else None
            ss = anova_table.loc[term, "sum_sq"] if "sum_sq" in anova_table.columns else None
            eta_partial = None
            if ss is not None and sum_sq_resid is not None and (ss + sum_sq_resid) > 0:
                eta_partial = float(ss) / float(ss + sum_sq_resid)
            interaction_terms.append({
                "term": term,
                "pvalue": float(pval) if pval is not None else None,
                "fvalue": float(fval) if fval is not None else None,
                "eta_sq_partial": eta_partial
            })

    result = {
        "formula": formula,
        "anova_table": serial_anova,
        "interaction_terms": interaction_terms,
        "model_metrics": {
            "rsquared": float(getattr(model, "rsquared", float('nan'))),
            "rsquared_adj": float(getattr(model, "rsquared_adj", float('nan'))),
            "nobs": int(getattr(model, "nobs", 0)),
            "f_pvalue": float(getattr(model, "f_pvalue", float('nan'))),
        },
        "notes": notes
    }
    return result