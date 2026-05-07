# -*- coding: utf-8 -*-
"""
Two-way ANOVA Tool

执行双因素方差分析（含交互项）：
- 公式：dv ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)
- 返回 ANOVA 表与模型指标，结构化且可序列化
"""

from typing import Dict, Any, List
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def _serialize_anova_table(tbl: pd.DataFrame) -> List[Dict[str, Any]]:
    df = tbl.reset_index().rename(columns={"index": "term"})
    rows = []
    for _, r in df.iterrows():
        row = {}
        for k, v in r.items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
            else:
                row[k] = None if pd.isna(v) else v
        rows.append(row)
    return rows

def two_way_anova(data: pd.DataFrame, dv: str, factor_a: str, factor_b: str, typ: int = 2) -> Dict[str, Any]:
    """
    参数:
        data: DataFrame
        dv: 因变量列名
        factor_a: 因子 A 列名（视为分类）
        factor_b: 因子 B 列名（视为分类）
        typ: ANOVA 类型（2 或 3）

    返回:
        {
            "formula": str,
            "anova_table": [...],
            "model_metrics": {rsquared, rsquared_adj, nobs, f_pvalue},
            "notes": [...]
        }
    """
    notes: List[str] = []
    formula = f"{dv} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})"
    try:
        model = smf.ols(formula=formula, data=data).fit()
        anova_typ = 3 if typ == 3 else 2
        anova_table = sm.stats.anova_lm(model, typ=anova_typ)
    except Exception as e:
        return {
            "formula": formula,
            "error": f"Two-way ANOVA 计算失败: {e}",
            "notes": ["请确认 dv/factor_a/factor_b 是否存在且类型合理（因子需为分类变量或可转换为分类）。"]
        }

    return {
        "formula": formula,
        "anova_table": _serialize_anova_table(anova_table),
        "model_metrics": {
            "rsquared": float(getattr(model, "rsquared", float('nan'))),
            "rsquared_adj": float(getattr(model, "rsquared_adj", float('nan'))),
            "nobs": int(getattr(model, "nobs", 0)),
            "f_pvalue": float(getattr(model, "f_pvalue", float('nan'))),
        },
        "notes": notes
    }