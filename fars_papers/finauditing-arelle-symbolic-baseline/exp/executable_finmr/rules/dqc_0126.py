# DQC_0126: Calculation-consistency rule.
# Uses Arelle's summation-item relationship set to find child concepts of the
# target parent, retrieves their fact values, and computes a weighted sum.
# extracted_value = parent fact value, calculated_value = weighted sum of children.

from decimal import Decimal, InvalidOperation
from typing import Optional

from executable_finmr.engine.arelle_runner import (
    ArelleResult,
    get_calc_children,
    get_fact_value_by_qname_str,
)
from executable_finmr.data.target_extractor import TargetInfo


def _normalize(val: str) -> Optional[str]:
    s = val.strip().replace(",", "").replace(" ", "").replace("$", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    if not s:
        return None
    try:
        d = Decimal(s)
        if neg:
            d = -d
        if d == d.to_integral_value():
            return str(int(d))
        return str(d)
    except InvalidOperation:
        return None


def _fmt(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    return str(d)


def execute(result: ArelleResult, target: TargetInfo) -> dict:
    if not result.executable or result.target_fact is None:
        return {"success": False, "reason": "no_target_fact"}

    if result.model is None:
        return {"success": False, "reason": "no_model"}

    children = get_calc_children(result.model, target)
    if not children:
        return {"success": False, "reason": "no_calc_children"}

    total = Decimal(0)
    found_any = False
    missing_children = []

    for child_qname_str, weight, linkrole in children:
        child_fact = get_fact_value_by_qname_str(result.model, child_qname_str, target)
        if child_fact is not None and child_fact.value:
            cv_norm = _normalize(child_fact.value)
            if cv_norm is not None:
                total += Decimal(cv_norm) * Decimal(str(weight))
                found_any = True
            else:
                missing_children.append(child_qname_str)
        else:
            missing_children.append(child_qname_str)

    if not found_any:
        return {"success": False, "reason": "no_child_facts_found"}

    return {
        "success": True,
        "extracted_value": result.target_fact.value,
        "calculated_value": _fmt(total),
        "n_children": len(children),
        "n_missing": len(missing_children),
    }
