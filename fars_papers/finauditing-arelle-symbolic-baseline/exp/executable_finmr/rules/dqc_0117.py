# DQC_0117: Dimensional cross-check rule.
# Sums dimensionalized member facts for the target concept (those with explicit
# dimensions in matching period). The default-context fact is the extracted_value;
# the sum of dimensional members is the calculated_value.

from decimal import Decimal, InvalidOperation
from typing import Optional

from executable_finmr.engine.arelle_runner import (
    ArelleResult,
    get_dimensional_facts,
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


def _group_by_axis(dim_facts):
    axis_groups = {}
    for af in dim_facts:
        for axis_qname in af.dimensions:
            if axis_qname not in axis_groups:
                axis_groups[axis_qname] = []
            axis_groups[axis_qname].append(af)
    return axis_groups


def _sum_facts(facts):
    total = Decimal(0)
    n_summed = 0
    for af in facts:
        if af.value:
            v = _normalize(af.value)
            if v is not None:
                total += Decimal(v)
                n_summed += 1
    return total, n_summed


def execute(result: ArelleResult, target: TargetInfo) -> dict:
    if not result.executable or result.target_fact is None:
        return {"success": False, "reason": "no_target_fact"}

    if result.model is None:
        return {"success": False, "reason": "no_model"}

    dim_facts = get_dimensional_facts(result.model, target)
    if not dim_facts:
        return {"success": False, "reason": "no_dimensional_facts"}

    ev_norm = _normalize(result.target_fact.value)
    ev_decimal = Decimal(ev_norm) if ev_norm else None

    axis_groups = _group_by_axis(dim_facts)

    if len(axis_groups) >= 1 and ev_decimal is not None:
        candidates = []
        for axis, facts in axis_groups.items():
            total, n_summed = _sum_facts(facts)
            if n_summed == 0:
                continue
            diff = abs(total - ev_decimal)
            candidates.append((axis, facts, total, n_summed, diff))

        nonzero = [(a, f, t, n, d) for a, f, t, n, d in candidates if d > 0]
        if nonzero:
            nonzero.sort(key=lambda x: x[4], reverse=True)
            best_axis, best_facts, best_total, best_n, best_diff = nonzero[0]
            return {
                "success": True,
                "extracted_value": result.target_fact.value,
                "calculated_value": _fmt(best_total),
                "n_dimensional_facts": len(best_facts),
                "n_summed": best_n,
                "axis": best_axis,
            }

    total, n_summed = _sum_facts(dim_facts)

    if n_summed == 0:
        return {"success": False, "reason": "no_parseable_dim_values"}

    return {
        "success": True,
        "extracted_value": result.target_fact.value,
        "calculated_value": _fmt(total),
        "n_dimensional_facts": len(dim_facts),
        "n_summed": n_summed,
    }
