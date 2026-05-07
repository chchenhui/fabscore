# DQC_0015: Sign/negativity constraint rule.
# extracted_value = target fact value (expected negative if DQC flagged it).
# calculated_value = abs(extracted_value).

from decimal import Decimal, InvalidOperation
from typing import Optional

from executable_finmr.engine.arelle_runner import ArelleResult
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


def execute(result: ArelleResult, target: TargetInfo) -> dict:
    if not result.executable or result.target_fact is None:
        return {"success": False, "reason": "no_target_fact"}

    ev_raw = result.target_fact.value
    ev_norm = _normalize(ev_raw)
    if ev_norm is None:
        return {"success": False, "reason": "unparseable_value"}

    try:
        d = Decimal(ev_norm)
        cv = str(int(abs(d))) if abs(d) == abs(d).to_integral_value() else str(abs(d))
    except InvalidOperation:
        return {"success": False, "reason": "decimal_error"}

    return {
        "success": True,
        "extracted_value": ev_raw,
        "calculated_value": cv,
    }
