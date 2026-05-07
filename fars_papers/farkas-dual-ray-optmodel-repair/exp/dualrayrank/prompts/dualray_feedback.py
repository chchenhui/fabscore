"""DualRay-TopK feedback construction for repair prompts (Condition B).

Given a DualRayResult from HiGHS, ranks constraints by absolute Farkas multiplier
magnitude, selects the top K, and formats a feedback string WITHOUT numeric weights.
Falls back to None if dual ray is invalid (caller should use IIS-TopK fallback).
"""

from dualrayrank.solver.highs_wrapper import DualRayResult
from dualrayrank.solver.lp_parser import LPModel
from dualrayrank.prompts.repair_prompt import truncate_constraint

DUALRAY_THRESHOLD = 1e-10


def verify_dual_ray(dual_ray: DualRayResult) -> dict:
    if dual_ray is None:
        return {"valid": False, "reason": "dual_ray_is_none", "num_nonzero": 0}

    if not dual_ray.exists:
        return {"valid": False, "reason": "dual_ray_does_not_exist", "num_nonzero": 0}

    if not dual_ray.multipliers:
        return {"valid": False, "reason": "no_multipliers_returned", "num_nonzero": 0}

    num_nonzero = sum(1 for m in dual_ray.multipliers if abs(m) > DUALRAY_THRESHOLD)
    if num_nonzero == 0:
        return {"valid": False, "reason": "all_multipliers_below_threshold", "num_nonzero": 0}

    return {"valid": True, "reason": "ok", "num_nonzero": num_nonzero}


def build_dualray_topk_feedback(
    dual_ray: DualRayResult,
    lp_model: LPModel,
    k: int = 5,
) -> str | None:
    check = verify_dual_ray(dual_ray)
    if not check["valid"]:
        return None

    entries = []
    for name, text, mult in zip(dual_ray.row_names, dual_ray.row_texts, dual_ray.multipliers):
        if abs(mult) > DUALRAY_THRESHOLD:
            entries.append((name, text, abs(mult)))

    entries.sort(key=lambda x: x[2], reverse=True)
    selected = entries[:k]

    lines = []
    header = f"The following {len(selected)} constraints are most implicated in the infeasibility (ranked by importance)"
    if len(entries) > k:
        header += f" (showing top {k} of {len(entries)} implicated constraints)"
    header += ":"
    lines.append(header)

    for i, (name, text, _) in enumerate(selected, 1):
        constraint_text = text
        if ": " in constraint_text:
            constraint_text = constraint_text.split(": ", 1)[1]
        truncated = truncate_constraint(constraint_text)
        lines.append(f"{i}. [{name}]: {truncated}")

    return "\n".join(lines)
