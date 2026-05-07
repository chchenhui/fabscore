"""DualRay-TopK+Weights feedback construction for repair prompts (Condition C).

Extends Condition B by including normalized Farkas multiplier weights alongside
constraint names. Weights are computed as w_i = |y_i| / sum_{j in TopK} |y_j|,
giving the LLM a quantitative signal about each constraint's contribution to
infeasibility.
"""

from dualrayrank.solver.highs_wrapper import DualRayResult
from dualrayrank.solver.lp_parser import LPModel
from dualrayrank.prompts.repair_prompt import truncate_constraint
from dualrayrank.prompts.dualray_feedback import verify_dual_ray, DUALRAY_THRESHOLD


def build_dualray_weighted_feedback(
    dual_ray: DualRayResult,
    lp_model: LPModel,
    k: int = 10,
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

    weight_sum = sum(m for _, _, m in selected)
    if weight_sum < 1e-15:
        return None

    lines = []
    header = f"The following {len(selected)} constraints are most implicated in the infeasibility (ranked by importance, with contribution weights)"
    if len(entries) > k:
        header += f" (showing top {k} of {len(entries)} implicated constraints)"
    header += ":"
    lines.append(header)

    for i, (name, text, mult) in enumerate(selected, 1):
        weight = mult / weight_sum
        constraint_text = text
        if ": " in constraint_text:
            constraint_text = constraint_text.split(": ", 1)[1]
        truncated = truncate_constraint(constraint_text)
        lines.append(f"{i}. [{name}] (weight: {weight:.3f}): {truncated}")

    return "\n".join(lines)
