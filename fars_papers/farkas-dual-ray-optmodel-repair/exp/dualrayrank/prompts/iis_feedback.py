"""IIS-TopK feedback construction for repair prompts (Condition A).

Given an IISResult from HiGHS, sorts constraints alphabetically by name,
takes the top K, and formats a feedback string for the repair prompt.
"""

from dualrayrank.solver.highs_wrapper import IISResult
from dualrayrank.solver.lp_parser import LPModel
from dualrayrank.prompts.repair_prompt import truncate_constraint


def build_iis_topk_feedback(
    iis: IISResult,
    lp_model: LPModel,
    k: int = 5,
) -> str:
    if not iis.success or not iis.row_names:
        return "No IIS could be computed for this instance."

    entries = []
    for name, text in zip(iis.row_names, iis.row_texts):
        entries.append((name, text))

    entries.sort(key=lambda x: x[0])

    selected = entries[:k]
    iis_size = len(entries)
    was_truncated = iis_size > k

    lines = []
    header = f"The following {len(selected)} constraints form part of an irreducible infeasible subsystem (IIS)"
    if was_truncated:
        header += f" (showing top {k} of {iis_size} IIS constraints, sorted alphabetically by name)"
    header += ":"
    lines.append(header)

    for i, (name, text) in enumerate(selected, 1):
        constraint_text = text
        if ": " in constraint_text:
            constraint_text = constraint_text.split(": ", 1)[1]
        truncated = truncate_constraint(constraint_text)
        lines.append(f"{i}. [{name}]: {truncated}")

    return "\n".join(lines)
