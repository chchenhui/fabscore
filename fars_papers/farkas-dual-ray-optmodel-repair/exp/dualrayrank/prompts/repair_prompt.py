"""Repair prompt template shared by all repair conditions (IIS-TopK, DualRay-TopK, etc.).

Constructs a prompt with: (1) original problem, (2) attempt-0 .lp file,
(3) feedback section (variable per condition), (4) instruction to output corrected .lp.
"""

import re


def truncate_constraint(text: str, max_chars: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def build_repair_prompt(question: str, attempt0_lp: str, feedback_section: str) -> str:
    return _REPAIR_TEMPLATE.format(
        question=question,
        attempt0_lp=attempt0_lp.strip(),
        feedback_section=feedback_section.strip(),
    )


def build_enhanced_repair_prompt(question: str, attempt0_lp: str, feedback_section: str) -> str:
    return _ENHANCED_REPAIR_TEMPLATE.format(
        question=question,
        attempt0_lp=attempt0_lp.strip(),
        feedback_section=feedback_section.strip(),
    )


_REPAIR_TEMPLATE = """\
You are an expert in mathematical optimization. A previous attempt to formulate the following problem as an .lp file produced an infeasible model. Your task is to correct the .lp file so that it becomes feasible and solves to the correct optimum.

## Original Problem
{question}

## Previous (Infeasible) .lp File
```
{attempt0_lp}
```

## Solver Feedback
{feedback_section}

## Instructions
- Carefully analyze the solver feedback above to identify which constraints are causing infeasibility.
- Output a corrected, complete .lp file that fixes the infeasibility while faithfully modeling the original problem.
- Keep the same constraint naming convention (c0001, c0002, c0003, ...).
- Include explicit bounds for all variables in the Bounds section.
- If integer variables are needed, include the Generals/Binary section.
- Output ONLY the .lp file content, starting with the objective function. Do not include any explanation.
Your response:
"""


_ENHANCED_REPAIR_TEMPLATE = """\
You are an expert in mathematical optimization and LP debugging. A previous attempt to formulate the following problem as an .lp file produced an INFEASIBLE model. The solver has identified which constraints contribute most to the infeasibility. Your task is to fix the .lp file so it becomes feasible and solves to the correct optimal value.

## Original Problem
{question}

## Previous (Infeasible) .lp File
```
{attempt0_lp}
```

## Solver Diagnostic Feedback
{feedback_section}

## Repair Strategy
The constraints listed above are involved in an infeasible combination. To fix the model, carefully re-read the original problem and compare it against each flagged constraint. Common root causes include:
1. **Wrong inequality direction**: e.g., `>=` should be `<=` or vice versa, based on the problem statement.
2. **Wrong coefficient or RHS value**: a number was misread from the problem (e.g., 10000 instead of 1000, or a sign error).
3. **Missing or extra constraint**: a constraint was omitted or an extraneous constraint was added that conflicts with the rest.
4. **Wrong variable bounds**: bounds in the Bounds section contradict the constraints.
5. **Modeling error**: the constraint does not faithfully represent what the problem statement says.

Focus on the constraints with the HIGHEST weights first -- these are the strongest contributors to infeasibility. Fix the root cause, not just the symptom.

## Output Requirements
- Output a corrected, COMPLETE .lp file that fixes the infeasibility while faithfully modeling the original problem.
- Keep the same constraint naming convention (c0001, c0002, c0003, ...).
- Include explicit bounds for all variables in the Bounds section.
- If integer variables are needed, include the Generals/Binary section.
- Output ONLY the .lp file content inside a code block. Do not include any explanation.
Your response:
"""
