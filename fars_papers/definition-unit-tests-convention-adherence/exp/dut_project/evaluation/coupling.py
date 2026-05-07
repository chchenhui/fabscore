"""Coupling diagnostic: P[main correct | all checks correct] vs
P[main correct | any check wrong] for conditions B and C.

Tests whether passing checks predicts correct downstream interpretation.
"""

from typing import Any


def compute_coupling(scored_items: list[dict[str, Any]]) -> dict[str, Any]:
    items_with_checks = [s for s in scored_items if s["check_results"]]
    if not items_with_checks:
        return {"error": "No items with check results"}

    all_checks_correct = []
    any_check_wrong = []

    for s in items_with_checks:
        checks_ok = all(cr["correct"] for cr in s["check_results"])
        if checks_ok:
            all_checks_correct.append(s)
        else:
            any_check_wrong.append(s)

    p_main_given_all_checks = (
        sum(1 for s in all_checks_correct if s["main_correct"]) / len(all_checks_correct)
        if all_checks_correct
        else None
    )
    p_main_given_any_wrong = (
        sum(1 for s in any_check_wrong if s["main_correct"]) / len(any_check_wrong)
        if any_check_wrong
        else None
    )

    return {
        "n_all_checks_correct": len(all_checks_correct),
        "n_any_check_wrong": len(any_check_wrong),
        "p_main_correct_given_all_checks_correct": p_main_given_all_checks,
        "p_main_correct_given_any_check_wrong": p_main_given_any_wrong,
        "coupling_gap": (
            (p_main_given_all_checks - p_main_given_any_wrong)
            if p_main_given_all_checks is not None and p_main_given_any_wrong is not None
            else None
        ),
    }
