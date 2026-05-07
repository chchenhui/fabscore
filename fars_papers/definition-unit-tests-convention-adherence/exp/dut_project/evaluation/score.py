"""Exact-match scoring for benchmark evaluation.

Computes main accuracy (FINAL_ANSWER vs ground_truth_glossary),
check accuracy per check, and per-family breakdowns. Outputs JSON.
"""

import json
from typing import Any

from dut_project.inference.parse_outputs import _normalize_answer


def _match(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return _normalize_answer(predicted) == _normalize_answer(ground_truth)


def score_item(
    parsed: dict[str, Any],
    item: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    main_correct = _match(parsed["final_answer"], item["ground_truth_glossary"])

    check_results = []
    if condition in ("B", "C"):
        checks_key = "neutral_checks" if condition == "B" else "discriminative_checks"
        gt_checks = item[checks_key]
        for i, (pred, gt) in enumerate(zip(parsed["checks"], gt_checks)):
            check_results.append({
                "check_idx": i,
                "predicted": pred,
                "ground_truth": gt["answer_glossary"],
                "correct": _match(pred, gt["answer_glossary"]),
            })

    return {
        "item_id": item.get("item_id"),
        "family": item["family"],
        "condition": condition,
        "main_correct": main_correct,
        "predicted_answer": parsed["final_answer"],
        "ground_truth": item["ground_truth_glossary"],
        "check_results": check_results,
    }


def aggregate_scores(scored_items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_items)
    main_correct = sum(1 for s in scored_items if s["main_correct"])

    family_scores: dict[str, dict[str, int]] = {}
    for s in scored_items:
        fam = s["family"]
        if fam not in family_scores:
            family_scores[fam] = {"total": 0, "correct": 0}
        family_scores[fam]["total"] += 1
        family_scores[fam]["correct"] += int(s["main_correct"])

    check_correct_count = 0
    check_total_count = 0
    for s in scored_items:
        for cr in s["check_results"]:
            check_total_count += 1
            check_correct_count += int(cr["correct"])

    results = {
        "main_accuracy": main_correct / total if total > 0 else 0.0,
        "main_correct": main_correct,
        "total": total,
        "per_family": {
            fam: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                **v,
            }
            for fam, v in family_scores.items()
        },
        "check_accuracy": (
            check_correct_count / check_total_count
            if check_total_count > 0
            else None
        ),
        "check_correct": check_correct_count,
        "check_total": check_total_count,
    }
    return results


def save_results(results: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
