# Deterministic evaluator for FinMR, replicating FINAUDITING Appendix C.3.
# Classifies each prediction into one of: A (accurate), S (structural error),
# E (extraction error), C (calculation error).

import json
from decimal import Decimal, InvalidOperation
from typing import Optional


def _parse_decimal(s) -> Optional[Decimal]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return Decimal(str(s))
    s = str(s).strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "").replace(" ", "").replace("$", "")
    if not s:
        return None
    try:
        d = Decimal(s)
        if neg:
            d = -d
        return d
    except InvalidOperation:
        return None


def judge_single(prediction: dict, gold: dict) -> str:
    if not isinstance(prediction, dict):
        return "S"
    if "extracted_value" not in prediction or "calculated_value" not in prediction:
        return "S"

    gold_ev = _parse_decimal(gold.get("extracted_value"))
    pred_ev = _parse_decimal(prediction.get("extracted_value"))
    gold_cv = _parse_decimal(gold.get("calculated_value"))
    pred_cv = _parse_decimal(prediction.get("calculated_value"))

    if gold_ev is None or pred_ev is None:
        return "S"

    if gold_ev != pred_ev:
        return "E"

    if gold_cv is None or pred_cv is None:
        return "C"

    if gold_cv != pred_cv:
        return "C"

    return "A"


def evaluate(results: list[dict]) -> dict:
    labels = {"A": 0, "S": 0, "E": 0, "C": 0}
    per_instance = []

    for r in results:
        pred = r.get("prediction", {})
        gold = r.get("gold", {})
        label = judge_single(pred, gold)
        labels[label] += 1
        per_instance.append({**r, "label": label})

    n = len(results)
    metrics = {
        "ACC": labels["A"] / n if n else 0,
        "SER": labels["S"] / n if n else 0,
        "EER": labels["E"] / n if n else 0,
        "CER": labels["C"] / n if n else 0,
        "N": n,
        "N_A": labels["A"],
        "N_S": labels["S"],
        "N_E": labels["E"],
        "N_C": labels["C"],
    }
    return {"metrics": metrics, "per_instance": per_instance}


if __name__ == "__main__":
    test_cases = [
        ({"extracted_value": "-1284", "calculated_value": "1284"},
         {"extracted_value": "-1284", "calculated_value": "1284"}, "A"),
        ({"extracted_value": "-1284", "calculated_value": "1284"},
         {"extracted_value": "-1,284", "calculated_value": "1,284"}, "A"),
        ({}, {"extracted_value": "-1284", "calculated_value": "1284"}, "S"),
        ({"extracted_value": "100", "calculated_value": "200"},
         {"extracted_value": "-1284", "calculated_value": "1284"}, "E"),
        ({"extracted_value": "-1284", "calculated_value": "9999"},
         {"extracted_value": "-1284", "calculated_value": "1284"}, "C"),
        ({"extracted_value": "-2881000", "calculated_value": "2881000"},
         {"extracted_value": "-2,881,000", "calculated_value": "2,881,000"}, "A"),
    ]

    print("Running evaluator tests:")
    all_pass = True
    for pred, gold, expected in test_cases:
        result = judge_single(pred, gold)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status}: pred={pred} gold={gold} -> {result} (expected {expected})")

    print(f"\nAll tests passed: {all_pass}")
