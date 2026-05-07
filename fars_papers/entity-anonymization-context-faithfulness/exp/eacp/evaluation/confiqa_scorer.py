"""ConFiQA evaluation scorer. Re-implements alias-matching logic from
Context-DPO/evaluation.py with per-instance and aggregate metrics."""

import re
import string
from typing import Any

NEGATION_WORDS = [
    "no", "not", "never", "none", "cannot", "nobody", "nothing", "nowhere",
    "neither", "nor", "without", "hardly",
]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _contains_negation(prediction: str) -> bool:
    return any(w in prediction.split() for w in NEGATION_WORDS)


def exact_match_score(prediction: str, ground_truth: str, is_cf: bool) -> bool:
    if is_cf and _contains_negation(normalize_answer(prediction)):
        return False
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def recall_score(prediction: str, ground_truth: str, is_cf: bool) -> bool:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if is_cf and _contains_negation(pred_norm):
        return False
    return gt_norm in pred_norm


def get_score(
    prediction: str,
    cf_answer: str | list[str],
    orig_answer: str | list[str],
    cf_alias: list[str] | None = None,
    orig_alias: list[str] | None = None,
) -> tuple[bool, bool, bool]:
    """Score a single prediction.
    Returns (is_cf_recall, is_orig_recall, is_em)."""
    cf_candidates = [cf_answer] if isinstance(cf_answer, str) else list(cf_answer)
    if cf_alias:
        cf_candidates.extend(cf_alias)

    orig_candidates = [orig_answer] if isinstance(orig_answer, str) else list(orig_answer)
    if orig_alias:
        orig_candidates.extend(orig_alias)

    cf_recall = any(recall_score(prediction, g, True) for g in cf_candidates)
    orig_recall = any(recall_score(prediction, o, False) for o in orig_candidates)
    em = any(exact_match_score(prediction, g, True) for g in cf_candidates)

    is_cf_match = cf_recall and not orig_recall
    return is_cf_match, orig_recall, em


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics over all instances.
    Each dict in results must have: parsed_answer, cf_answer, orig_answer, cf_alias, orig_alias.
    Returns dict with Pc, Po, MR, EM."""
    n = len(results)
    if n == 0:
        return {"Pc": 0.0, "Po": 0.0, "MR": 0.0, "EM": 0.0}

    total_cf = 0
    total_orig = 0
    total_em = 0

    for r in results:
        pred = r["parsed_answer"]
        is_cf, is_orig, is_em = get_score(
            pred,
            r["cf_answer"],
            r["orig_answer"],
            r.get("cf_alias", []),
            r.get("orig_alias", []),
        )
        total_cf += int(is_cf)
        total_orig += int(is_orig)
        total_em += int(is_em)

    pc = total_cf * 100.0 / n
    po = total_orig * 100.0 / n
    mr = po / (pc + po + 1e-10) * 100.0
    em = total_em * 100.0 / n

    return {"Pc": round(pc, 2), "Po": round(po, 2), "MR": round(mr, 2), "EM": round(em, 2)}
