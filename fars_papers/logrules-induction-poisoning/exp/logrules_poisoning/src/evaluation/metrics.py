"""Evaluation metrics for log parsing: PA, FTA, and wildcard_ratio.
PA = exact string match after whitespace normalization.
FTA = template-level F1 (precision over unique predicted templates,
      recall over unique ground-truth templates).
wildcard_ratio = average fraction of <*> tokens per predicted template.
"""

from collections import defaultdict
from typing import List, Dict, Tuple


def normalize_template(template: str) -> str:
    return " ".join(template.split())


def compute_pa(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    if len(predictions) == 0:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize_template(p) == normalize_template(g)
    )
    return correct / len(predictions)


def compute_fta(
    predictions: List[str],
    ground_truths: List[str],
    event_ids: List[str],
) -> Tuple[float, float, float]:
    """Compute FTA (F1 of Template Accuracy).

    For each ground-truth template group (by event_id), check if all predictions
    in that group produce the same template AND it matches the normalized
    ground-truth template. A ground-truth template is "correctly parsed" if
    this condition holds.

    precision = # correctly parsed GT templates / # unique predicted templates
    recall    = # correctly parsed GT templates / # unique GT templates
    FTA       = 2 * precision * recall / (precision + recall)
    """
    assert len(predictions) == len(ground_truths) == len(event_ids)

    gt_groups: Dict[str, List[int]] = defaultdict(list)
    for i, eid in enumerate(event_ids):
        gt_groups[eid].append(i)

    num_gt_templates = len(gt_groups)
    correctly_parsed = 0

    pred_to_gt_correct = {}

    for eid, indices in gt_groups.items():
        gt_template = normalize_template(ground_truths[indices[0]])
        pred_templates = [normalize_template(predictions[i]) for i in indices]

        all_same = all(p == pred_templates[0] for p in pred_templates)
        matches_gt = pred_templates[0] == gt_template

        if all_same and matches_gt:
            correctly_parsed += 1
            pred_to_gt_correct[pred_templates[0]] = True

    all_pred_templates = set(normalize_template(p) for p in predictions)
    num_pred_templates = len(all_pred_templates)

    if num_pred_templates == 0 or num_gt_templates == 0:
        return 0.0, 0.0, 0.0

    precision = correctly_parsed / num_pred_templates
    recall = correctly_parsed / num_gt_templates

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_wildcard_ratio(predictions: List[str]) -> float:
    if len(predictions) == 0:
        return 0.0
    ratios = []
    for pred in predictions:
        tokens = pred.split()
        if len(tokens) == 0:
            ratios.append(0.0)
            continue
        wildcard_count = tokens.count("<*>")
        ratios.append(wildcard_count / len(tokens))
    return sum(ratios) / len(ratios)


def evaluate_all(
    predictions: List[str],
    ground_truths: List[str],
    event_ids: List[str],
) -> Dict[str, float]:
    pa = compute_pa(predictions, ground_truths)
    precision, recall, fta = compute_fta(predictions, ground_truths, event_ids)
    wr = compute_wildcard_ratio(predictions)
    return {
        "PA": pa,
        "FTA": fta,
        "FTA_precision": precision,
        "FTA_recall": recall,
        "wildcard_ratio": wr,
    }
