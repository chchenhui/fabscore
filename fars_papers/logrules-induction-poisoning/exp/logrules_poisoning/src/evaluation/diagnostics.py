"""Template disagreement and rule diff diagnostics for Phase-0 evaluation.
Compares C0 (clean) vs C1 (poisoned) predictions to quantify the effect
of induction-stage prompt injection on downstream parsing.
"""

from typing import List

from logrules_poisoning.src.evaluation.metrics import normalize_template


def template_disagreement(templates_a: List[str], templates_b: List[str]) -> float:
    assert len(templates_a) == len(templates_b)
    if len(templates_a) == 0:
        return 0.0
    differ = sum(
        1 for a, b in zip(templates_a, templates_b)
        if normalize_template(a) != normalize_template(b)
    )
    return differ / len(templates_a)


def print_rule_diff(rules_clean: List[str], rules_poisoned: List[str]) -> str:
    lines = []
    clean_set = set(rules_clean)
    poisoned_set = set(rules_poisoned)

    removed = clean_set - poisoned_set
    added = poisoned_set - clean_set
    shared = clean_set & poisoned_set

    lines.append(f"Clean rules: {len(rules_clean)}, Poisoned rules: {len(rules_poisoned)}")
    lines.append(f"Shared: {len(shared)}, Removed: {len(removed)}, Added: {len(added)}")

    if removed:
        lines.append("\n--- Removed (in clean, not in poisoned) ---")
        for r in sorted(removed):
            lines.append(f"  - {r}")

    if added:
        lines.append("\n+++ Added (in poisoned, not in clean) +++")
        for r in sorted(added):
            lines.append(f"  + {r}")

    if shared:
        lines.append(f"\n=== Shared ({len(shared)} rules) ===")
        for r in sorted(shared):
            lines.append(f"  = {r}")

    output = "\n".join(lines)
    print(output)
    return output
