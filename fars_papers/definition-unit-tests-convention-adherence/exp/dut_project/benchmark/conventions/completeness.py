"""Completeness convention benchmark items.

Glossary convention (ErdosProblems): A is "complete" if P(A) (the set of subset
sums of A) contains all sufficiently large integers.
Alternate (strict): A is "complete" if P(A) contains ALL positive integers.

Items define sets A such that P(A) covers all sufficiently large integers but
misses some small integers, so the answer is Yes under glossary but No under strict.
Each discriminative check has a different answer under the two conventions.
"""

import random
from typing import Any


GLOSSARY_SNIPPET = (
    "A set A of positive integers is called complete if the set P(A) of all "
    "finite subset sums of A contains all sufficiently large positive integers. "
    "That is, there exists a threshold M such that every integer n >= M can be "
    "expressed as a sum of distinct elements of A. A is strongly complete if "
    "A \\ B is complete for every finite subset B of A."
)


def _subset_sums(elements: list[int], max_val: int) -> set[int]:
    sums = {0}
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns <= max_val:
                new_sums.add(ns)
        sums |= new_sums
    sums.discard(0)
    return sums


def _find_threshold(elements: list[int], check_up_to: int) -> int | None:
    sums = _subset_sums(elements, check_up_to)
    for M in range(1, check_up_to + 1):
        if all(n in sums for n in range(M, check_up_to + 1)):
            return M
    return None


def _make_disc_checks(
    A: list[int], sums_set: set[int], small_missing: list[int], M: int
) -> list[dict[str, Any]]:
    checks = []
    test_val = small_missing[0]

    checks.append({
        "question": (
            f"The integer {test_val} is NOT in P(A). Does this mean A is not complete?"
        ),
        "answer_glossary": "No",
        "answer_alternate": "Yes",
    })

    checks.append({
        "question": (
            f"Does the definition of 'complete' require that ALL positive integers "
            f"are in P(A), or only all sufficiently large integers?"
        ),
        "answer_glossary": "Only all sufficiently large integers",
        "answer_alternate": "All positive integers",
    })

    checks.append({
        "question": (
            f"Given that every integer n >= {M} is in P(A), but {test_val} is not "
            f"in P(A), is A complete?"
        ),
        "answer_glossary": "Yes",
        "answer_alternate": "No",
    })

    return checks


def _generate_complete_item(
    rng: random.Random, item_id: int
) -> dict[str, Any] | None:
    pattern_type = rng.choice(["threshold_set", "missing_small", "strongly_complete"])

    if pattern_type == "threshold_set":
        k = rng.randint(3, 8)
        A = list(range(k, k + 20))
        check_up_to = sum(A[:10])
        M = _find_threshold(A, check_up_to)
        if M is None or M <= 1:
            return None

        missing = sorted(set(range(1, M)) - _subset_sums(A, check_up_to))
        if not missing:
            return None

        set_desc = f"A = {{{k}, {k+1}, {k+2}, ..., {k+19}}} (all integers from {k} to {k+19})"
        question = (
            f"Let {set_desc}.\n"
            f"Is A complete? (Recall: A is complete if P(A) contains all "
            f"sufficiently large positive integers.)"
        )

    elif pattern_type == "missing_small":
        exc_size = rng.randint(1, 3)
        exc_set = sorted(rng.sample(range(1, 6), exc_size))
        base = list(range(1, 25))
        A = sorted(set(base) - set(exc_set))
        check_up_to = sum(A[:12])
        M = _find_threshold(A, check_up_to)
        if M is None or M <= 1:
            return None

        missing = sorted(set(range(1, M)) - _subset_sums(A, check_up_to))
        if not missing:
            return None

        exc_str = ", ".join(str(x) for x in exc_set)
        set_desc = f"A = {{1, 2, ..., 24}} \\ {{{exc_str}}}"
        question = (
            f"Let {set_desc}.\n"
            f"Is A complete?"
        )

    elif pattern_type == "strongly_complete":
        k = rng.randint(2, 5)
        A = list(range(k, k + 25))
        check_up_to = sum(A[:12])
        M = _find_threshold(A, check_up_to)
        if M is None or M <= 1:
            return None

        missing = sorted(set(range(1, M)) - _subset_sums(A, check_up_to))
        if not missing:
            return None

        remove_count = rng.randint(1, 2)
        remove_elts = rng.sample(A[:5], min(remove_count, len(A[:5])))
        A_reduced = sorted(set(A) - set(remove_elts))
        M2 = _find_threshold(A_reduced, check_up_to)
        if M2 is None:
            return None

        set_desc = (
            f"A = {{{k}, {k+1}, ..., {k+24}}} "
            f"(all integers from {k} to {k+24})"
        )
        rem_str = ", ".join(str(x) for x in sorted(remove_elts))
        question = (
            f"Let {set_desc}.\n"
            f"Is A strongly complete? (Recall: A is strongly complete if "
            f"A \\ B is complete for every finite set B.)\n"
            f"Consider in particular removing B = {{{rem_str}}} from A."
        )

    gt_glossary = "Yes"
    gt_alternate = "No"

    sums_set = _subset_sums(A, check_up_to)
    small_missing = sorted(set(range(1, max(20, M + 5))) - sums_set)

    if not small_missing:
        return None

    disc_checks = _make_disc_checks(A, sums_set, small_missing, M)

    intersection_size = len(sums_set & set(range(1, 11)))
    neutral_checks = [
        {
            "question": f"How many elements does A have?",
            "answer_glossary": str(len(A)),
            "answer_alternate": str(len(A)),
        },
        {
            "question": f"What is the smallest element of A?",
            "answer_glossary": str(min(A)),
            "answer_alternate": str(min(A)),
        },
        {
            "question": f"How many integers in {{1, 2, ..., 10}} are in P(A)?",
            "answer_glossary": str(intersection_size),
            "answer_alternate": str(intersection_size),
        },
    ]

    return {
        "family": "completeness",
        "glossary_snippet": GLOSSARY_SNIPPET,
        "neutral_checks": neutral_checks[:3],
        "discriminative_checks": disc_checks[:3],
        "main_question": question,
        "ground_truth_glossary": gt_glossary,
        "ground_truth_alternate": gt_alternate,
        "metadata": {
            "pattern_type": pattern_type,
            "A": A,
            "threshold_M": M,
            "missing_small": small_missing[:5],
            "item_id": item_id,
        },
    }


def generate_items(target_count: int = 100, seed: int = 44) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = target_count * 30

    while len(items) < target_count and attempts < max_attempts:
        attempts += 1
        item = _generate_complete_item(rng, len(items))
        if item is not None:
            items.append(item)

    return items
