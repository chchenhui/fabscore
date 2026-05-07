"""Asymptotic quantifier benchmark items (O/o notation).

Glossary convention (ErdosProblems): f = O(g) means |f(x)| <= C*g(x) for all
sufficiently large x (finitely many exceptions allowed).
Alternate (strict): f = O(g) means |f(x)| <= C*g(x) for ALL x (no exceptions).

Items define piecewise f,g where the bound fails for small x but holds for large x.
Each discriminative check has a different answer under the two conventions.
"""

import random
from typing import Any


GLOSSARY_SNIPPET = (
    "We say f = O(g) if there exists a constant C > 0 such that |f(x)| <= C * g(x) "
    "for all sufficiently large x. That is, the bound need only hold beyond some "
    "threshold x_0; finitely many exceptions for small x are permitted."
)

X0_VALUES = [1, 2, 3, 5]


def _poly_str(coeffs: list[int], var: str = "x") -> str:
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(f"{c}*{var}" if c != 1 else var)
        else:
            terms.append(f"{c}*{var}^{i}" if c != 1 else f"{var}^{i}")
    return " + ".join(terms) if terms else "0"


def _eval_poly(coeffs: list[int], x: int) -> int:
    return sum(c * x**i for i, c in enumerate(coeffs))


def _make_disc_checks_big_o(
    x0: int, spike_val: int, g_at_x0: int, f_coeffs: list[int], g_coeffs: list[int]
) -> list[dict[str, Any]]:
    checks = []

    checks.append({
        "question": (
            f"Does |f(x)| <= C * g(x) hold for ALL x >= 1 with some constant C? "
            f"(Consider x = {x0} where |f({x0})| = {spike_val} and g({x0}) = {g_at_x0}.)"
        ),
        "answer_glossary": "No, but this does not matter because the definition only requires the bound for sufficiently large x",
        "answer_alternate": "No",
    })

    checks.append({
        "question": (
            f"Is the violation |f({x0})| = {spike_val} > C * g({x0}) = C * {g_at_x0} "
            f"(for any C) a problem for concluding f = O(g)?"
        ),
        "answer_glossary": "No",
        "answer_alternate": "Yes",
    })

    checks.append({
        "question": (
            f"If we only consider x > {x0}, can we find C > 0 such that "
            f"|f(x)| <= C * g(x) for all x > {x0}?"
        ),
        "answer_glossary": "Yes",
        "answer_alternate": "Yes, but this is not sufficient because the bound must hold for all x",
    })

    return checks


def _make_disc_checks_little_o(
    x0: int, spike_val: int, g_at_x0: int
) -> list[dict[str, Any]]:
    checks = []

    checks.append({
        "question": (
            f"Does the large value |f({x0})| = {spike_val} compared to g({x0}) = {g_at_x0} "
            f"prevent f = o(g) from holding?"
        ),
        "answer_glossary": "No",
        "answer_alternate": "Yes",
    })

    checks.append({
        "question": (
            f"For the definition f = o(g), do we need f(x)/g(x) -> 0 considering "
            f"all x >= 1, or only as x -> infinity?"
        ),
        "answer_glossary": "Only as x -> infinity",
        "answer_alternate": "For all x >= 1",
    })

    checks.append({
        "question": (
            f"Is the ratio |f({x0})|/g({x0}) = {spike_val}/{g_at_x0} relevant "
            f"to determining whether f = o(g)?"
        ),
        "answer_glossary": "No",
        "answer_alternate": "Yes",
    })

    return checks


def _generate_big_o_item(
    rng: random.Random, item_id: int
) -> dict[str, Any] | None:
    x0 = rng.choice(X0_VALUES)
    deg_g = rng.randint(1, 3)
    g_coeffs = [0] * (deg_g + 1)
    g_coeffs[deg_g] = rng.randint(1, 3)
    if deg_g >= 1:
        g_coeffs[0] = rng.randint(0, 2)

    f_leading = rng.randint(1, min(5, 3 * g_coeffs[deg_g]))
    f_coeffs = [0] * (deg_g + 1)
    f_coeffs[deg_g] = f_leading

    spike_val = abs(_eval_poly(g_coeffs, x0)) * (10 + rng.randint(1, 20))
    if spike_val == 0:
        spike_val = 100

    f_str = _poly_str(f_coeffs)
    g_str = _poly_str(g_coeffs)
    f_pw_str = f"|f(x)| = {spike_val} for x <= {x0}, and f(x) = {f_str} for x > {x0}"
    g_pw_str = f"g(x) = {g_str} for all x >= 1"

    question = (
        f"Define f and g for positive integers x as follows:\n"
        f"  {f_pw_str}\n"
        f"  {g_pw_str}\n"
        f"Is f = O(g)?"
    )

    g_at_x0 = _eval_poly(g_coeffs, x0)
    if g_at_x0 == 0:
        return None

    gt_glossary = "Yes"
    gt_alternate = "No"

    disc_checks = _make_disc_checks_big_o(x0, spike_val, g_at_x0, f_coeffs, g_coeffs)

    eval_x = max(x0 + 10, 50)
    f_at_eval = _eval_poly(f_coeffs, eval_x)
    g_at_eval = _eval_poly(g_coeffs, eval_x)
    ratio_large = round(abs(f_at_eval) / g_at_eval, 4) if g_at_eval != 0 else "undefined"

    neutral_checks = [
        {
            "question": f"What is f({eval_x})?",
            "answer_glossary": str(f_at_eval),
            "answer_alternate": str(f_at_eval),
        },
        {
            "question": f"What is g({eval_x})?",
            "answer_glossary": str(g_at_eval),
            "answer_alternate": str(g_at_eval),
        },
        {
            "question": f"What is |f({eval_x})| / g({eval_x})? (Give a decimal or fraction.)",
            "answer_glossary": str(ratio_large),
            "answer_alternate": str(ratio_large),
        },
    ]

    return {
        "family": "asymptotics",
        "glossary_snippet": GLOSSARY_SNIPPET,
        "neutral_checks": neutral_checks[:3],
        "discriminative_checks": disc_checks[:3],
        "main_question": question,
        "ground_truth_glossary": gt_glossary,
        "ground_truth_alternate": gt_alternate,
        "metadata": {
            "x0": x0,
            "spike_val": spike_val,
            "f_coeffs": f_coeffs,
            "g_coeffs": g_coeffs,
            "item_id": item_id,
        },
    }


def _generate_little_o_item(
    rng: random.Random, item_id: int
) -> dict[str, Any] | None:
    x0 = rng.choice(X0_VALUES)
    deg_g = rng.randint(2, 3)
    g_coeffs = [0] * (deg_g + 1)
    g_coeffs[deg_g] = rng.randint(1, 3)

    f_deg = deg_g - 1
    f_coeffs = [0] * (f_deg + 1)
    f_coeffs[f_deg] = rng.randint(1, 5)

    spike_val = abs(_eval_poly(g_coeffs, x0)) * (5 + rng.randint(1, 10))
    if spike_val == 0:
        spike_val = 50

    f_str = _poly_str(f_coeffs)
    g_str = _poly_str(g_coeffs)
    f_pw_str = f"|f(x)| = {spike_val} for x <= {x0}, and f(x) = {f_str} for x > {x0}"
    g_pw_str = f"g(x) = {g_str} for all x >= 1"

    question = (
        f"Define f and g for positive integers x as follows:\n"
        f"  {f_pw_str}\n"
        f"  {g_pw_str}\n"
        f"Is f = o(g)? (i.e., does f(x)/g(x) -> 0 as x -> infinity?)"
    )

    g_at_x0 = _eval_poly(g_coeffs, x0)
    if g_at_x0 == 0:
        return None

    gt_glossary = "Yes"
    gt_alternate = "No"

    disc_checks = _make_disc_checks_little_o(x0, spike_val, g_at_x0)

    eval_x = max(x0 + 10, 50)
    f_at_eval = _eval_poly(f_coeffs, eval_x)
    g_at_eval = _eval_poly(g_coeffs, eval_x)
    ratio = round(abs(f_at_eval) / g_at_eval, 6) if g_at_eval != 0 else "undefined"

    neutral_checks = [
        {
            "question": f"What is f({eval_x})?",
            "answer_glossary": str(f_at_eval),
            "answer_alternate": str(f_at_eval),
        },
        {
            "question": f"What is g({eval_x})?",
            "answer_glossary": str(g_at_eval),
            "answer_alternate": str(g_at_eval),
        },
        {
            "question": f"What is |f({eval_x})| / g({eval_x})?",
            "answer_glossary": str(ratio),
            "answer_alternate": str(ratio),
        },
    ]

    return {
        "family": "asymptotics",
        "glossary_snippet": GLOSSARY_SNIPPET,
        "neutral_checks": neutral_checks[:3],
        "discriminative_checks": disc_checks[:3],
        "main_question": question,
        "ground_truth_glossary": gt_glossary,
        "ground_truth_alternate": gt_alternate,
        "metadata": {
            "x0": x0,
            "spike_val": spike_val,
            "f_coeffs": f_coeffs,
            "g_coeffs": g_coeffs,
            "subtype": "little_o",
            "item_id": item_id,
        },
    }


def generate_items(target_count: int = 100, seed: int = 43) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = target_count * 20

    big_o_target = target_count * 2 // 3
    little_o_target = target_count - big_o_target
    big_o_count = 0
    little_o_count = 0

    while len(items) < target_count and attempts < max_attempts:
        attempts += 1
        if big_o_count < big_o_target:
            item = _generate_big_o_item(rng, len(items))
        elif little_o_count < little_o_target:
            item = _generate_little_o_item(rng, len(items))
        else:
            if rng.random() < 0.67:
                item = _generate_big_o_item(rng, len(items))
            else:
                item = _generate_little_o_item(rng, len(items))

        if item is not None:
            if item["metadata"].get("subtype") == "little_o":
                little_o_count += 1
            else:
                big_o_count += 1
            items.append(item)

    return items
