"""Additive vs Dirichlet convolution benchmark items.

Glossary convention (ErdosProblems): (f*g)(n) = sum_{a+b=n} f(a)g(b)
Alternate convention (Dirichlet):   (f*g)(n) = sum_{ab=n}  f(a)g(b)

Items provide sparse function tables for f,g on {1,...,N} and ask for (f*g)(n).
Only items where the two conventions give different answers are retained.
"""

import random
from typing import Any


GLOSSARY_SNIPPET = (
    "We define the convolution of two arithmetic functions f and g as:\n"
    "  (f * g)(n) = \\sum_{a+b=n} f(a) g(b)\n"
    "where the sum ranges over all pairs of positive integers (a, b) with a + b = n."
)

N_VALUES = [6, 8, 10, 12]
SUPPORT_SIZES = [2, 3, 4]
VALUE_RANGE = range(1, 6)


def _make_sparse_func(N: int, support_size: int, rng: random.Random) -> dict[int, int]:
    keys = sorted(rng.sample(range(1, N + 1), support_size))
    return {k: rng.choice(VALUE_RANGE) for k in keys}


def _additive_conv(f: dict[int, int], g: dict[int, int], n: int) -> int:
    total = 0
    for a in range(1, n):
        b = n - a
        total += f.get(a, 0) * g.get(b, 0)
    return total


def _dirichlet_conv(f: dict[int, int], g: dict[int, int], n: int) -> int:
    total = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            total += f.get(d, 0) * g.get(n // d, 0)
            if d != n // d:
                total += f.get(n // d, 0) * g.get(d, 0)
        d += 1
    return total


def _func_table_str(name: str, func: dict[int, int], N: int) -> str:
    parts = [f"{name}({k}) = {v}" for k, v in sorted(func.items())]
    parts.append(f"{name}(n) = 0 for all other n in {{1, ..., {N}}}")
    return "; ".join(parts)


def _find_discriminative_targets(
    f: dict[int, int], g: dict[int, int], N: int, exclude: int, count: int = 3
) -> list[int]:
    candidates = []
    for m in range(3, N + 1):
        if m == exclude:
            continue
        a_val = _additive_conv(f, g, m)
        d_val = _dirichlet_conv(f, g, m)
        if a_val != d_val:
            candidates.append(m)
    random.shuffle(candidates)
    return candidates[:count]


def _make_neutral_checks(
    f: dict[int, int], g: dict[int, int], N: int, target_n: int, rng: random.Random
) -> list[dict[str, Any]]:
    checks = []

    pair_count = max(0, target_n - 2 + 1)  # pairs (a,b) with a+b=n, a,b in {1..N}, a>=1, b>=1
    pair_count = min(pair_count, target_n - 1)
    if target_n - 1 <= N:
        pair_count = target_n - 1
    else:
        pair_count = N - (target_n - N) + 1 if target_n - N >= 1 else target_n - 1
    actual_pairs = sum(1 for a in range(1, N + 1) for b in range(1, N + 1) if a + b == target_n)
    checks.append({
        "question": f"How many pairs (a, b) with a, b in {{1, ..., {N}}} satisfy a + b = {target_n}?",
        "answer_glossary": str(actual_pairs),
        "answer_alternate": str(actual_pairs),
    })

    all_keys = sorted(set(list(f.keys()) + list(g.keys())))
    if len(all_keys) >= 2:
        k1, k2 = rng.sample(all_keys, 2)
        val = f.get(k1, 0) + g.get(k2, 0)
        checks.append({
            "question": f"What is f({k1}) + g({k2})?",
            "answer_glossary": str(val),
            "answer_alternate": str(val),
        })
    else:
        k1 = all_keys[0] if all_keys else 1
        val = f.get(k1, 0)
        checks.append({
            "question": f"What is f({k1})?",
            "answer_glossary": str(val),
            "answer_alternate": str(val),
        })

    fsum = sum(f.get(i, 0) for i in range(1, N + 1))
    checks.append({
        "question": f"What is the sum of f(n) for n = 1, 2, ..., {N}?",
        "answer_glossary": str(fsum),
        "answer_alternate": str(fsum),
    })

    return checks[:3]


def generate_items(target_count: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items = []
    attempts = 0
    max_attempts = target_count * 20

    while len(items) < target_count and attempts < max_attempts:
        attempts += 1
        N = rng.choice(N_VALUES)
        ss_f = rng.choice(SUPPORT_SIZES)
        ss_g = rng.choice(SUPPORT_SIZES)
        f = _make_sparse_func(N, ss_f, rng)
        g = _make_sparse_func(N, ss_g, rng)

        target_n = rng.randint(3, N)
        a_val = _additive_conv(f, g, target_n)
        d_val = _dirichlet_conv(f, g, target_n)

        if a_val == d_val:
            continue

        disc_targets = _find_discriminative_targets(f, g, N, target_n, count=3)
        if len(disc_targets) < 3:
            continue

        disc_checks = []
        for m in disc_targets:
            a_m = _additive_conv(f, g, m)
            d_m = _dirichlet_conv(f, g, m)
            disc_checks.append({
                "question": f"What is (f * g)({m})?",
                "answer_glossary": str(a_m),
                "answer_alternate": str(d_m),
            })

        neutral_checks = _make_neutral_checks(f, g, N, target_n, rng)

        f_str = _func_table_str("f", f, N)
        g_str = _func_table_str("g", g, N)
        question = (
            f"Let f and g be arithmetic functions defined on {{1, 2, ..., {N}}} as follows:\n"
            f"  {f_str}\n"
            f"  {g_str}\n"
            f"What is (f * g)({target_n})?"
        )

        items.append({
            "family": "convolution",
            "glossary_snippet": GLOSSARY_SNIPPET,
            "neutral_checks": neutral_checks,
            "discriminative_checks": disc_checks,
            "main_question": question,
            "ground_truth_glossary": str(a_val),
            "ground_truth_alternate": str(d_val),
            "metadata": {"N": N, "target_n": target_n, "f": f, "g": g},
        })

    return items
