"""Helpers to select and format neutral/discriminative checks from item data."""

from typing import Any


def get_neutral_checks(item: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    return item["neutral_checks"][:k]


def get_discriminative_checks(item: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    return item["discriminative_checks"][:k]


def get_check_ground_truths(
    item: dict[str, Any], condition: str, k: int = 3
) -> list[str]:
    if condition == "B":
        checks = get_neutral_checks(item, k)
    elif condition == "C":
        checks = get_discriminative_checks(item, k)
    else:
        return []
    return [c["answer_glossary"] for c in checks]
