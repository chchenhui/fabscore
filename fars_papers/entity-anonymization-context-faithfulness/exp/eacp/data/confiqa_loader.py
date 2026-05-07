"""ConFiQA dataset loader. Reads ConFiQA-MC.json from Context-DPO repo and provides
entity extraction utilities for conditions B/C."""

import ast
import json
import os
from typing import Any

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "Context-DPO", "ConFiQA"
)


def load_confiqa(split: str = "MC") -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, f"ConFiQA-{split}.json")
    with open(path, "r") as f:
        data = json.load(f)
    for idx, inst in enumerate(data):
        inst["instance_id"] = idx
    return data


def _parse_path_labeled(path_str: str) -> list[tuple[str, str, str]]:
    try:
        return ast.literal_eval(path_str)
    except (ValueError, SyntaxError):
        return []


def extract_path_entities(instance: dict) -> dict[str, set[str]]:
    """Collect all entity labels from cf_path_labeled and orig_path_labeled,
    along with aliases from cf_alias and orig_alias.
    Returns {entity_label: {alias1, alias2, ...}} where the label itself is included."""
    entities: dict[str, set[str]] = {}

    for key in ("cf_path_labeled", "orig_path_labeled"):
        triples = _parse_path_labeled(instance.get(key, "[]"))
        for triple in triples:
            for entity in (triple[0], triple[2]):
                if entity not in entities:
                    entities[entity] = {entity}

    cf_answer = instance.get("cf_answer", "")
    orig_answer = instance.get("orig_answer", "")
    cf_aliases = instance.get("cf_alias", [])
    orig_aliases = instance.get("orig_alias", [])

    if cf_answer:
        entities.setdefault(cf_answer, {cf_answer})
        for alias in cf_aliases:
            entities[cf_answer].add(alias)

    if orig_answer:
        entities.setdefault(orig_answer, {orig_answer})
        for alias in orig_aliases:
            entities[orig_answer].add(alias)

    return entities
