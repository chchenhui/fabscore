"""Entity inventory builder for conditions B and C. Assigns sequential IDs
(ENT_1, ENT_2, ...) to path entities, infers entity types from relation context,
and generates the inventory text block."""

import ast
from typing import Any

from eacp.data.confiqa_loader import extract_path_entities

RELATION_SUBJECT_TYPE = {
    "country of citizenship": "person",
    "head of state": "country",
    "head of government": "country",
    "spouse": "person",
    "country of origin": "entity",
    "country": "entity",
    "work location": "person",
    "religion": "entity",
    "position held": "person",
    "member of": "person",
    "employer": "person",
    "award received": "person",
    "field of work": "person",
    "capital": "country",
    "member of sports team": "person",
    "official language": "country",
    "continent": "country",
    "sport": "person",
    "currency": "country",
    "follows": "entity",
    "ethnic group": "country",
    "native language": "person",
    "headquarters location": "organization",
    "founded by": "organization",
    "director": "film",
    "capital of": "city",
    "composer": "work",
    "performer": "work",
    "creator": "entity",
    "owned by": "organization",
    "author": "work",
    "chairperson": "organization",
    "head coach": "team",
    "location": "entity",
    "chief executive officer": "organization",
    "genre": "work",
    "developer": "entity",
    "record label": "work",
    "language of work or name": "work",
    "position played": "person",
}

RELATION_OBJECT_TYPE = {
    "country of citizenship": "country",
    "head of state": "person",
    "head of government": "person",
    "spouse": "person",
    "country of origin": "country",
    "country": "country",
    "work location": "location",
    "religion": "religion",
    "position held": "position",
    "member of": "organization",
    "employer": "organization",
    "award received": "award",
    "field of work": "field",
    "capital": "city",
    "member of sports team": "team",
    "official language": "language",
    "continent": "continent",
    "sport": "sport",
    "currency": "currency",
    "follows": "entity",
    "ethnic group": "ethnic_group",
    "native language": "language",
    "headquarters location": "location",
    "founded by": "person",
    "director": "person",
    "capital of": "country",
    "composer": "person",
    "performer": "person",
    "creator": "person",
    "owned by": "organization",
    "author": "person",
    "chairperson": "person",
    "head coach": "person",
    "location": "location",
    "chief executive officer": "person",
    "genre": "genre",
    "developer": "organization",
    "record label": "organization",
    "language of work or name": "language",
    "position played": "position",
}


def _parse_path_labeled(path_str: str) -> list[tuple[str, str, str]]:
    try:
        return ast.literal_eval(path_str)
    except (ValueError, SyntaxError):
        return []


def _infer_entity_types(instance: dict) -> dict[str, str]:
    """Infer entity types from relation triples in cf_path_labeled and orig_path_labeled."""
    entity_types: dict[str, str] = {}
    for key in ("cf_path_labeled", "orig_path_labeled"):
        triples = _parse_path_labeled(instance.get(key, "[]"))
        for subj, rel, obj in triples:
            if subj not in entity_types:
                entity_types[subj] = RELATION_SUBJECT_TYPE.get(rel, "entity")
            if obj not in entity_types:
                entity_types[obj] = RELATION_OBJECT_TYPE.get(rel, "entity")
    return entity_types


def build_entity_inventory(instance: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Build entity inventory: {entity_label: (ent_id, entity_type)}.
    Entities are ordered deterministically by first appearance in cf_path_labeled triples,
    followed by any additional entities from orig_path_labeled/aliases."""
    entities = extract_path_entities(instance)
    entity_types = _infer_entity_types(instance)

    seen_order: list[str] = []
    for key in ("cf_path_labeled", "orig_path_labeled"):
        triples = _parse_path_labeled(instance.get(key, "[]"))
        for subj, _, obj in triples:
            if subj not in seen_order and subj in entities:
                seen_order.append(subj)
            if obj not in seen_order and obj in entities:
                seen_order.append(obj)

    for label in entities:
        if label not in seen_order:
            seen_order.append(label)

    entity_map: dict[str, tuple[str, str]] = {}
    for idx, label in enumerate(seen_order, start=1):
        ent_id = f"ENT_{idx}"
        ent_type = entity_types.get(label, "entity")
        entity_map[label] = (ent_id, ent_type)

    return entity_map


def build_inventory_block_B(entity_map: dict[str, tuple[str, str]]) -> str:
    """Generate inventory text block with real names visible (condition B).
    Example: 'Entity Inventory:\n- ENT_1 = Bobby Moore (type=person)\n...'"""
    lines = ["Entity Inventory:"]
    sorted_entries = sorted(entity_map.items(), key=lambda x: int(x[1][0].split("_")[1]))
    for label, (ent_id, ent_type) in sorted_entries:
        lines.append(f"- {ent_id} = {label} (type={ent_type})")
    return "\n".join(lines)


def build_id_to_entity_map(entity_map: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Return {ent_id -> entity_label} for scoring."""
    return {ent_id: label for label, (ent_id, _) in entity_map.items()}
