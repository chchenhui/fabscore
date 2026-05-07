"""Condition B prompt builder: Inventory+IDs control.
Prepends entity inventory block (with real names) to O&I template.
Model answers with entity IDs (ENT_k or UNKNOWN) instead of free text."""

from typing import Any

from eacp.data.entity_inventory import build_entity_inventory, build_inventory_block_B

SYSTEM_MESSAGE = "Answer concisely with the answer only."


def build_prompt_b(instance: dict[str, Any], inventory_block: str) -> str:
    context = instance["cf_context"].replace('"', "")
    question = instance["question"]
    if question.endswith("?"):
        question = question[:-1]
    prompt = (
        f"{inventory_block}\n\n"
        "Instruction: read the given information and answer the corresponding question. "
        "Answer with exactly one entity ID (e.g., ENT_1) or UNKNOWN. Do not explain.\n\n"
        f'Bob said "{context}"\n'
        f"Q: {question} in Bob's opinion?\n"
        "A:"
    )
    return prompt


def build_chat_messages_b(
    instance: dict[str, Any],
    entity_map: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, tuple[str, str]]]:
    """Build chat messages for condition B. Returns (messages, entity_map).
    If entity_map is not provided, it is built from the instance."""
    if entity_map is None:
        entity_map = build_entity_inventory(instance)
    inventory_block = build_inventory_block_B(entity_map)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_prompt_b(instance, inventory_block)},
    ]
    return messages, entity_map
