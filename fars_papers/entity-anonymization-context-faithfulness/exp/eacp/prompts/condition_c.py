"""Condition C prompt builder: Entity-Anonymized Context Prompts (EACP).
Replaces all entity surface forms with anonymized placeholders in context and
question. Inventory shows ENT_k (type=...) with NO real names. Model answers
with entity IDs, same output space as condition B.

Optimizations applied (iteration 1):
  Inventory annotation: phantom entities (those that appear in the knowledge
  graph but not in the given text) are tagged [not in text] in the inventory.
  This provides the model with a signal about which entities are mentioned in
  the passage, reducing confusion from large inventories."""

from typing import Any

from eacp.data.entity_anonymizer import anonymize_instance

SYSTEM_MESSAGE = "Answer concisely with the answer only."


def build_inventory_block_C(
    entity_map: dict[str, tuple[str, str]],
    replacement_stats: dict[str, int] | None = None,
) -> str:
    lines = ["Entity Inventory:"]
    sorted_entries = sorted(entity_map.items(), key=lambda x: int(x[1][0].split("_")[1]))
    for _label, (ent_id, ent_type) in sorted_entries:
        tag = ""
        if replacement_stats is not None and replacement_stats.get(ent_id, 0) == 0:
            tag = " [not in text]"
        lines.append(f"- {ent_id} (type={ent_type}){tag}")
    return "\n".join(lines)


def build_prompt_c(anon_context: str, anon_question: str, inventory_block: str) -> str:
    anon_context_clean = anon_context.replace('"', "")
    if anon_question.endswith("?"):
        anon_question = anon_question[:-1]
    prompt = (
        f"{inventory_block}\n\n"
        "Instruction: read the given information and answer the corresponding question. "
        "Answer with exactly one entity ID (e.g., ENT_1) or UNKNOWN. Do not explain.\n\n"
        f'Bob said "{anon_context_clean}"\n'
        f"Q: {anon_question} in Bob's opinion?\n"
        "A:"
    )
    return prompt


def build_chat_messages_c(
    instance: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, tuple[str, str]], dict[str, Any]]:
    """Build chat messages for condition C.
    Returns (messages, entity_map, anonymization_result)."""
    anon_result = anonymize_instance(instance)
    entity_map = anon_result["entity_map"]
    inventory_block = build_inventory_block_C(
        entity_map,
        replacement_stats=anon_result["replacement_stats"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_prompt_c(
            anon_result["anon_context"],
            anon_result["anon_question"],
            inventory_block,
        )},
    ]
    return messages, entity_map, anon_result
