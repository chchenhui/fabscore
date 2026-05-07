"""Condition A prompt builder: Opinion & Instruction (O&I) baseline.
Matches the 'instr+opin' schema from Context-DPO evaluation.py."""

from typing import Any

SYSTEM_MESSAGE = "Answer concisely with the answer only."


def build_prompt_a(instance: dict[str, Any]) -> str:
    """Build the raw O&I prompt string (before chat template wrapping)."""
    context = instance["cf_context"].replace('"', "")
    question = instance["question"]
    if question.endswith("?"):
        question = question[:-1]
    prompt = (
        "Instruction: read the given information and answer the corresponding question.\n\n"
        f'Bob said "{context}"\n'
        f"Q: {question} in Bob's opinion?\n"
        "A:"
    )
    return prompt


def build_chat_messages_a(instance: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for Llama-3.1-Instruct chat template."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_prompt_a(instance)},
    ]
