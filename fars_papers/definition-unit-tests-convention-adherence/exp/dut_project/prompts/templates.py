"""Prompt template formatters for three experimental conditions.

Condition A: Glossary-only (no checks) - chat message format
Condition B: Neutral checks + solve - chat message format
Condition C: Discriminative DUT checks + solve - chat message format with
             strong convention-binding instructions

All conditions use chat message format for consistent instruction-following.
Condition C has stronger glossary-grounding instructions to force convention binding.
"""

from typing import Any


def format_condition_a(item: dict[str, Any]) -> str:
    return (
        f"[GLOSSARY]\n"
        f"{item['glossary_snippet']}\n"
        f"[QUESTION]\n"
        f"{item['main_question']}\n"
        f"FINAL_ANSWER:"
    )


def format_condition_a_chat(item: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise mathematics assistant. Solve the problem "
                "step-by-step and put your final answer in \\boxed{}. "
                "For Yes/No questions, answer with \\boxed{Yes} or \\boxed{No}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[GLOSSARY]\n{item['glossary_snippet']}\n\n"
                f"[QUESTION]\n{item['main_question']}\n\n"
                "Solve step-by-step using the glossary definition. "
                "Put your final answer in \\boxed{}. "
                "For Yes/No questions, answer \\boxed{Yes} or \\boxed{No}."
            ),
        },
    ]


def format_condition_b_chat(item: dict[str, Any], k: int = 3) -> list[dict[str, str]]:
    checks = item["neutral_checks"][:k]
    glossary = item["glossary_snippet"]

    verified_block = ""
    for i, c in enumerate(checks):
        verified_block += (
            f"VERIFIED FACT {i+1}: Q: {c['question']}\n"
            f"  A: {c['answer_glossary']}\n"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are a precise mathematics assistant. You will be given a glossary "
                "with definitions, verified facts, and a main question.\n\n"
                "Put your final answer in \\boxed{}. "
                "For Yes/No questions, answer \\boxed{Yes} or \\boxed{No}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[GLOSSARY]\n{glossary}\n\n"
                f"[VERIFIED FACTS]\n{verified_block}\n"
                f"[QUESTION]\n{item['main_question']}\n\n"
                "Using the glossary definition (and consistent with the verified facts above), "
                "solve step-by-step. Put your final answer in \\boxed{}. "
                "For Yes/No questions, answer \\boxed{Yes} or \\boxed{No}."
            ),
        },
    ]


def format_condition_c_chat(item: dict[str, Any], k: int = 3) -> list[dict[str, str]]:
    checks = item["discriminative_checks"][:k]
    glossary = item["glossary_snippet"]

    verified_block = ""
    for i, c in enumerate(checks):
        verified_block += (
            f"VERIFIED FACT {i+1}: Q: {c['question']}\n"
            f"  A: {c['answer_glossary']}\n"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are a precise mathematics assistant. You will be given a glossary "
                "that defines specific mathematical conventions. These definitions may differ "
                "from standard textbook conventions.\n\n"
                "CRITICAL: You MUST follow the glossary definitions EXACTLY, even if they "
                "differ from what you normally use.\n\n"
                "You will also be given VERIFIED FACTS that demonstrate how the glossary "
                "definition applies. These facts are correct under the glossary convention "
                "and you must be consistent with them.\n\n"
                "Put your final answer in \\boxed{}. "
                "For Yes/No questions, answer \\boxed{Yes} or \\boxed{No}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[GLOSSARY - READ CAREFULLY]\n{glossary}\n\n"
                f"[VERIFIED FACTS - These are correct under the glossary definition]\n{verified_block}\n"
                f"[QUESTION]\n{item['main_question']}\n\n"
                "Using the glossary definition (and consistent with the verified facts above), "
                "solve step-by-step. Put your final answer in \\boxed{}. "
                "For Yes/No questions, answer \\boxed{Yes} or \\boxed{No}."
            ),
        },
    ]


def format_condition_b(item: dict[str, Any], k: int = 3) -> str:
    checks = item["neutral_checks"][:k]
    checks_str = "\n".join(
        f"CHECK_{i+1}: {c['question']}" for i, c in enumerate(checks)
    )
    return (
        f"[GLOSSARY]\n"
        f"{item['glossary_snippet']}\n"
        f"[CHECKS]\n"
        f"{checks_str}\n"
        f"[QUESTION]\n"
        f"{item['main_question']}\n"
        f"Please answer each CHECK first, then provide FINAL_ANSWER:"
    )


def format_condition_c(item: dict[str, Any], k: int = 3) -> str:
    checks = item["discriminative_checks"][:k]
    checks_str = "\n".join(
        f"CHECK_{i+1}: {c['question']}" for i, c in enumerate(checks)
    )
    return (
        f"[GLOSSARY]\n"
        f"{item['glossary_snippet']}\n"
        f"[CHECKS]\n"
        f"{checks_str}\n"
        f"[QUESTION]\n"
        f"{item['main_question']}\n"
        f"Please answer each CHECK first, then provide FINAL_ANSWER:"
    )


CONDITION_FORMATTERS = {
    "A": format_condition_a,
    "B": format_condition_b,
    "C": format_condition_c,
}

CONDITION_CHAT_FORMATTERS = {
    "A": format_condition_a_chat,
    "B": format_condition_b_chat,
    "C": format_condition_c_chat,
}
