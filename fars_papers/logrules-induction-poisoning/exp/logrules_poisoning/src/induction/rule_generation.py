"""LogRules induction stage: generate parsing rules from labeled examples via LLM.
Uses gpt-4o-mini (or compatible) to induce general log parsing rules from K
labeled (raw_log, template) pairs, following the LogRules Figure 6 prompt format.
Returns a list of rule strings extracted from the JSON response.
"""

import json
import re
from typing import Dict, List, Optional

from openai import OpenAI


INDUCTION_PROMPT_TEMPLATE = """You are an expert in log parsing. Below are {num_examples} examples of raw log messages and their corresponding log templates. In the templates, variable parts are replaced with <*>.

{examples_block}

Based on these examples, induce a set of general rules for log parsing. The rules should describe how to identify and abstract variable parts in log messages. Each rule should be a concise natural language statement.

Return the rules in the following JSON format:
{{"task": "log_parsing", "rules": ["rule 1", "rule 2", ...]}}

Only output the JSON object, nothing else."""


RETRY_PROMPT_SUFFIX = """

IMPORTANT: Your response must be valid JSON and nothing else. Use this exact format:
{"task": "log_parsing", "rules": ["rule 1", "rule 2", ...]}"""


def _format_examples(examples: List[Dict]) -> str:
    lines = []
    for ex in examples:
        lines.append(f"Raw log: {ex['raw_log']}")
        lines.append(f"Log template: {ex['template']}")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_rules_json(text: str) -> Optional[List[str]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "rules" in data:
            rules = data["rules"]
            if isinstance(rules, list):
                return [str(r) for r in rules if r]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"rules"\s*:\s*\[', text)
    if match:
        start = match.start()
        try:
            data = json.loads(text[start:])
            if isinstance(data, dict) and "rules" in data:
                return [str(r) for r in data["rules"] if r]
        except json.JSONDecodeError:
            pass

    return None


def induce_rules(
    client: OpenAI,
    induction_examples: List[Dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 2048,
) -> Dict:
    examples_block = _format_examples(induction_examples)
    prompt = INDUCTION_PROMPT_TEMPLATE.format(
        num_examples=len(induction_examples),
        examples_block=examples_block,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content or ""

    rules = _parse_rules_json(raw_text)

    if rules is None:
        retry_prompt = prompt + RETRY_PROMPT_SUFFIX
        response2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text_retry = response2.choices[0].message.content or ""
        rules = _parse_rules_json(raw_text_retry)

        if rules is None:
            print(f"WARNING: Failed to parse rules JSON after retry. Raw: {raw_text_retry[:500]}")
            rules = []

        return {
            "raw_response": raw_text,
            "retry_response": raw_text_retry,
            "rules": rules,
            "num_rules": len(rules),
        }

    return {
        "raw_response": raw_text,
        "rules": rules,
        "num_rules": len(rules),
    }
