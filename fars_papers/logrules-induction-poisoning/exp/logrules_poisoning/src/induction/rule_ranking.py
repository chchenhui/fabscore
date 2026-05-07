"""LogRules rule ranking: prioritize rules by usage frequency on training examples.
Embeds all rules and K training examples into a prompt, asks gpt-4o-mini to
annotate which rules apply to each example, tallies frequency, returns top N.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from openai import OpenAI


RANKING_PROMPT_TEMPLATE = """You are an expert in log parsing. Below are a set of rules for abstracting variables in log messages, followed by training examples.

Rules:
{numbered_rules}

Training examples:
{examples_block}

For each training example, identify which rule numbers (from the list above) were applied to produce the template from the raw log. Return your answer as a JSON object where each key is the example number (starting from 1) and the value is a list of rule numbers that were applied.

Return only the JSON object, nothing else. Example format:
{{"1": [1, 3, 5], "2": [2, 4], "3": [1]}}"""


def _format_numbered_rules(rules: List[str]) -> str:
    return "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))


def _format_examples_for_ranking(examples: List[Dict]) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"Example {i+1}:")
        lines.append(f"  Raw log: {ex['raw_log']}")
        lines.append(f"  Log template: {ex['template']}")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_ranking_response(text: str, num_rules: int) -> Optional[Dict[str, List[int]]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, list):
                    valid_ids = [int(x) for x in v if isinstance(x, (int, float)) and 1 <= int(x) <= num_rules]
                    result[str(k)] = valid_ids
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def rank_rules(
    client: OpenAI,
    rules: List[str],
    induction_examples: List[Dict],
    model: str = "gpt-4o-mini",
    max_rules: int = 15,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> Dict:
    if len(rules) <= max_rules:
        return {
            "ranked_rules": rules,
            "rule_frequencies": {str(i+1): 0 for i in range(len(rules))},
            "num_original": len(rules),
            "num_selected": len(rules),
            "truncated": False,
        }

    numbered_rules = _format_numbered_rules(rules)
    examples_block = _format_examples_for_ranking(induction_examples)

    prompt = RANKING_PROMPT_TEMPLATE.format(
        numbered_rules=numbered_rules,
        examples_block=examples_block,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content or ""

    ranking_data = _parse_ranking_response(raw_text, len(rules))

    if ranking_data is None:
        print(f"WARNING: Failed to parse ranking response. Using first {max_rules} rules.")
        return {
            "ranked_rules": rules[:max_rules],
            "rule_frequencies": {},
            "raw_response": raw_text,
            "num_original": len(rules),
            "num_selected": max_rules,
            "truncated": True,
            "parse_failed": True,
        }

    counter = Counter()
    for example_id, rule_ids in ranking_data.items():
        for rid in rule_ids:
            counter[rid] += 1

    sorted_rule_ids = sorted(range(1, len(rules) + 1), key=lambda x: -counter.get(x, 0))
    selected_ids = sorted_rule_ids[:max_rules]

    ranked_rules = [rules[rid - 1] for rid in selected_ids]
    frequencies = {str(rid): counter.get(rid, 0) for rid in selected_ids}

    return {
        "ranked_rules": ranked_rules,
        "rule_frequencies": frequencies,
        "all_frequencies": {str(rid): counter.get(rid, 0) for rid in range(1, len(rules) + 1)},
        "raw_response": raw_text,
        "num_original": len(rules),
        "num_selected": len(ranked_rules),
        "truncated": True,
    }
