"""Best-of-5 inference-time scaling for log parsing.
Generates n=5 candidate templates per log at temperature>0, selects the first
that passes a template-format verifier, falling back to the first candidate.
"""

import asyncio
import re
from typing import List

from openai import AsyncOpenAI

from logrules_poisoning.src.deduction.parse import clean_prediction


def is_valid_template(template: str, raw_log: str) -> bool:
    if not template or not template.strip():
        return False

    template = template.strip()
    tokens = template.split()

    if not tokens:
        return False

    placeholder_pattern = re.compile(r"<\*[^>]*>")
    placeholders = placeholder_pattern.findall(template)
    for ph in placeholders:
        if ph != "<*>":
            return False

    non_wildcard = [t for t in tokens if t != "<*>"]
    if len(non_wildcard) == 0:
        return False

    log_tokens = set(raw_log.split())
    for t in non_wildcard:
        if t not in log_tokens:
            return False

    return True


async def _parse_single_best_of_n(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    raw_log: str,
    semaphore: asyncio.Semaphore,
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            candidates = [
                clean_prediction(choice.message.content or "")
                for choice in response.choices
            ]

            for c in candidates:
                if is_valid_template(c, raw_log):
                    return c

            return candidates[0] if candidates else ""
        except Exception as e:
            print(f"Error parsing log: {e}")
            return ""


async def parse_logs_best_of_5_async(
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    logs: List[str],
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[str]:
    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = []
    for log_line in logs:
        prompt = prompt_template.replace("{log_line}", log_line)
        tasks.append(
            _parse_single_best_of_n(
                client, model, prompt, log_line, semaphore,
                n=n, temperature=temperature, max_tokens=max_tokens,
            )
        )
    results = await asyncio.gather(*tasks)
    return list(results)


def parse_logs_best_of_5(
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    logs: List[str],
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[str]:
    return asyncio.run(
        parse_logs_best_of_5_async(
            api_base, api_key, model, prompt_template, logs,
            n=n, temperature=temperature, max_tokens=max_tokens,
            max_concurrency=max_concurrency,
        )
    )
