"""Log parsing via LLM deduction using OpenAI-compatible API.
Sends log lines to a vLLM-served model, extracts predicted templates.
Uses async requests with concurrency for throughput.
"""

import asyncio
import re
from typing import List, Optional

from openai import AsyncOpenAI


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^`", "", text)
    text = re.sub(r"`$", "", text)
    text = re.sub(r"^[Ll]og [Tt]emplate:\s*", "", text)
    text = text.strip()
    return text


async def _parse_single(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 2048,
) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content or ""
            return clean_prediction(raw)
        except Exception as e:
            print(f"Error parsing log: {e}")
            return ""


async def parse_logs_async(
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    logs: List[str],
    rules: Optional[str] = None,
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[str]:
    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    prompts = []
    for log_line in logs:
        p = prompt_template.replace("{log_line}", log_line)
        if rules and "{rules}" in prompt_template:
            p = p.replace("{rules}", rules)
        prompts.append(p)

    tasks = [
        _parse_single(client, model, p, semaphore, temperature, top_p, max_tokens)
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


def parse_logs(
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    logs: List[str],
    rules: Optional[str] = None,
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[str]:
    return asyncio.run(
        parse_logs_async(
            api_base, api_key, model, prompt_template, logs, rules,
            temperature, top_p, max_tokens, max_concurrency,
        )
    )
