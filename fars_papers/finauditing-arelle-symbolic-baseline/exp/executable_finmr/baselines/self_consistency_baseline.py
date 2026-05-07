# Self-consistency LLM baseline for FinMR (Level 4).
# Calls a frontier LLM 4 times per instance with temperature=0.7,
# parses JSON from each completion, and majority-votes among valid answers.
# Uses asyncio + semaphore for concurrent API calls.

import asyncio
import json
import os
import re
from collections import Counter
from typing import Optional

from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI

from executable_finmr.data.load_finmr import FinMRInstance

load_dotenv()

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_N_SAMPLES = 4
MAX_CONCURRENT = 10
MAX_RETRIES = 5
RETRY_BASE_DELAY = 3.0
REQUEST_TIMEOUT = 120.0


def _build_prompt(instance: FinMRInstance) -> str:
    return instance.query


def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "extracted_value" in obj and "calculated_value" in obj:
        return {
            "extracted_value": str(obj["extracted_value"]),
            "calculated_value": str(obj["calculated_value"]),
        }
    return None


def _answer_key(parsed: dict) -> str:
    return f"{parsed['extracted_value']}|||{parsed['calculated_value']}"


def majority_vote(parsed_answers: list[dict]) -> str:
    if not parsed_answers:
        return ""
    keys = [_answer_key(a) for a in parsed_answers]
    counts = Counter(keys)
    max_count = max(counts.values())
    for key, ans in zip(keys, parsed_answers):
        if counts[key] == max_count:
            return ans
    return parsed_answers[0]


async def _call_llm_single(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = 42,
) -> str:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed + attempt,
                    timeout=REQUEST_TIMEOUT,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  API error after {MAX_RETRIES} retries: {e}")
                    return ""
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
        return ""


async def run_sc_single_instance(
    client: AsyncOpenAI,
    instance: FinMRInstance,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 42,
) -> dict:
    prompt = _build_prompt(instance)
    tasks = [
        _call_llm_single(
            client, prompt, semaphore, model=model, seed=seed + i
        )
        for i in range(n_samples)
    ]
    completions = await asyncio.gather(*tasks)

    parsed_answers = []
    raw_completions = []
    for comp in completions:
        raw_completions.append(comp)
        parsed = _parse_json_response(comp)
        if parsed is not None:
            parsed_answers.append(parsed)

    answer = majority_vote(parsed_answers)

    return {
        "id": instance.id,
        "dqc_id": instance.dqc_id,
        "raw_completions": raw_completions,
        "parsed_answers": parsed_answers,
        "majority_answer": answer,
        "prediction": answer if answer != "" else "",
        "gold": instance.gold_answer,
    }


async def run_sc_baseline_async(
    instances: list[FinMRInstance],
    model: str = DEFAULT_MODEL,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 42,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[dict]:
    client = AsyncOpenAI(
        api_key=os.environ.get("LEMMA_MAAS_API_KEY", ""),
        base_url=f"http://{os.environ.get('LEMMA_MAAS_BASE_URL', '')}/v1",
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=30.0),
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        run_sc_single_instance(client, inst, semaphore, model, n_samples, seed)
        for inst in instances
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


def run_sc_baseline(
    instances: list[FinMRInstance],
    model: str = DEFAULT_MODEL,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 42,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[dict]:
    return asyncio.run(
        run_sc_baseline_async(instances, model, n_samples, seed, max_concurrent)
    )
