#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read ./HumanEval.jsonl, call OpenAI Responses API in parallel (max 10 concurrent),
add a new field 'nl_request' converted from each 'prompt', and write to
./HumanEval_with_nl.jsonl. Other fields remain unchanged.

Deps:
  pip install --upgrade openai
Env:
  export OPENAI_API_KEY="sk-..."
"""

import os
import json
import asyncio
import random
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI

# ====== Config ======
INPUT_PATH = "HumanEval.jsonl"
OUTPUT_PATH = "HumanEval_with_nl.jsonl"
API_KEY = ""   
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-2025-08-07")
MAX_CONCURRENCY = 10
TEMPERATURE = 0.2
OUTPUT_FIELD = "nl_request"
# =====================

TEMPLATE = (
    'You are a rewriting assistant for "Vibe Programming". '
    "Pretend you are a real human user who wants to ask an LLM to write code. "
    "Look at the given code-style prompt and function signature, then express the same requirement "
    "in plain natural language, as if you were asking for help.\n\n"
    "The request must: (1) use only natural language; "
    "(2) preserve all requirements about inputs, outputs, behavior, edge cases, and constraints; "
    "(3) be 2–6 sentences; "
    '(4) directly address the model (e.g., "Please write a function that ..."); '
    "(5) not add anything beyond the original prompt.\n\n"
    "Return only the rewritten request as plain text.\n\n"
    "Task:\n\n"
    "<<<PROMPT>>>\n"
)

def build_input_text(humaneval_prompt: str) -> str:
    return TEMPLATE.replace("<<<PROMPT>>>", humaneval_prompt.strip())

async def call_api(client: AsyncOpenAI, prompt_text: str) -> str:
    """Call OpenAI Responses API with retries and return plain text output."""
    backoff = 1.0
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = await client.responses.create(
                model=MODEL,
                input=prompt_text,
            )
            # Preferred convenience property
            out = getattr(resp, "output_text", None)
            if out:
                return out.strip()
            # Fallback: dig into blocks
            try:
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", "") == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", "") in ("output_text", "text"):
                                return str(c.text).strip()
            except Exception:
                pass
            # Last resort
            return str(resp)
        except Exception as e:
            last_err = e
            if attempt == 4:
                raise
            await asyncio.sleep(backoff + random.random() * 0.5)
            backoff *= 2
    # Should not reach here
    raise RuntimeError(f"API call failed after retries: {last_err}")

async def process_one(obj: Dict[str, Any], client: AsyncOpenAI, sem: asyncio.Semaphore) -> Dict[str, Any]:
    """Process a single JSONL object: add OUTPUT_FIELD while keeping others unchanged."""
    prompt = obj.get("prompt", "")
    input_text = build_input_text(prompt)
    async with sem:
        nl = await call_api(client, input_text)
    new_obj = dict(obj)
    new_obj[OUTPUT_FIELD] = nl
    return new_obj

async def main_async():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Please export it or fill API_KEY in the script.")

    client = AsyncOpenAI(api_key=API_KEY)

    # Load JSONL
    items: List[Dict[str, Any]] = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Create tasks in order to preserve ordering in results
    tasks = [asyncio.create_task(process_one(obj, client, sem)) for obj in items]
    results = await asyncio.gather(*tasks)

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main_async())
