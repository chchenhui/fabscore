# Quick test: verify that passing prompt_token_ids to vLLM works correctly.
# Compares text-based prompts vs token-id-based prompts to confirm the fix.
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI
from transformers import AutoTokenizer

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "base_models", "Qwen2-1.5B")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trace_tasks", "TRACE-Benchmark", "LLM-CL-Benchmark_5000")

async def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "checkpoint"

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    fomc_path = os.path.join(DATA_PATH, "FOMC", "test.json")
    with open(fomc_path) as f:
        test_data = json.load(f)

    samples = test_data[:10]

    print("=== Comparing text-based vs token-id-based prompts ===\n")
    for i, sample in enumerate(samples):
        prompt_text = sample["prompt"]
        gt = sample["answer"]
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        print(f"--- Sample {i} (GT: {gt}) ---")
        print(f"  Prompt ends with: ...{prompt_text[-40:]!r}")
        print(f"  Last 5 token IDs: {token_ids[-5:]}")
        print(f"  Decoded last 5: {tokenizer.decode(token_ids[-5:])!r}")

        resp_text = await client.completions.create(
            model=model_name, prompt=prompt_text,
            max_tokens=20, temperature=0.1, top_p=1.0,
        )
        pred_text = resp_text.choices[0].text

        resp_ids = await client.completions.create(
            model=model_name, prompt=token_ids,
            max_tokens=20, temperature=0.1, top_p=1.0,
        )
        pred_ids = resp_ids.choices[0].text

        match_text = "OK" if gt.strip() in pred_text.strip() else "MISS"
        match_ids = "OK" if gt.strip() in pred_ids.strip() else "MISS"
        print(f"  Text prompt -> {pred_text.strip()!r:40s} [{match_text}]")
        print(f"  Token IDs   -> {pred_ids.strip()!r:40s} [{match_ids}]")
        print()

asyncio.run(main())
