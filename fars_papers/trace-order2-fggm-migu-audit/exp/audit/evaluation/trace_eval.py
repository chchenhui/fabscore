# Per-task TRACE evaluation using vLLM for generation + task-specific metrics.
# Deploys a vLLM server, generates predictions via async OpenAI API, computes metrics.
# Prompts are truncated to max_prompt_len tokens (default 1024) matching TRACE reference.
# Usage: python trace_eval.py --base_url <url> --model_name <name> --data_path <path> --tasks <t1,t2> --output_dir <dir>

import os
import sys
import json
import asyncio
import argparse
import random
from typing import List, Dict

from openai import AsyncOpenAI, RateLimitError, APIError
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.trace_data import TASK_NAMES, DEFAULT_ORDER
from evaluation.metrics import compute_task_metric, TASK_PRIMARY_METRIC


def load_test_data(data_path: str, task_name: str) -> List[Dict]:
    file_path = os.path.join(data_path, task_name, "test.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_prompts(tokenizer, prompts: List[str], max_prompt_len: int) -> List[str]:
    truncated = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
            p = tokenizer.decode(ids, skip_special_tokens=True)
        truncated.append(p)
    return truncated


def tokenize_prompts(tokenizer, prompts: List[str], max_prompt_len: int):
    token_ids_list = []
    dropped_suffixes = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        ids_with_a = tokenizer.encode(p + "A", add_special_tokens=False)
        if len(ids_with_a) > max_prompt_len:
            ids_with_a = ids_with_a[-max_prompt_len:]
        if len(ids) > 0 and len(ids_with_a) > 0 and ids[-1] != ids_with_a[len(ids) - 1]:
            dropped = tokenizer.decode([ids[-1]], skip_special_tokens=True)
            ids = ids[:-1]
            dropped_suffixes.append(dropped)
        else:
            dropped_suffixes.append(None)
        token_ids_list.append(ids)
    n_dropped = sum(1 for d in dropped_suffixes if d is not None)
    if n_dropped > 0:
        print(f"  Dropped trailing token for {n_dropped}/{len(prompts)} prompts (token-merge fix)")
    return token_ids_list, dropped_suffixes


async def generate_predictions(
    client: AsyncOpenAI,
    model_name: str,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_concurrent: int = 64,
    max_retries: int = 5,
    prompt_token_ids_list: List[List[int]] = None,
) -> List[str]:
    semaphore = asyncio.Semaphore(max_concurrent)
    use_token_ids = prompt_token_ids_list is not None

    async def generate_one(idx: int, prompt, token_ids=None) -> tuple:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    kwargs = dict(
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=1.0,
                    )
                    if use_token_ids and token_ids is not None:
                        kwargs["prompt"] = token_ids
                    else:
                        kwargs["prompt"] = prompt
                    response = await client.completions.create(**kwargs)
                    return idx, response.choices[0].text
                except (RateLimitError, APIError) as e:
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} retries for idx {idx}: {e}")
                        return idx, ""
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                except Exception as e:
                    print(f"Unexpected error for idx {idx}: {e}")
                    return idx, ""
        return idx, ""

    if use_token_ids:
        tasks_list = [generate_one(i, p, t) for i, (p, t) in enumerate(zip(prompts, prompt_token_ids_list))]
    else:
        tasks_list = [generate_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks_list)
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def evaluate_task(
    task_name: str,
    predicted_sequences: List[str],
    ground_truths: List[str],
    source_sequences: List[str] = None,
) -> Dict:
    return compute_task_metric(task_name, predicted_sequences, ground_truths, source_sequences)


async def run_evaluation(
    base_url: str,
    api_key: str,
    model_name: str,
    data_path: str,
    tasks: List[str],
    output_dir: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    tokenizer_path: str = None,
    max_prompt_len: int = 1024,
):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = None
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    results = {}
    for task_name in tasks:
        print(f"\nEvaluating {task_name}...")
        test_data = load_test_data(data_path, task_name)
        raw_prompts = [d["prompt"] for d in test_data]
        ground_truths = [d["answer"] for d in test_data]

        prompt_token_ids_list = None
        dropped_suffixes = None
        if tokenizer:
            prompt_token_ids_list, dropped_suffixes = tokenize_prompts(tokenizer, raw_prompts, max_prompt_len)
            prompts = truncate_prompts(tokenizer, raw_prompts, max_prompt_len)
            n_truncated = sum(1 for r, t in zip(raw_prompts, prompts) if r != t)
            if n_truncated > 0:
                print(f"  Truncated {n_truncated}/{len(prompts)} prompts to {max_prompt_len} tokens")
        else:
            prompts = raw_prompts

        predictions = await generate_predictions(
            client, model_name, prompts,
            max_tokens=max_tokens, temperature=temperature,
            prompt_token_ids_list=prompt_token_ids_list,
        )

        if dropped_suffixes:
            for i, suffix in enumerate(dropped_suffixes):
                if suffix is not None and predictions[i]:
                    pred = predictions[i]
                    if pred.startswith(suffix):
                        predictions[i] = pred[len(suffix):]

        pred_file = os.path.join(output_dir, f"predictions_{task_name}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "prompts": prompts,
                "predictions": predictions,
                "ground_truths": ground_truths,
            }, f, ensure_ascii=False, indent=2)

        eval_result = evaluate_task(task_name, predictions, ground_truths, prompts)
        results[task_name] = eval_result
        print(f"  {task_name}: {eval_result}")

    results_file = os.path.join(output_dir, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    asyncio.run(run_evaluation(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        data_path=args.data_path,
        tasks=tasks,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tokenizer_path=args.tokenizer_path or args.model_name,
        max_prompt_len=args.max_prompt_len,
    ))


if __name__ == "__main__":
    main()
