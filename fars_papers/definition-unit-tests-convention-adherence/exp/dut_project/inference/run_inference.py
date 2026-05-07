"""vLLM-based batched inference runner for ErdosConventionsBench.

Supports both raw text prompts (generate) and chat message format (chat).
Uses chat format for all conditions to leverage instruction-following.

Usage:
  python -m dut_project.inference.run_inference \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --condition C \
    --input dut_project/data/erdos_conventions_bench.jsonl \
    --output dut_project/outputs/qwen_cond_c.jsonl
"""

import argparse
import json
import os
from typing import Any

from vllm import LLM, SamplingParams

from dut_project.prompts.templates import CONDITION_CHAT_FORMATTERS
from dut_project.inference.parse_outputs import parse_output


def load_items(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def run_inference_chat(
    model_name: str,
    items: list[dict[str, Any]],
    condition: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    k: int = 3,
    num_samples: int = 1,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    top_p: float = 1.0,
) -> list[dict[str, Any]]:
    formatter = CONDITION_CHAT_FORMATTERS[condition]

    all_messages = []
    for item in items:
        if condition == "A":
            all_messages.append(formatter(item))
        else:
            all_messages.append(formatter(item, k=k))

    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_samples,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
    )

    outputs = llm.chat(all_messages, sampling_params)

    results = []
    for item, msgs, output in zip(items, all_messages, outputs):
        for sample_idx, completion in enumerate(output.outputs):
            text = completion.text
            parsed = parse_output(text, k=k)
            results.append({
                "item_id": item.get("item_id"),
                "family": item["family"],
                "condition": condition,
                "model": model_name,
                "sample_idx": sample_idx,
                "prompt": json.dumps(msgs),
                "raw_output": text,
                "parsed": parsed,
            })

    return results


def save_outputs(results: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            serializable = {k: v for k, v in r.items()}
            serializable["parsed"] = {
                k: v for k, v in r["parsed"].items() if k != "raw_text"
            }
            f.write(json.dumps(serializable) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C"])
    parser.add_argument("--input", required=True, help="Benchmark JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--k", type=int, default=3, help="Number of checks")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    items = load_items(args.input)
    results = run_inference_chat(
        model_name=args.model,
        items=items,
        condition=args.condition,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        k=args.k,
        num_samples=args.num_samples,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        top_p=args.top_p,
    )
    save_outputs(results, args.output)
    print(f"Saved {len(results)} outputs -> {args.output}")


if __name__ == "__main__":
    main()
