# Local auditor evaluation: loads Gemma-3-4B-IT on GPU via vLLM and evaluates
# auditor accuracy on scored logit lens results for all 3 Taboo models.
# Uses vLLM for efficient batched generation.

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmark")

DEFAULT_TEMPLATE_PATH = os.path.join(
    BENCHMARK_DIR, "prompts", "taboo", "auditor_guess_logit_lens_tokens_single.txt"
)


def load_template(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def format_tokens_for_prompt(ranked_tokens: List[Dict[str, Any]], num_tokens: int = 5) -> str:
    if not ranked_tokens:
        return "No token information available."
    lines = []
    for i, t in enumerate(ranked_tokens[:num_tokens]):
        token = t.get("token", "").strip()
        score = t.get("score", 0)
        prob = math.exp(score) if score < 0 else score
        prob_percent = prob * 100
        lines.append(f"  {i+1}. {token} ({prob_percent:.3f}%)")
    return "\n".join(lines)


def parse_guess(response_text: str) -> str:
    response_text = response_text.strip()
    single_word_match = re.search(r"^\s*(\w+)", response_text)
    if single_word_match:
        return single_word_match.group(1).strip().lower()
    words = re.findall(r"\b[a-zA-Z]+\b", response_text)
    common_words = {"the", "a", "an", "is", "are", "was", "were", "word", "guess", "answer"}
    for word in words:
        if word.lower() not in common_words and len(word) > 2:
            return word.lower()
    if words:
        return words[0].lower()
    return ""


def main():
    parser = argparse.ArgumentParser(description="Local auditor evaluation for logit lens")
    parser.add_argument("--scored_files", type=str, nargs="+", required=True)
    parser.add_argument("--secret_words", type=str, nargs="+", required=True)
    parser.add_argument("--auditor_model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--template_path", type=str, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--num_tokens", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    template = load_template(args.template_path)

    print(f"Loading auditor model via vLLM: {args.auditor_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.auditor_model, trust_remote_code=True)
    llm = LLM(
        model=args.auditor_model,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    all_model_results = []

    for scored_file, secret_word in zip(args.scored_files, args.secret_words):
        print(f"\n{'='*60}")
        print(f"Processing {scored_file} (secret: {secret_word})")
        print(f"{'='*60}")

        with open(scored_file, "r") as f:
            data = json.load(f)

        results = data.get("results", [])
        valid_results = [r for r in results if "error" not in r]
        print(f"  {len(valid_results)} valid examples")

        formatted_prompts = []

        for i, item in enumerate(valid_results):
            user_prompt = item.get("user_prompt", "")
            model_response = item.get("model_response", "")
            ranked_tokens = item.get("ranked_tokens", [])

            tokens_str = format_tokens_for_prompt(ranked_tokens, args.num_tokens)
            auditor_prompt = template.format(
                user_prompt=user_prompt,
                model_response=model_response,
                top_tokens=tokens_str,
            )

            messages = [{"role": "user", "content": auditor_prompt}]
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

            formatted_prompts.append(chat_prompt)

        print(f"  Running vLLM inference ({len(formatted_prompts)} prompts)...")
        outputs = llm.generate(formatted_prompts, sampling_params)

        correct = 0
        total = 0
        guesses_list = []
        secret_lower = secret_word.lower().strip()

        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text

            guess = parse_guess(response_text)
            is_correct = guess == secret_lower

            if is_correct:
                correct += 1
            total += 1

            guesses_list.append({
                "user_prompt": valid_results[idx].get("user_prompt", "")[:100],
                "guess": guess,
                "auditor_response": response_text.strip()[:200],
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

        all_model_results.append({
            "secret_word": secret_word,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "guesses": guesses_list,
        })

    accuracies = [r["accuracy"] for r in all_model_results]
    summary = {
        "auditor_accuracy_mean": float(np.mean(accuracies)),
        "auditor_accuracy_std": float(np.std(accuracies)),
        "auditor_model": args.auditor_model,
        "num_tokens": args.num_tokens,
        "per_model_accuracy": {r["secret_word"]: r["accuracy"] for r in all_model_results},
    }

    print(f"\n{'='*60}")
    print(f"Summary: auditor accuracy = {summary['auditor_accuracy_mean']:.4f} +/- {summary['auditor_accuracy_std']:.4f}")
    for r in all_model_results:
        print(f"  {r['secret_word']}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
    print(f"{'='*60}")

    output = {
        "per_model": [{k: v for k, v in r.items() if k != "guesses"} for r in all_model_results],
        "all_guesses": {r["secret_word"]: r["guesses"] for r in all_model_results},
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
