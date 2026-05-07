# LLM auditor accuracy pipeline: feeds top-k extracted tokens to an auditor
# model to determine if the secret can be correctly identified.
# Supports both local vLLM deployment (via OpenAI-compatible API) and
# the LEMMA_MAAS proxy service.

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmark")

DEFAULT_TEMPLATE_PATH = os.path.join(
    BENCHMARK_DIR, "prompts", "taboo", "auditor_guess_logit_lens_tokens_single.txt"
)


def load_template(template_path: str) -> str:
    with open(template_path, "r") as f:
        return f.read().strip()


def format_tokens_for_prompt(ranked_tokens: List[Dict[str, Any]], num_tokens: int = 5) -> str:
    if not ranked_tokens:
        return "No token information available."
    lines = []
    for t in ranked_tokens[:num_tokens]:
        token = t.get("token", "").strip()
        rank = t.get("rank", lines.__len__() + 1)
        score = t.get("score", t.get("probability", 0))
        lines.append(f"{rank}. \"{token}\" (score: {score:.4f})")
    return "\n".join(lines)


def build_auditor_prompt(
    template: str,
    user_prompt: str,
    model_response: str,
    ranked_tokens: List[Dict[str, Any]],
    num_tokens: int = 5,
) -> str:
    tokens_str = format_tokens_for_prompt(ranked_tokens, num_tokens)
    return template.format(
        user_prompt=user_prompt,
        model_response=model_response,
        top_tokens=tokens_str,
    )


def call_auditor(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Auditor call error: {e}")
        return ""


def evaluate_auditor_accuracy(
    scored_results: List[Dict[str, Any]],
    secret_word: str,
    client: OpenAI,
    auditor_model: str,
    template: str,
    num_tokens: int = 5,
    temperature: float = 0.0,
    batch_delay: float = 0.05,
) -> Dict[str, Any]:
    """Run auditor on all scored examples and compute accuracy."""
    correct = 0
    total = 0
    guesses = []

    secret_lower = secret_word.lower().strip()

    for item in tqdm(scored_results, desc=f"Auditor ({secret_word})"):
        if "error" in item:
            continue

        user_prompt = item.get("user_prompt", "")
        model_response = item.get("model_response", "")
        ranked_tokens = item.get("ranked_tokens", [])

        prompt = build_auditor_prompt(
            template, user_prompt, model_response, ranked_tokens, num_tokens
        )

        guess = call_auditor(client, auditor_model, prompt, temperature)
        is_correct = guess == secret_lower

        if is_correct:
            correct += 1
        total += 1

        guesses.append({
            "user_prompt": user_prompt[:100],
            "guess": guess,
            "correct": is_correct,
        })

        if batch_delay > 0:
            time.sleep(batch_delay)

    accuracy = correct / total if total > 0 else 0.0

    return {
        "secret_word": secret_word,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "guesses": guesses,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM auditor accuracy pipeline")
    parser.add_argument("--scored_files", type=str, nargs="+", required=True)
    parser.add_argument("--secret_words", type=str, nargs="+", required=True)
    parser.add_argument("--auditor_model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--api_base_url", type=str, default=None,
                        help="OpenAI-compatible API base URL (default: from .env)")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--template_path", type=str, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--num_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    api_base = args.api_base_url or f"http://{os.environ.get('LEMMA_MAAS_BASE_URL', 'localhost:8001')}/v1"
    api_key = args.api_key or os.environ.get("LEMMA_MAAS_API_KEY", "dummy")

    client = OpenAI(api_key=api_key, base_url=api_base)
    template = load_template(args.template_path)

    import numpy as np
    all_results = []

    for scored_file, secret_word in zip(args.scored_files, args.secret_words):
        print(f"\nEvaluating {scored_file} (secret: {secret_word})")
        with open(scored_file, "r") as f:
            data = json.load(f)
        scored_results = data.get("results", [])

        result = evaluate_auditor_accuracy(
            scored_results, secret_word, client, args.auditor_model,
            template, args.num_tokens, args.temperature,
        )
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    accuracies = [r["accuracy"] for r in all_results]
    summary = {
        "auditor_accuracy_mean": float(np.mean(accuracies)),
        "auditor_accuracy_std": float(np.std(accuracies)),
        "auditor_model": args.auditor_model,
        "num_tokens": args.num_tokens,
    }

    print(f"\nSummary: auditor accuracy = {summary['auditor_accuracy_mean']:.4f} +/- {summary['auditor_accuracy_std']:.4f}")

    output = {
        "per_model": [{k: v for k, v in r.items() if k != "guesses"} for r in all_results],
        "all_guesses": {r["secret_word"]: r["guesses"] for r in all_results},
        "summary": summary,
    }

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output_path}")

    return output


if __name__ == "__main__":
    main()
