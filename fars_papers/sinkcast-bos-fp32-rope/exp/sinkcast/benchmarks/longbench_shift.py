# LongBench benchmark evaluation with position-shift protocol.
# Evaluates a subset of English LongBench tasks at position offset 0
# and offset M to measure BF16 RoPE shift-invariance impact on real tasks.

import argparse
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sinkcast.models.loader import load_model_and_tokenizer

TASKS = {
    "narrativeqa": {"metric": "qa_f1", "max_gen": 128},
    "hotpotqa": {"metric": "qa_f1", "max_gen": 32},
    "gov_report": {"metric": "rouge_l", "max_gen": 512},
    "trec": {"metric": "classification", "max_gen": 64},
    "passage_retrieval_en": {"metric": "retrieval", "max_gen": 32},
}

PROMPT_TEMPLATES = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
}


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores["rougeL"].fmeasure


def classification_score(prediction: str, ground_truth: str, all_classes: List[str]) -> float:
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str) -> float:
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return right_num / len(numbers) if numbers else 0.0


def compute_score(task: str, prediction: str, ground_truths: List[str],
                  all_classes: List[str] = None) -> float:
    metric = TASKS[task]["metric"]
    if metric == "qa_f1":
        if task in ["trec"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        return max(qa_f1_score(prediction, gt) for gt in ground_truths)
    elif metric == "rouge_l":
        return max(rouge_l_score(prediction, gt) for gt in ground_truths)
    elif metric == "classification":
        prediction = prediction.lstrip("\n").split("\n")[0]
        return max(classification_score(prediction, gt, all_classes or []) for gt in ground_truths)
    elif metric == "retrieval":
        return max(retrieval_score(prediction, gt) for gt in ground_truths)
    return 0.0


def truncate_middle(tokenizer, prompt: str, max_length: int) -> str:
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) <= max_length:
        return prompt
    half = max_length // 2
    tokens = tokens[:half] + tokens[-half:]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def run_inference(model, tokenizer, prompt: str, shift_M: int = 0,
                  max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    device = next(model.parameters()).device
    T = input_ids.shape[1]

    if shift_M > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_ids = torch.full((1, shift_M), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([pad_ids, input_ids], dim=1)
        attention_mask = torch.cat([
            torch.zeros(1, shift_M, dtype=torch.long),
            torch.ones(1, T, dtype=torch.long),
        ], dim=1)
        position_ids = torch.cat([
            torch.arange(shift_M, dtype=torch.long).unsqueeze(0),
            (torch.arange(T, dtype=torch.long) + shift_M).unsqueeze(0),
        ], dim=1)
    else:
        attention_mask = torch.ones(1, T, dtype=torch.long)
        position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--shift_M", type=int, default=4096)
    parser.add_argument("--max_context", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--debug_n", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_dotenv(PROJECT_ROOT / ".env")

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "sinkcast" / "results" / "downstream" / "bf16_flash")
    os.makedirs(args.output_dir, exist_ok=True)

    import wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "sinkcast-bos-fp32-rope")
    model_short = args.model.replace("/", "-")
    wandb.init(
        project=wandb_project,
        name=f"longbench-shift-{model_short}",
        config={
            "model": args.model,
            "benchmark": "LongBench",
            "shift_M": args.shift_M,
            "max_context": args.max_context,
            "seed": args.seed,
            "batch_size": 1,
        },
    )

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.bfloat16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    task_list = args.tasks if args.tasks else list(TASKS.keys())
    all_results = {}

    for task_name in task_list:
        if task_name not in TASKS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_cfg = TASKS[task_name]
        print(f"\n=== Task: {task_name} (metric: {task_cfg['metric']}) ===")

        dataset = load_dataset("THUDM/LongBench", task_name, split="test", trust_remote_code=True)
        data = list(dataset)
        if args.debug_n > 0:
            data = data[:args.debug_n]

        print(f"Loaded {len(data)} samples")

        results_0 = []
        results_M = []

        for i, item in enumerate(data):
            context = item["context"]
            inp = item.get("input", "")
            answers = item["answers"] if isinstance(item["answers"], list) else json.loads(item["answers"])
            all_classes = item.get("all_classes", None)
            if all_classes is None:
                all_classes = []
            elif isinstance(all_classes, str):
                try:
                    all_classes = json.loads(all_classes)
                except (json.JSONDecodeError, TypeError):
                    all_classes = []

            template = PROMPT_TEMPLATES[task_name]
            prompt = template.replace("{context}", context.strip())
            if "{input}" in template:
                prompt = prompt.replace("{input}", inp.strip())

            prompt = truncate_middle(tokenizer, prompt, args.max_context)

            t0 = time.time()
            pred_0 = run_inference(model, tokenizer, prompt, shift_M=0,
                                    max_new_tokens=task_cfg["max_gen"])
            score_0 = compute_score(task_name, pred_0, answers, all_classes)
            results_0.append({"prediction": pred_0, "score": score_0})

            pred_M = run_inference(model, tokenizer, prompt, shift_M=args.shift_M,
                                    max_new_tokens=task_cfg["max_gen"])
            score_M = compute_score(task_name, pred_M, answers, all_classes)
            results_M.append({"prediction": pred_M, "score": score_M})

            elapsed = time.time() - t0
            if (i + 1) % 10 == 0:
                print(f"  [{task_name}] {i+1}/{len(data)} done (last: d0={score_0:.2f}, dM={score_M:.2f}, {elapsed:.1f}s)")

        acc_0 = round(100.0 * np.mean([r["score"] for r in results_0]), 2)
        acc_M = round(100.0 * np.mean([r["score"] for r in results_M]), 2)
        drop = round(acc_0 - acc_M, 2)

        all_results[task_name] = {
            "accuracy_delta0": acc_0,
            "accuracy_deltaM": acc_M,
            "drop": drop,
            "n_samples": len(data),
            "metric": task_cfg["metric"],
        }

        wandb.log({
            f"longbench/{task_name}/acc_delta0": acc_0,
            f"longbench/{task_name}/acc_deltaM": acc_M,
            f"longbench/{task_name}/drop": drop,
        })

        print(f"  {task_name}: d=0={acc_0:.1f}, d=M={acc_M:.1f}, drop={drop:.1f}")

    all_accs_0 = [v["accuracy_delta0"] for v in all_results.values()]
    all_accs_M = [v["accuracy_deltaM"] for v in all_results.values()]
    avg_drop = round(np.mean([v["drop"] for v in all_results.values()]), 2)
    all_results["overall"] = {
        "avg_accuracy_delta0": round(np.mean(all_accs_0), 2),
        "avg_accuracy_deltaM": round(np.mean(all_accs_M), 2),
        "avg_drop": avg_drop,
    }

    wandb.log({
        "longbench/overall/avg_acc_delta0": all_results["overall"]["avg_accuracy_delta0"],
        "longbench/overall/avg_acc_deltaM": all_results["overall"]["avg_accuracy_deltaM"],
        "longbench/overall/avg_drop": avg_drop,
    })

    output_file = os.path.join(args.output_dir, f"longbench_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump({"model": args.model, "shift_M": args.shift_M, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_file}")

    wandb.summary["results"] = all_results
    wandb.finish()


if __name__ == "__main__":
    main()
