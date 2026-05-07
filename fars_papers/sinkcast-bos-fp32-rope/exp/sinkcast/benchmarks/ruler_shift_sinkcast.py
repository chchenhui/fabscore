# RULER benchmark evaluation with position-shift protocol + SinkCast correction.
# Uses new hook infrastructure (sinkcast_hooks) that preserves HF's exact
# attention path and applies SinkCast correction on top, ensuring d=0 accuracy
# is identical to the BF16 baseline.

import argparse
import json
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sinkcast.core.rope_utils import extract_rope_config
from sinkcast.core.sinkcast_hooks import find_attention_layers, sinkcast_hooks
from sinkcast.models.loader import load_model_and_tokenizer

RULER_DATA = PROJECT_ROOT / "RULER" / "scripts" / "data" / "synthetic" / "json"
ESSAYS_PATH = RULER_DATA / "PaulGrahamEssays.json"
SQUAD_PATH = RULER_DATA / "squad.json"

NIAH_TEMPLATE = (
    "Some special pass keys are hidden within the following text. "
    "Find the pass key and memorize it. I will quiz you about the pass key at the end.\n\n"
    "{context}\n\n"
    "What is the pass key associated with \"{key}\"? "
    "The pass key associated with \"{key}\" is"
)

VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignments in the following text.\n\n"
    "{context}\n\n"
    "Question: What is the value assigned to {query_var}?"
)

QA_TEMPLATE = (
    "Answer the question based on the given passage. "
    "Only give me the answer and do not output any other words.\n\n"
    "{context}\n\n"
    "Question: {question}\nAnswer:"
)


def load_haystack_text() -> str:
    with open(ESSAYS_PATH) as f:
        essays = json.load(f)
    return " ".join(essays)


def load_squad_qa() -> List[Dict]:
    with open(SQUAD_PATH) as f:
        data = json.load(f)
    qa_pairs = []
    for article in data.get("data", data) if isinstance(data, dict) else data:
        if isinstance(article, dict) and "paragraphs" in article:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para.get("qas", []):
                    if qa.get("answers"):
                        answers = list(set(a["text"] for a in qa["answers"]))
                        qa_pairs.append({"context": context, "question": qa["question"], "answers": answers})
        elif isinstance(article, dict) and "context" in article:
            answers = article.get("answers", [])
            if isinstance(answers, list) and answers:
                if isinstance(answers[0], dict):
                    answers = list(set(a["text"] for a in answers))
                qa_pairs.append({"context": article["context"], "question": article["question"], "answers": answers})
    return qa_pairs


def generate_passkey():
    return "".join(random.choices(string.digits, k=5))


def build_haystack_tokens(tokenizer, target_len: int, text: str) -> list:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < target_len:
        repeats = (target_len // len(tokens)) + 1
        tokens = (tokens * repeats)[:target_len]
    else:
        tokens = tokens[:target_len]
    return tokens


def generate_niah_single(tokenizer, haystack_text: str, seq_len: int, n_samples: int) -> List[Dict]:
    samples = []
    for _ in range(n_samples):
        key_name = "city_" + "".join(random.choices(string.ascii_lowercase, k=4))
        passkey = generate_passkey()
        needle = f'The pass key associated with "{key_name}" is {passkey}.'
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        prompt_shell = NIAH_TEMPLATE.format(context="", key=key_name)
        shell_tokens = tokenizer.encode(prompt_shell, add_special_tokens=False)
        available = seq_len - len(shell_tokens) - len(needle_tokens) - 20
        if available < 100:
            continue
        hay_tokens = build_haystack_tokens(tokenizer, available, haystack_text)
        insert_pos = random.randint(len(hay_tokens) // 4, 3 * len(hay_tokens) // 4)
        hay_tokens = hay_tokens[:insert_pos] + needle_tokens + hay_tokens[insert_pos:]
        hay_tokens = hay_tokens[:available]
        context_text = tokenizer.decode(hay_tokens, skip_special_tokens=True)
        prompt = NIAH_TEMPLATE.format(context=context_text, key=key_name)
        samples.append({
            "task": "niah_single",
            "prompt": prompt,
            "answer": passkey,
            "metric": "string_match",
        })
    return samples


def generate_niah_multikey(tokenizer, haystack_text: str, seq_len: int, n_samples: int) -> List[Dict]:
    samples = []
    for _ in range(n_samples):
        keys_and_vals = []
        for _ in range(2):
            kn = "item_" + "".join(random.choices(string.ascii_lowercase, k=4))
            pv = generate_passkey()
            keys_and_vals.append((kn, pv))
        needles = [f'The pass key associated with "{k}" is {v}.' for k, v in keys_and_vals]
        query_idx = random.randint(0, len(keys_and_vals) - 1)
        query_key = keys_and_vals[query_idx][0]
        answer = keys_and_vals[query_idx][1]
        all_needle_tokens = []
        for n in needles:
            all_needle_tokens.append(tokenizer.encode(n, add_special_tokens=False))
        total_needle_len = sum(len(t) for t in all_needle_tokens)
        prompt_shell = NIAH_TEMPLATE.format(context="", key=query_key)
        shell_tokens = tokenizer.encode(prompt_shell, add_special_tokens=False)
        available = seq_len - len(shell_tokens) - total_needle_len - 20
        if available < 100:
            continue
        hay_tokens = build_haystack_tokens(tokenizer, available, haystack_text)
        positions = sorted(random.sample(range(len(hay_tokens) // 4, 3 * len(hay_tokens) // 4), len(needles)))
        offset = 0
        for i, pos in enumerate(positions):
            real_pos = pos + offset
            hay_tokens = hay_tokens[:real_pos] + all_needle_tokens[i] + hay_tokens[real_pos:]
            offset += len(all_needle_tokens[i])
        hay_tokens = hay_tokens[:available + total_needle_len]
        context_text = tokenizer.decode(hay_tokens, skip_special_tokens=True)
        prompt = NIAH_TEMPLATE.format(context=context_text, key=query_key)
        samples.append({
            "task": "niah_multikey",
            "prompt": prompt,
            "answer": answer,
            "metric": "string_match",
        })
    return samples


def generate_variable_tracing(tokenizer, haystack_text: str, seq_len: int, n_samples: int) -> List[Dict]:
    samples = []
    for _ in range(n_samples):
        n_chains = 2
        n_hops = 2
        chains = []
        all_assignments = []
        for c in range(n_chains):
            value = "".join(random.choices(string.ascii_uppercase, k=3))
            chain_vars = [f"VAR_{c}_{h}" for h in range(n_hops + 1)]
            assignments = [f"{chain_vars[0]} = {value}"]
            for h in range(1, n_hops + 1):
                assignments.append(f"{chain_vars[h]} = {chain_vars[h-1]}")
            chains.append({"vars": chain_vars, "value": value, "assignments": assignments})
            all_assignments.extend(assignments)
        random.shuffle(all_assignments)
        assignments_text = " ; ".join(all_assignments)
        assign_tokens = tokenizer.encode(assignments_text, add_special_tokens=False)
        query_chain = random.choice(chains)
        query_var = query_chain["vars"][-1]
        answer = query_chain["value"]
        prompt_shell = VT_TEMPLATE.format(context="", query_var=query_var)
        shell_tokens = tokenizer.encode(prompt_shell, add_special_tokens=False)
        available = seq_len - len(shell_tokens) - len(assign_tokens) - 20
        if available < 100:
            continue
        hay_tokens = build_haystack_tokens(tokenizer, available, haystack_text)
        insert_pos = random.randint(len(hay_tokens) // 4, 3 * len(hay_tokens) // 4)
        hay_tokens = hay_tokens[:insert_pos] + assign_tokens + hay_tokens[insert_pos:]
        hay_tokens = hay_tokens[:available + len(assign_tokens)]
        context_text = tokenizer.decode(hay_tokens, skip_special_tokens=True)
        prompt = VT_TEMPLATE.format(context=context_text, query_var=query_var)
        samples.append({
            "task": "variable_tracing",
            "prompt": prompt,
            "answer": answer,
            "metric": "exact_match",
        })
    return samples


def normalize_answer(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
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


def generate_qa(tokenizer, haystack_text: str, seq_len: int, n_samples: int) -> List[Dict]:
    qa_pairs = load_squad_qa()
    random.shuffle(qa_pairs)
    samples = []
    for qa in qa_pairs[:n_samples * 3]:
        if len(samples) >= n_samples:
            break
        passage = qa["context"]
        question = qa["question"]
        answers = qa["answers"]
        passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
        prompt_shell = QA_TEMPLATE.format(context="", question=question)
        shell_tokens = tokenizer.encode(prompt_shell, add_special_tokens=False)
        available = seq_len - len(shell_tokens) - len(passage_tokens) - 20
        if available < 50:
            continue
        hay_tokens = build_haystack_tokens(tokenizer, available, haystack_text)
        insert_pos = random.randint(0, len(hay_tokens))
        hay_tokens = hay_tokens[:insert_pos] + passage_tokens + hay_tokens[insert_pos:]
        hay_tokens = hay_tokens[:available + len(passage_tokens)]
        context_text = tokenizer.decode(hay_tokens, skip_special_tokens=True)
        prompt = QA_TEMPLATE.format(context=context_text, question=question)
        samples.append({
            "task": "qa",
            "prompt": prompt,
            "answer": answers[0],
            "all_answers": answers,
            "metric": "f1",
        })
    return samples


def generate_all_tasks(tokenizer, seq_len: int, n_samples: int = 50) -> List[Dict]:
    haystack_text = load_haystack_text()
    all_samples = []
    all_samples.extend(generate_niah_single(tokenizer, haystack_text, seq_len, n_samples))
    all_samples.extend(generate_niah_multikey(tokenizer, haystack_text, seq_len, n_samples))
    all_samples.extend(generate_variable_tracing(tokenizer, haystack_text, seq_len, n_samples))
    all_samples.extend(generate_qa(tokenizer, haystack_text, seq_len, n_samples))
    return all_samples


def evaluate_sample(prediction: str, sample: Dict) -> float:
    metric = sample["metric"]
    if metric == "string_match":
        return 1.0 if sample["answer"] in prediction else 0.0
    elif metric == "exact_match":
        return 1.0 if normalize_answer(sample["answer"]) in normalize_answer(prediction) else 0.0
    elif metric == "f1":
        all_answers = sample.get("all_answers", [sample["answer"]])
        return max(f1_score(prediction, a) for a in all_answers)
    return 0.0


# --------------- SinkCast hook machinery ---------------



def run_inference_sinkcast(model, tokenizer, prompt: str, rope_config: dict,
                           attn_layers: list, K: int = 1,
                           position_offset: int = 0, shift_M: int = 0,
                           max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    device = next(model.parameters()).device
    T = input_ids.shape[1]
    offset = shift_M if shift_M > 0 else position_offset

    if offset > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_ids = torch.full((1, offset), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([pad_ids, input_ids], dim=1)
        attention_mask = torch.cat([
            torch.zeros(1, offset, dtype=torch.long),
            torch.ones(1, T, dtype=torch.long),
        ], dim=1)
        position_ids = torch.cat([
            torch.arange(offset, dtype=torch.long).unsqueeze(0),
            (torch.arange(T, dtype=torch.long) + offset).unsqueeze(0),
        ], dim=1)
        real_pos_ids = (torch.arange(T, dtype=torch.long) + offset).unsqueeze(0)
        real_mask = torch.cat([
            torch.zeros(1, offset, dtype=torch.bool),
            torch.ones(1, T, dtype=torch.bool),
        ], dim=1)
    else:
        attention_mask = torch.ones(1, T, dtype=torch.long)
        position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
        real_pos_ids = position_ids.clone()
        real_mask = None

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    real_pos_ids = real_pos_ids.to(device)
    if real_mask is not None:
        real_mask = real_mask.to(device)

    position_ids_ref = [real_pos_ids]
    real_mask_ref = [real_mask]

    with sinkcast_hooks(attn_layers, rope_config, position_ids_ref, real_mask_ref, K=K):
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

    prompt_len = input_ids.shape[1]
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_evaluation(model, tokenizer, samples: List[Dict], rope_config: dict,
                   attn_layers: list, K: int = 1, shift_M: int = 0,
                   max_new_tokens: int = 128, desc: str = "") -> List[Dict]:
    results = []
    for i, sample in enumerate(samples):
        t0 = time.time()
        prediction = run_inference_sinkcast(
            model, tokenizer, sample["prompt"],
            rope_config=rope_config, attn_layers=attn_layers, K=K,
            shift_M=shift_M, max_new_tokens=max_new_tokens,
        )
        score = evaluate_sample(prediction, sample)
        elapsed = time.time() - t0
        results.append({
            "task": sample["task"],
            "prediction": prediction,
            "answer": sample["answer"],
            "score": score,
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  [{desc}] {i+1}/{len(samples)}: score={score:.2f}, time={elapsed:.1f}s")
    return results


def aggregate_results(results: List[Dict]) -> Dict:
    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in results:
        task_scores[r["task"]].append(r["score"])
    summary = {}
    for task, scores in task_scores.items():
        summary[task] = {
            "accuracy": round(100.0 * np.mean(scores), 2),
            "n_samples": len(scores),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="RULER position-shift eval with SinkCast correction")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--K", type=int, default=1, help="Number of sink keys to correct")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--shift_M", type=int, default=4096)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
        args.output_dir = str(PROJECT_ROOT / "sinkcast" / "results" / "downstream" / f"sinkcast_k{args.K}")
    os.makedirs(args.output_dir, exist_ok=True)

    import wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "sinkcast-bos-fp32-rope")
    model_short = args.model.replace("/", "-")
    wandb.init(
        project=wandb_project,
        name=f"ruler-shift-sinkcast-k{args.K}-{model_short}",
        config={
            "model": args.model,
            "benchmark": "RULER",
            "method": f"sinkcast_k{args.K}",
            "K": args.K,
            "seq_lengths": args.seq_lengths,
            "shift_M": args.shift_M,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "batch_size": 1,
            "use_cache": False,
        },
    )

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.bfloat16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rope_config = extract_rope_config(model)
    attn_layers = find_attention_layers(model)
    print(f"Model loaded. SinkCast K={args.K}, {len(attn_layers)} attention layers hooked.")
    print(f"RoPE config: theta={rope_config['rope_theta']}, head_dim={rope_config['head_dim']}")

    all_results = {}
    for seq_len in args.seq_lengths:
        print(f"\n=== Sequence length: {seq_len} ===")
        random.seed(args.seed)
        np.random.seed(args.seed)
        n = args.debug_n if args.debug_n > 0 else args.n_samples
        samples = generate_all_tasks(tokenizer, seq_len, n_samples=n)

        if args.tasks:
            samples = [s for s in samples if s["task"] in args.tasks]

        print(f"Generated {len(samples)} samples")
        task_counts = {}
        for s in samples:
            task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
        print(f"  Task distribution: {task_counts}")

        print(f"\n--- Evaluating at delta=0 (SinkCast K={args.K}) ---")
        results_0 = run_evaluation(model, tokenizer, samples,
                                   rope_config=rope_config, attn_layers=attn_layers,
                                   K=args.K, shift_M=0,
                                   max_new_tokens=args.max_new_tokens,
                                   desc=f"len={seq_len},d=0")
        summary_0 = aggregate_results(results_0)

        print(f"\n--- Evaluating at delta=M={args.shift_M} (SinkCast K={args.K}) ---")
        results_M = run_evaluation(model, tokenizer, samples,
                                   rope_config=rope_config, attn_layers=attn_layers,
                                   K=args.K, shift_M=args.shift_M,
                                   max_new_tokens=args.max_new_tokens,
                                   desc=f"len={seq_len},d=M")
        summary_M = aggregate_results(results_M)

        comparison = {}
        for task in summary_0:
            acc_0 = summary_0[task]["accuracy"]
            acc_M = summary_M[task]["accuracy"]
            drop = round(acc_0 - acc_M, 2)
            comparison[task] = {
                "accuracy_delta0": acc_0,
                "accuracy_deltaM": acc_M,
                "drop": drop,
                "n_samples": summary_0[task]["n_samples"],
            }
            wandb.log({
                f"ruler/seq{seq_len}/{task}/acc_delta0": acc_0,
                f"ruler/seq{seq_len}/{task}/acc_deltaM": acc_M,
                f"ruler/seq{seq_len}/{task}/drop": drop,
            })

        all_results[str(seq_len)] = comparison

        print(f"\n--- Results for seq_len={seq_len} ---")
        for task, vals in comparison.items():
            print(f"  {task}: d=0={vals['accuracy_delta0']:.1f}, d=M={vals['accuracy_deltaM']:.1f}, drop={vals['drop']:.1f}")

    output_file = os.path.join(args.output_dir, f"ruler_{model_short}.json")
    with open(output_file, "w") as f:
        json.dump({"model": args.model, "shift_M": args.shift_M, "K": args.K,
                    "method": "sinkcast", "results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_file}")

    wandb.summary["results"] = all_results
    wandb.finish()


if __name__ == "__main__":
    main()
