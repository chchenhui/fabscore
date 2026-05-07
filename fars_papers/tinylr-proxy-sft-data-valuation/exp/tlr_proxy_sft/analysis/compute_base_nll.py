# Compute base-model NLL (cross-entropy) of Qwen2.5-1.5B on each of 12 candidate
# datasets WITHOUT any fine-tuning. Masks system/user tokens; scores assistant
# response tokens only. Each invocation processes a single dataset on a single GPU.
# Output: per-dataset JSON result file, then merge into CSV separately.

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_NAMES = [
    "AM-Thinking-v1-Distilled-math",
    "DeepMath-309K",
    "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
    "Maths-College",
    "OpenR1-Math",
    "QwQ-LongCoT-130K-math",
    "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard",
    "mathplus",
    "numinamath-cot",
    "numinamath1_5",
    "openmathinstruct-2",
]

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_TOKEN_ID = 77091
NEWLINE_ID = 198


def find_assistant_spans(input_ids: list[int]) -> list[tuple[int, int]]:
    spans = []
    i = 0
    while i < len(input_ids) - 2:
        if (input_ids[i] == IM_START_ID
                and input_ids[i + 1] == ASSISTANT_TOKEN_ID
                and input_ids[i + 2] == NEWLINE_ID):
            content_start = i + 3
            j = content_start
            while j < len(input_ids) and input_ids[j] != IM_END_ID:
                j += 1
            if j > content_start:
                spans.append((content_start, j))
            i = j + 1
        else:
            i += 1
    return spans


def format_sample(sample: dict) -> list[dict]:
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")
    user_content = instruction
    if inp:
        user_content = f"{instruction}\n{inp}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def compute_dataset_nll(
    model,
    tokenizer,
    data_path: str,
    cutoff_len: int = 4096,
    batch_size: int = 8,
    max_samples: int = -1,
    device: torch.device = None,
):
    with open(data_path) as f:
        dataset = json.load(f)

    if max_samples > 0:
        dataset = dataset[:max_samples]

    total_nll = 0.0
    total_tokens = 0
    n_samples_processed = 0
    loss_fn = CrossEntropyLoss(reduction="none")

    batch_input_ids = []
    batch_labels = []

    for idx, sample in enumerate(dataset):
        messages = format_sample(sample)
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        if len(input_ids) > cutoff_len:
            input_ids = input_ids[:cutoff_len]

        spans = find_assistant_spans(input_ids)
        if not spans:
            continue

        labels = [-100] * len(input_ids)
        for start, end in spans:
            actual_end = min(end, len(input_ids))
            for k in range(start, actual_end):
                labels[k] = input_ids[k]

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

        if len(batch_input_ids) >= batch_size or idx == len(dataset) - 1:
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_ids = []
            padded_labels = []
            attention_masks = []
            for ids, lbl in zip(batch_input_ids, batch_labels):
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [tokenizer.pad_token_id or 0] * pad_len)
                padded_labels.append(lbl + [-100] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
            label_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
            attn_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(input_ids=input_tensor, attention_mask=attn_tensor)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = label_tensor[:, 1:].contiguous()

            loss_per_token = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss_per_token = loss_per_token.view(shift_labels.size())

            valid_mask = shift_labels != -100
            batch_nll = loss_per_token[valid_mask].sum().item()
            batch_tokens = valid_mask.sum().item()

            total_nll += batch_nll
            total_tokens += batch_tokens
            n_samples_processed += len(batch_input_ids)

            batch_input_ids = []
            batch_labels = []

        if (idx + 1) % 5000 == 0:
            avg = total_nll / total_tokens if total_tokens > 0 else 0
            print(f"  [{idx+1}/{len(dataset)}] running avg NLL: {avg:.4f}, tokens: {total_tokens}")

    mean_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    return mean_nll, total_tokens, n_samples_processed


def run_single(args):
    gpu_id = args.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    data_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    print(f"[GPU {gpu_id}] Computing NLL for {args.dataset}...")
    t0 = time.time()
    mean_nll, n_tokens, n_samples = compute_dataset_nll(
        model, tokenizer, data_path,
        cutoff_len=args.cutoff_len,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] {args.dataset}: NLL={mean_nll:.4f}, tokens={n_tokens}, "
          f"samples={n_samples}, time={elapsed:.1f}s")

    result = {
        "dataset": args.dataset,
        "mean_nll": mean_nll,
        "total_tokens": n_tokens,
        "n_samples": n_samples,
        "elapsed_seconds": elapsed,
    }
    output_path = Path(args.output_dir) / f"{args.dataset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[GPU {gpu_id}] Saved to {output_path}")


def merge_results(args):
    output_dir = Path(args.output_dir)
    rows = []
    for ds_name in DATASET_NAMES:
        result_path = output_dir / f"{ds_name}.json"
        if not result_path.exists():
            print(f"WARNING: {result_path} not found, skipping")
            continue
        with open(result_path) as f:
            result = json.load(f)
        rows.append({
            "dataset": result["dataset"],
            "mean_nll": result["mean_nll"],
            "total_tokens": result["total_tokens"],
            "n_samples": result["n_samples"],
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "mean_nll", "total_tokens", "n_samples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Base-Model NLL Scores ===")
    for row in sorted(rows, key=lambda r: r["mean_nll"]):
        print(f"  {row['dataset']}: NLL={row['mean_nll']:.4f} ({row['n_samples']} samples, {row['total_tokens']} tokens)")
    print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    run_parser.add_argument("--data-dir", type=str, required=True)
    run_parser.add_argument("--output-dir", type=str, required=True)
    run_parser.add_argument("--dataset", type=str, required=True)
    run_parser.add_argument("--gpu-id", type=int, default=0)
    run_parser.add_argument("--cutoff-len", type=int, default=4096)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--max-samples", type=int, default=-1)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--output-dir", type=str, required=True)
    merge_parser.add_argument("--output-csv", type=str, required=True)

    args = parser.parse_args()
    if args.command == "run":
        run_single(args)
    elif args.command == "merge":
        merge_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
