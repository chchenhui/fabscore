"""
Unconstrained diffusion inference for LLaDA-8B-Instruct on BFCL-v3 Non-Live.
Uses the CD4dLLM semi-autoregressive generation with constrain=False.
Hyperparameters match LAVE defaults: max_tokens=256, steps=128, block_length=32, temp=0.2.
"""
import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
CD4DLLM_ROOT = EXP_ROOT / "CD4dLLM"
sys.path.insert(0, str(CD4DLLM_ROOT))

from transformers import AutoTokenizer, AutoModel
from constrained_diffusion.eval.dllm.models.llada.generate_constrained import generate as generate_diffusion


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt_text: str,
    seed: int,
    max_tokens: int = 256,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.2,
):
    set_seed(seed)

    input_ids = tokenizer(prompt_text)["input_ids"]
    prompt_len = len(input_ids)
    prompt_tensor = torch.tensor(input_ids).to(model.device).unsqueeze(0)

    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.get("mask_token", "[MASK]"))
    if mask_id is None:
        mask_id = 126336

    start_time = time.monotonic()
    out = None
    for out, resamples, valid, gen_start_time in generate_diffusion(
        model,
        prompt_tensor,
        tokenizer,
        prelex=None,
        constraint_lang=None,
        lex_map=None,
        prompt_len=prompt_len,
        steps=steps,
        gen_length=max_tokens,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
        trace=False,
        subtokens={},
        additional_stuff=None,
        strip_chars=None,
        max_total_injections=0,
        inject_gap_size=0,
        constrain=False,
    ):
        pass

    wall_time = time.monotonic() - start_time

    if out is None:
        raw_output = ""
    else:
        raw_output = tokenizer.batch_decode(
            out[:, prompt_tensor.shape[1]:], skip_special_tokens=True
        )[0]

    return raw_output, wall_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--data_path", type=str, default=str(EXP_ROOT / "bfcl_cfg_diffusion" / "data" / "bfcl_nonlive_300.json"))
    args = parser.parse_args()

    output_dir = EXP_ROOT / "bfcl_cfg_diffusion" / "results" / "unconstrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{args.seed}.jsonl"

    with open(args.data_path) as f:
        examples = json.load(f)

    if args.limit > 0:
        examples = examples[:args.limit]

    already_done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                already_done.add(entry["id"])
        print(f"Resuming: {len(already_done)} already done")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    print(f"Model loaded. Running inference on {len(examples)} examples with seed={args.seed}")

    from bfcl_cfg_diffusion.inference.prompt_formatter import build_messages

    times = []
    for i, ex in enumerate(tqdm(examples, desc=f"Seed {args.seed}")):
        if ex["id"] in already_done:
            continue

        messages = build_messages(ex)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages.pop(0)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        raw_output, wall_time = run_inference(
            model, tokenizer, prompt_text, args.seed,
            max_tokens=args.max_tokens, steps=args.steps,
            block_length=args.block_length, temperature=args.temperature,
        )
        times.append(wall_time)

        result = {
            "id": ex["id"],
            "category": ex["category"],
            "result": raw_output,
            "wall_time": wall_time,
            "seed": args.seed,
        }
        with open(output_path, "a") as f:
            print(json.dumps(result), file=f, flush=True)

        if (i + 1) % 10 == 0:
            avg_time = sum(times[-10:]) / min(10, len(times[-10:]))
            print(f"  [{i+1}/{len(examples)}] Last 10 avg time: {avg_time:.2f}s")

    print(f"\nDone. Total examples: {len(examples)}, Mean time: {sum(times)/max(len(times),1):.2f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
