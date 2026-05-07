"""
LAVE CFG-constrained diffusion inference for LLaDA-8B-Instruct on BFCL-v3 Non-Live.
Uses LAVE generate() from CD4dLLM/generate_our.py with a Lark grammar for syntax-only
BFCL tool-call format. For irrelevance category, falls back to unconstrained generation
since grammar constraint cannot express "no function call" outputs.
"""
import argparse
import json
import os
import sys
import time
import random
import traceback

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
CD4DLLM_ROOT = EXP_ROOT / "CD4dLLM"
sys.path.insert(0, str(CD4DLLM_ROOT))

from transformers import AutoTokenizer, AutoModel
from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as generate_ours
from constrained_diffusion.eval.dllm.models.llada.generate_constrained import generate as generate_diffusion


def truncate_to_bracket(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    depth = 0
    for i, c in enumerate(text):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return text[:i + 1]
    return text


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


def run_unconstrained_inference(
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


def run_lave_inference(
    model,
    tokenizer,
    prompt_text: str,
    grammar_str: str,
    seed: int,
    max_tokens: int = 256,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.2,
    change_logits: bool = True,
    top_k_per_mask: int = 5,
    top_n_beam: int = 5,
    random_n_beam: int = 5,
    max_retry_num_total: int = 5,
    timeout: float = 600.0,
):
    set_seed(seed)

    input_ids = tokenizer(prompt_text)["input_ids"]
    prompt_len = len(input_ids)
    prompt_tensor = torch.tensor(input_ids).to(model.device).unsqueeze(0)

    start_time = time.monotonic()
    timed_out = False
    total_retry_num = 0

    try:
        x, total_retry_num, gen_start_time = generate_ours(
            model,
            tokenizer,
            prompt_tensor,
            input_len=prompt_len,
            grammar=grammar_str,
            steps=steps,
            gen_length=max_tokens,
            block_length=block_length,
            temperature=temperature,
            remasking="low_confidence",
            trace=False,
            change_logits=change_logits,
            top_k_per_mask=top_k_per_mask,
            top_n_beam=top_n_beam,
            random_n_beam=random_n_beam,
            max_retry_num_total=max_retry_num_total,
        )
        wall_time = time.monotonic() - start_time

        if x is None:
            raw_output = ""
        else:
            raw_output = tokenizer.batch_decode(
                x[:, prompt_tensor.shape[1]:], skip_special_tokens=True
            )[0]
    except Exception as e:
        wall_time = time.monotonic() - start_time
        raw_output = ""
        print(f"  ERROR in generate_ours: {e}")
        traceback.print_exc()

    return raw_output, wall_time, total_retry_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--change_logits", type=bool, default=True)
    parser.add_argument("--top_k_per_mask", type=int, default=10)
    parser.add_argument("--top_n_beam", type=int, default=10)
    parser.add_argument("--random_n_beam", type=int, default=10)
    parser.add_argument("--max_retry_num_total", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--data_path", type=str,
                        default=str(EXP_ROOT / "bfcl_cfg_diffusion" / "data" / "bfcl_nonlive_300.json"))
    parser.add_argument("--grammar_path", type=str,
                        default=str(EXP_ROOT / "bfcl_cfg_diffusion" / "grammars" / "bfcl_toolcall.lark"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_lave_categories", type=str, nargs="*", default=["irrelevance"],
                        help="Categories to run unconstrained (no grammar)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else EXP_ROOT / "bfcl_cfg_diffusion" / "results" / "lave_cfg"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{args.seed}.jsonl"

    with open(args.grammar_path) as f:
        grammar_str = f.read()
    print(f"Loaded grammar from {args.grammar_path} ({len(grammar_str)} chars)")

    skip_lave_cats = set(args.skip_lave_categories) if args.skip_lave_categories else set()
    print(f"Categories using unconstrained (no LAVE): {skip_lave_cats or 'none'}")

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
    print(f"Model loaded. Running LAVE CFG inference on {len(examples)} examples with seed={args.seed}")

    from bfcl_cfg_diffusion.inference.prompt_formatter import build_messages

    times = []
    for i, ex in enumerate(tqdm(examples, desc=f"LAVE seed {args.seed}")):
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

        category = ex.get("category", "")

        if category in skip_lave_cats:
            raw_output, wall_time = run_unconstrained_inference(
                model, tokenizer, prompt_text, args.seed,
                max_tokens=args.max_tokens, steps=args.steps,
                block_length=args.block_length, temperature=args.temperature,
            )
            total_retry_num = 0
            cleaned_output = raw_output.strip()
        else:
            raw_output, wall_time, total_retry_num = run_lave_inference(
                model, tokenizer, prompt_text, grammar_str, args.seed,
                max_tokens=args.max_tokens, steps=args.steps,
                block_length=args.block_length, temperature=args.temperature,
                change_logits=args.change_logits,
                top_k_per_mask=args.top_k_per_mask,
                top_n_beam=args.top_n_beam,
                random_n_beam=args.random_n_beam,
                max_retry_num_total=args.max_retry_num_total,
            )
            cleaned_output = truncate_to_bracket(raw_output)

        times.append(wall_time)

        result = {
            "id": ex["id"],
            "category": category,
            "result": cleaned_output,
            "wall_time": wall_time,
            "seed": args.seed,
            "total_retry_num": total_retry_num,
        }
        with open(output_path, "a") as f:
            print(json.dumps(result), file=f, flush=True)

        if (i + 1) % 10 == 0:
            avg_time = sum(times[-10:]) / min(10, len(times[-10:]))
            print(f"  [{i+1}/{len(examples)}] Last 10 avg time: {avg_time:.2f}s, total_retry: {total_retry_num}")

    if times:
        print(f"\nDone. Total examples: {len(examples)}, Mean time: {sum(times)/len(times):.2f}s")
    else:
        print(f"\nDone. All examples already completed.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
