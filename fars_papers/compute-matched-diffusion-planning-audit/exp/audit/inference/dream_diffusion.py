"""Dream-v0-Base-7B diffusion inference on Countdown and Mini Sudoku.
Uses HF transformers with trust_remote_code for Dream's diffusion_generate method.
Processes instances one at a time (batch_size=1) since diffusion generation is sequential.
Mirrors the approach in Dream's official eval_planning.py.
"""

import json
import os
import argparse
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_template(path):
    with open(path) as f:
        return f.read()


def build_prompts(template, instances):
    prompts = []
    for inst in instances:
        question = inst["question"].strip()
        full_prompt = template + "\n" + question + "\nOutput:"
        prompts.append(full_prompt)
    return prompts


def parse_countdown_answer(raw_output):
    text = raw_output.strip()
    first_line = text.split("\n")[0].strip()
    return first_line


def parse_sudoku_answer(raw_output):
    text = raw_output.strip()
    answer = text.split("\n\n")[0].strip()
    return answer


def run_task(model, tokenizer, task_name, data_file, template_file, output_file,
             max_new_tokens=64, limit=None):
    instances = load_jsonl(data_file)
    if limit is not None:
        instances = instances[:limit]
    template = load_template(template_file)
    prompts = build_prompts(template, instances)

    parse_fn = parse_sudoku_answer if task_name == "sudoku" else parse_countdown_answer

    print(f"Running {task_name}: {len(prompts)} instances, max_new_tokens={max_new_tokens}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []

    for i in trange(len(prompts), desc=task_name):
        inputs = tokenizer(
            [prompts[i]],
            padding=True,
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.diffusion_generate(
                inputs=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                diffusion_steps=max_new_tokens,
                temperature=0,
                top_p=1,
                alg="entropy",
                alg_temp=0,
            )

        gen_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = parse_fn(raw_text)

        record = {
            "id": instances[i]["id"],
            "prompt": prompts[i][:200] + "...",
            "output": raw_text,
            "parsed_answer": parsed,
        }
        results.append(record)

    with open(output_file, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["countdown", "sudoku"],
                        choices=["countdown", "sudoku"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of instances per task (for debugging)")
    args = parser.parse_args()

    print("Loading Dream-org/Dream-v0-Base-7B ...")
    model = AutoModel.from_pretrained(
        "Dream-org/Dream-v0-Base-7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Base-7B",
        trust_remote_code=True,
    )
    print(f"Model loaded on {next(model.parameters()).device}")

    task_configs = {
        "countdown": {
            "data_file": os.path.join(BASE_DIR, "data", "countdown_test.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "countdown_8shot.txt"),
            "output_file": os.path.join(BASE_DIR, "results", "raw", "dream_diffusion_countdown.jsonl"),
            "max_new_tokens": 64,
        },
        "sudoku": {
            "data_file": os.path.join(BASE_DIR, "data", "sudoku_test.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "sudoku_8shot.txt"),
            "output_file": os.path.join(BASE_DIR, "results", "raw", "dream_diffusion_sudoku.jsonl"),
            "max_new_tokens": 64,
        },
    }

    for task_name in args.tasks:
        cfg = task_configs[task_name]
        run_task(model, tokenizer, task_name, limit=args.limit, **cfg)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
