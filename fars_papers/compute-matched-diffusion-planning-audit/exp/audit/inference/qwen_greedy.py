"""Qwen2.5-7B greedy (temperature=0) inference on Countdown and Mini Sudoku.
Uses vLLM for efficient batch inference. Produces per-instance JSONL output.
"""

import json
import os
import argparse
from vllm import LLM, SamplingParams

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


def parse_answer(raw_output):
    text = raw_output.strip()
    first_line = text.split("\n")[0].strip()
    return first_line


def run_task(llm, task_name, data_file, template_file, output_file, max_tokens=64):
    instances = load_jsonl(data_file)
    template = load_template(template_file)
    prompts = build_prompts(template, instances)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["\n\n"],
    )

    print(f"Running {task_name}: {len(prompts)} instances, max_tokens={max_tokens}")
    outputs = llm.generate(prompts, sampling_params)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for inst, output in zip(instances, outputs):
            raw_text = output.outputs[0].text
            if task_name == "sudoku":
                parsed = raw_text.strip().split("\n\n")[0].strip()
            else:
                parsed = parse_answer(raw_text)
            record = {
                "id": inst["id"],
                "prompt": prompts[inst["id"]][:200] + "...",
                "output": raw_text,
                "parsed_answer": parsed,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(instances)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["countdown", "sudoku"],
                        choices=["countdown", "sudoku"])
    args = parser.parse_args()

    llm = LLM(
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
    )

    task_configs = {
        "countdown": {
            "data_file": os.path.join(BASE_DIR, "data", "countdown_test.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "countdown_8shot.txt"),
            "output_file": os.path.join(BASE_DIR, "results", "raw", "qwen_greedy_countdown.jsonl"),
            "max_tokens": 64,
        },
        "sudoku": {
            "data_file": os.path.join(BASE_DIR, "data", "sudoku_test.jsonl"),
            "template_file": os.path.join(BASE_DIR, "prompts", "sudoku_8shot.txt"),
            "output_file": os.path.join(BASE_DIR, "results", "raw", "qwen_greedy_sudoku.jsonl"),
            "max_tokens": 64,
        },
    }

    for task_name in args.tasks:
        cfg = task_configs[task_name]
        run_task(llm, task_name, **cfg)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
