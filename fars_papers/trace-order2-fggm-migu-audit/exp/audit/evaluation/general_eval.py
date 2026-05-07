# General ability evaluation via OpenCompass or vLLM-based evaluation.
# Evaluates on MMLU, BBH, TyDiQA, PIQA, BoolQ, GSM8K using the final checkpoint.
# Uses vLLM deployment for inference + lm-evaluation-harness or OpenCompass.

import os
import sys
import json
import subprocess
import argparse


GENERAL_BENCHMARKS = ["mmlu", "bbh", "tydiqa", "piqa", "boolq", "gsm8k"]

OPENCOMPASS_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "opencompass")


def run_opencompass_eval(
    model_path: str,
    output_dir: str,
    benchmarks: list = None,
    num_gpus: int = 8,
):
    if benchmarks is None:
        benchmarks = GENERAL_BENCHMARKS

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "eval_config.py")

    datasets_str = ", ".join([f'"{b}"' for b in benchmarks])
    config_content = f"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets
    from opencompass.configs.datasets.boolq.boolq_gen import boolq_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = mmlu_datasets + bbh_datasets + piqa_datasets + boolq_datasets + gsm8k_datasets

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-1.5b-sft',
        path='{model_path}',
        max_out_len=512,
        batch_size=16,
        run_cfg=dict(num_gpus={num_gpus}),
    )
]
"""
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"OpenCompass config written to {config_path}")
    print(f"Run manually: cd {OPENCOMPASS_DIR} && python run.py {config_path}")
    return config_path


def parse_opencompass_results(results_dir: str) -> dict:
    results = {}
    summary_file = None
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".csv") and "summary" in f.lower():
                summary_file = os.path.join(root, f)
                break

    if summary_file:
        import csv
        with open(summary_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results = dict(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    config_path = run_opencompass_eval(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
    )
    print(f"Config generated at: {config_path}")


if __name__ == "__main__":
    main()
