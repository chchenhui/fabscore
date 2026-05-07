"""Optimized Qwen2.5-7B best-of-k sampling with tunable temperature and diversity penalties.
Key improvements over qwen_best_of_k.py:
  - stop=["\n\n"] for BOTH tasks (sudoku was missing it, wasting tokens)
  - Configurable temperature (default 1.0, higher for more diversity)
  - Configurable presence_penalty for sample diversity
  - WandB offline logging support
"""

import json
import os
import csv
import argparse
import time
import numpy as np
import reasoning_gym
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

TASK_RG_NAME = {
    "countdown": "countdown",
    "sudoku": "mini_sudoku",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_template(path):
    with open(path) as f:
        return f.read()


def load_calibration(task):
    cal_path = os.path.join(BASE_DIR, "timing", "calibration_results.json")
    with open(cal_path) as f:
        cal = json.load(f)
    key = task if task in cal else ("sudoku" if task == "sudoku" else task)
    return cal[key]


def build_prompt(template, question):
    return template + "\n" + question.strip() + "\nOutput:"


def parse_countdown_answer(raw_output):
    text = raw_output.strip()
    return text.split("\n")[0].strip()


def parse_sudoku_answer(raw_output):
    text = raw_output.strip()
    return text.split("\n\n")[0].strip()


def run_best_of_k(llm, task, instances, template, k, seeds, ds, limit=None,
                  temperature=1.0, top_p=0.95, presence_penalty=0.0,
                  frequency_penalty=0.0, output_prefix="qwen_bok_opt"):
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))

    try:
        import wandb
        wandb_available = True
    except ImportError:
        wandb_available = False
        print("WARNING: wandb not available, skipping logging")

    if limit is not None:
        instances = instances[:limit]

    parse_fn = parse_sudoku_answer if task == "sudoku" else parse_countdown_answer

    raw_dir = os.path.join(BASE_DIR, "results", "raw")
    tables_dir = os.path.join(BASE_DIR, "results", "tables")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    prompts = [build_prompt(template, inst["question"]) for inst in instances]

    seed_accuracies = []

    for seed in seeds:
        run_name = f"{output_prefix}_{task}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"Task={task}, seed={seed}, k={k}, instances={len(instances)}")
        print(f"Config: temp={temperature}, top_p={top_p}, "
              f"pres_pen={presence_penalty}, freq_pen={frequency_penalty}")
        print(f"{'='*60}")

        if wandb_available:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "compute-matched-diffusion-planning-audit"),
                name=run_name,
                mode=os.environ.get("WANDB_MODE", "offline"),
                config={
                    "task": task, "seed": seed, "k": k,
                    "temperature": temperature, "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "max_tokens": 64, "num_instances": len(instances),
                    "method": "best_of_k_optimized",
                    "model": "Qwen/Qwen2.5-7B",
                    "stop": "\\n\\n",
                },
                reinit=True,
            )

        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=64,
            seed=seed,
            stop=["\n\n"],
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - t0
        print(f"Generation took {gen_time:.1f}s ({gen_time/len(instances):.2f}s/instance)")

        out_file = os.path.join(raw_dir, f"{output_prefix}_{task}_seed{seed}.jsonl")
        num_solved = 0
        results = []

        for idx, (inst, output) in enumerate(zip(instances, outputs)):
            all_texts = [o.text for o in output.outputs]
            all_parsed = [parse_fn(t) for t in all_texts]

            entry = ds[inst["id"]]
            all_scores = []
            for parsed in all_parsed:
                score = ds.score_answer(answer=parsed, entry=entry)
                all_scores.append(1.0 if score == 1.0 else 0.0)

            solved = any(s == 1.0 for s in all_scores)
            if solved:
                num_solved += 1

            n_unique = len(set(all_parsed))
            n_correct = sum(all_scores)

            record = {
                "id": inst["id"],
                "seed": seed,
                "k": k,
                "all_outputs": all_texts,
                "all_scores": all_scores,
                "solved": solved,
                "n_unique": n_unique,
                "n_correct": int(n_correct),
            }
            results.append(record)

            if wandb_available and (idx + 1) % 10 == 0:
                wandb.log({
                    "instance_id": inst["id"],
                    "solved": int(solved),
                    "running_accuracy": num_solved / (idx + 1),
                    "num_processed": idx + 1,
                    "num_solved": num_solved,
                    "n_unique_samples": n_unique,
                    "n_correct_samples": int(n_correct),
                })

        accuracy = num_solved / len(instances)
        seed_accuracies.append(accuracy)

        mean_unique = np.mean([r["n_unique"] for r in results])
        mean_correct = np.mean([r["n_correct"] for r in results])

        with open(out_file, "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        print(f"Seed {seed}: accuracy={accuracy:.4f} ({num_solved}/{len(instances)})")
        print(f"  Mean unique samples: {mean_unique:.1f}/{k}")
        print(f"  Mean correct samples: {mean_correct:.2f}/{k}")
        print(f"Saved to {out_file}")

        if wandb_available:
            wandb.log({
                "seed_accuracy": accuracy,
                "num_correct": num_solved,
                "num_total": len(instances),
                "generation_time_s": gen_time,
                "mean_unique_samples": mean_unique,
                "mean_correct_samples": mean_correct,
            })
            wandb.finish()

    mean_acc = np.mean(seed_accuracies)
    std_acc = np.std(seed_accuracies)
    print(f"\nOverall: mean={mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Per-seed: {[f'{a:.4f}' for a in seed_accuracies]}")

    csv_file = os.path.join(tables_dir, f"{output_prefix}_{task}_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "method", "seed", "k", "accuracy", "num_correct", "num_total",
                         "temperature", "presence_penalty"])
        for seed, acc in zip(seeds, seed_accuracies):
            nc = int(round(acc * len(instances)))
            writer.writerow([TASK_RG_NAME[task], output_prefix, seed, k, f"{acc:.4f}", nc,
                             len(instances), temperature, presence_penalty])
        writer.writerow([TASK_RG_NAME[task], output_prefix, "mean", k, f"{mean_acc:.4f}", "",
                         len(instances), temperature, presence_penalty])
        writer.writerow([TASK_RG_NAME[task], output_prefix, "std", k, f"{std_acc:.4f}", "",
                         len(instances), temperature, presence_penalty])
    print(f"CSV saved to {csv_file}")

    if wandb_available:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "compute-matched-diffusion-planning-audit"),
            name=f"{output_prefix}_{task}_summary",
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={"task": task, "k": k, "seeds": seeds, "method": "best_of_k_optimized",
                    "temperature": temperature, "presence_penalty": presence_penalty},
            reinit=True,
        )
        wandb.log({
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "k": k,
        })
        for seed, acc in zip(seeds, seed_accuracies):
            wandb.log({f"accuracy_seed{seed}": acc})
        wandb.finish()

    return mean_acc, std_acc, seed_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["countdown", "sudoku"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k_override", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--output_prefix", type=str, default="qwen_bok_opt")
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["test", "cal"],
                        help="Use test (500) or cal (50) data split")
    args = parser.parse_args()

    cal = load_calibration(args.task)
    k = args.k_override if args.k_override else cal["k_median"]
    print(f"Task: {args.task}, k={k} (from calibration k_median={cal['k_median']})")

    rg_name = TASK_RG_NAME[args.task]
    if args.data_split == "test":
        ds = reasoning_gym.create_dataset(rg_name, seed=2024, size=500)
        data_file = os.path.join(BASE_DIR, "data", f"{args.task}_test.jsonl")
    else:
        ds = reasoning_gym.create_dataset(rg_name, seed=9999, size=50)
        data_file = os.path.join(BASE_DIR, "data", f"{args.task}_cal.jsonl")

    template_file = os.path.join(BASE_DIR, "prompts", f"{args.task}_8shot.txt")

    instances = load_jsonl(data_file)
    template = load_template(template_file)

    llm = LLM(
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
    )

    mean_acc, std_acc, seed_accs = run_best_of_k(
        llm, args.task, instances, template, k, args.seeds, ds,
        limit=args.limit,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        output_prefix=args.output_prefix,
    )

    print(f"\nFinal: {args.task} best-of-{k} accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")


if __name__ == "__main__":
    main()
