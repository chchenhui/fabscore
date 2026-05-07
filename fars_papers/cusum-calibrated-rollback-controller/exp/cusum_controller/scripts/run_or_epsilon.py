# Runner script for the Or-epsilon controller experiment.
# Supports --debug (1 seed, 1 GPU) and full mode (20 seeds x 3 conditions, multi-GPU).
# Uses torch.multiprocessing.Process (non-daemonic) to allow DataLoader workers.

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from cusum_controller.controllers.or_controller import OrController
from cusum_controller.training.train_loop import run_single
from cusum_controller.evaluation.metrics import compute_all_metrics, aggregate_across_seeds


def load_epsilon(calibration_dir):
    path = os.path.join(calibration_dir, "calibration_params.json")
    with open(path) as f:
        params = json.load(f)
    return params["epsilon"]


def run_one_experiment(gpu_id, seed, perturb_type, epsilon, alpha, data_root, wandb_project, results_base, result_queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")

        controller = OrController(alpha=alpha, epsilon=epsilon)
        result = run_single(
            seed=seed,
            perturb_type=perturb_type,
            controller=controller,
            controller_name="or_epsilon",
            num_steps=250,
            data_root=data_root,
            device=device,
            use_wandb=True,
            wandb_project=wandb_project,
        )

        trace_path = os.path.join(results_base, "traces", f"{perturb_type}_seed{seed}.npz")
        np.savez(
            trace_path,
            probe_losses=result["probe_losses"],
            train_losses=result["train_losses"],
            innovations=result["innovations"],
            ema_refs=result["ema_refs"],
            decisions=result["decisions"],
            perturbation_active=result["perturbation_active"],
        )

        metrics = compute_all_metrics(result)
        result_queue.put({"seed": seed, "perturb_type": perturb_type, "metrics": metrics, "error": None})
    except Exception as e:
        traceback.print_exc()
        result_queue.put({"seed": seed, "perturb_type": perturb_type, "metrics": None, "error": str(e)})


def run_batch(batch_tasks, result_queue):
    processes = []
    for task_args in batch_tasks:
        p = mp.Process(target=run_one_experiment, args=(*task_args, result_queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_gpus", type=int, default=None)
    args = parser.parse_args()

    wandb_project = os.environ.get("WANDB_PROJECT", "cusum-calibrated-rollback-controller")

    calibration_dir = os.path.join(BASE_DIR, "cusum_controller", "results", "calibration")
    epsilon = load_epsilon(calibration_dir)
    print(f"Loaded calibrated epsilon = {epsilon:.6f}")

    results_base = os.path.join(BASE_DIR, "cusum_controller", "results", "or_epsilon")
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(os.path.join(results_base, "traces"), exist_ok=True)

    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs")

    if args.debug:
        seeds = [0]
        perturb_types = ["step"]
    else:
        seeds = list(range(args.num_seeds))
        perturb_types = ["step", "ramp", "nominal"]

    tasks = []
    for perturb_type in perturb_types:
        for seed in seeds:
            gpu_id = len(tasks) % num_gpus
            tasks.append((gpu_id, seed, perturb_type, epsilon, 0.1, args.data_root, wandb_project, results_base))

    print(f"Total experiments: {len(tasks)}")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    if num_gpus > 1 and len(tasks) > 1:
        batch_size = num_gpus
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch = [(j % num_gpus, t[1], t[2], t[3], t[4], t[5], t[6], t[7]) for j, t in enumerate(batch)]
            print(f"\nBatch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size}: "
                  f"{len(batch)} experiments")
            run_batch(batch, result_queue)
    else:
        for t in tasks:
            run_one_experiment(*t, result_queue)

    all_results_list = []
    while not result_queue.empty():
        all_results_list.append(result_queue.get())

    all_metrics = {}
    for perturb_type in perturb_types:
        all_metrics[perturb_type] = []
    for r in all_results_list:
        if r["error"] is None:
            all_metrics[r["perturb_type"]].append(r["metrics"])
        else:
            print(f"ERROR: seed={r['seed']} perturb={r['perturb_type']}: {r['error']}")

    summary = {}
    for perturb_type in perturb_types:
        agg = aggregate_across_seeds(all_metrics[perturb_type])
        summary[perturb_type] = agg
        print(f"\n{'='*60}")
        print(f"Aggregated metrics for {perturb_type} (Or-epsilon):")
        for k, v in agg.items():
            if v["mean"] is not None:
                print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f} (n={v['count']})")
            else:
                print(f"  {k}: None (n={v['count']})")

    summary_path = os.path.join(results_base, "summary_metrics.json")
    serializable = {}
    for pt, agg in summary.items():
        serializable[pt] = {}
        for k, v in agg.items():
            serializable[pt][k] = {"mean": v["mean"], "std": v["std"], "count": v["count"]}
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)

    per_seed_path = os.path.join(results_base, "per_seed_metrics.json")
    per_seed_data = {}
    for pt in perturb_types:
        per_seed_data[pt] = all_metrics[pt]
    with open(per_seed_path, "w") as f:
        json.dump(per_seed_data, f, indent=2)

    print(f"\nResults saved to {results_base}")
    print("Done!")


if __name__ == "__main__":
    main()
