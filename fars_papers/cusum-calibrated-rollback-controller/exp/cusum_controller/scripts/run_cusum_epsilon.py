# Runner script for the CUSUM-epsilon controller experiment.
# Supports --debug (1 seed, 1 GPU) and full mode (20 seeds x 3 conditions, multi-GPU).
# Mirrors run_or_epsilon.py; loads calibrated h, mu_0, sigma_0 from calibration_params.json.

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

from cusum_controller.controllers.cusum_controller import CUSUMController
from cusum_controller.training.train_loop import run_single
from cusum_controller.evaluation.metrics import compute_all_metrics, aggregate_across_seeds


def load_cusum_params(calibration_dir):
    path = os.path.join(calibration_dir, "calibration_params.json")
    with open(path) as f:
        params = json.load(f)
    reset_fraction = params.get("cusum_reset_fraction", 0.5)
    return params["cusum_h"], params["mu_0"], params["sigma_0"], reset_fraction


def run_one_experiment(gpu_id, seed, perturb_type, h, mu_0, sigma_0, alpha, k,
                       reset_fraction, data_root, wandb_project, results_base, result_queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")

        controller = CUSUMController(alpha=alpha, k=k, h=h, mu_0=mu_0, sigma_0=sigma_0,
                                     reset_fraction=reset_fraction)
        result = run_single(
            seed=seed,
            perturb_type=perturb_type,
            controller=controller,
            controller_name="cusum_epsilon",
            num_steps=250,
            data_root=data_root,
            device=device,
            use_wandb=True,
            wandb_project=wandb_project,
        )

        trace_path = os.path.join(results_base, "traces", f"{perturb_type}_seed{seed}.npz")
        save_dict = {
            "probe_losses": result["probe_losses"],
            "train_losses": result["train_losses"],
            "innovations": result["innovations"],
            "ema_refs": result["ema_refs"],
            "decisions": result["decisions"],
            "perturbation_active": result["perturbation_active"],
        }
        if "standardized_innovations" in result:
            save_dict["standardized_innovations"] = result["standardized_innovations"]
            save_dict["cusum_stats"] = result["cusum_stats"]
        np.savez(trace_path, **save_dict)

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
    h, mu_0, sigma_0, reset_fraction = load_cusum_params(calibration_dir)
    print(f"Loaded calibrated h={h}, mu_0={mu_0:.6f}, sigma_0={sigma_0:.6f}, reset_fraction={reset_fraction}")

    results_base = os.path.join(BASE_DIR, "cusum_controller", "results", "cusum_epsilon")
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(os.path.join(results_base, "traces"), exist_ok=True)

    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_gpus = args.num_gpus
    print(f"Using {num_gpus} GPUs")

    alpha = 0.1
    k = 0.5

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
            tasks.append((gpu_id, seed, perturb_type, h, mu_0, sigma_0, alpha, k,
                          reset_fraction, args.data_root, wandb_project, results_base))

    print(f"Total experiments: {len(tasks)}")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    if num_gpus > 1 and len(tasks) > 1:
        batch_size = num_gpus
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch = [(j % num_gpus, t[1], t[2], t[3], t[4], t[5], t[6], t[7],
                       t[8], t[9], t[10], t[11]) for j, t in enumerate(batch)]
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
        print(f"Aggregated metrics for {perturb_type} (CUSUM-epsilon):")
        for mk, v in agg.items():
            if v["mean"] is not None:
                print(f"  {mk}: {v['mean']:.4f} +/- {v['std']:.4f} (n={v['count']})")
            else:
                print(f"  {mk}: None (n={v['count']})")

    summary_path = os.path.join(results_base, "metrics_summary.json")
    serializable = {}
    for pt, agg in summary.items():
        serializable[pt] = {}
        for mk, v in agg.items():
            serializable[pt][mk] = {"mean": v["mean"], "std": v["std"], "count": v["count"]}
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
