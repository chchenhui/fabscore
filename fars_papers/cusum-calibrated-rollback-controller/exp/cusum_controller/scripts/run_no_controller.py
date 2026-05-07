# Runner script for the no-controller baseline experiment.
# Runs 20 seeds x 3 perturbation types (step, ramp, nominal) = 60 experiments.
# Use --debug for a single-seed quick sanity check.
# Results saved to cusum_controller/results/no_controller/.

import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

import wandb
from cusum_controller.controllers.no_controller import NoController
from cusum_controller.training.train_loop import run_single
from cusum_controller.evaluation.metrics import compute_all_metrics, aggregate_across_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run single seed=0 with step perturbation only")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    wandb_project = os.environ.get("WANDB_PROJECT", "cusum-calibrated-rollback-controller")

    results_base = os.path.join(BASE_DIR, "cusum_controller", "results", "no_controller")
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(os.path.join(results_base, "nominal_probe_traces"), exist_ok=True)
    os.makedirs(os.path.join(results_base, "traces"), exist_ok=True)

    if args.debug:
        seeds = [0]
        perturb_types = ["step"]
    else:
        seeds = list(range(args.num_seeds))
        perturb_types = ["step", "ramp", "nominal"]

    all_results = {}
    all_metrics = {}

    for perturb_type in perturb_types:
        all_results[perturb_type] = []
        all_metrics[perturb_type] = []

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running: no_controller | seed={seed} | perturb={perturb_type}")
            print(f"{'='*60}")

            controller = NoController(alpha=0.1)
            result = run_single(
                seed=seed,
                perturb_type=perturb_type,
                controller=controller,
                controller_name="no_controller",
                num_steps=250,
                data_root=args.data_root,
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

            if perturb_type == "nominal":
                np.save(
                    os.path.join(results_base, "nominal_probe_traces", f"seed{seed}.npy"),
                    result["probe_losses"],
                )

            metrics = compute_all_metrics(result)
            all_results[perturb_type].append(result)
            all_metrics[perturb_type].append(metrics)

            print(f"  Metrics: {metrics}")

    summary = {}
    for perturb_type in perturb_types:
        agg = aggregate_across_seeds(all_metrics[perturb_type])
        summary[perturb_type] = agg
        print(f"\n{'='*60}")
        print(f"Aggregated metrics for {perturb_type}:")
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
            serializable[pt][k] = {
                "mean": v["mean"],
                "std": v["std"],
                "count": v["count"],
            }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)

    per_seed_path = os.path.join(results_base, "per_seed_metrics.json")
    per_seed_data = {}
    for pt in perturb_types:
        per_seed_data[pt] = []
        for m in all_metrics[pt]:
            per_seed_data[pt].append({k: v for k, v in m.items()})
    with open(per_seed_path, "w") as f:
        json.dump(per_seed_data, f, indent=2)

    print(f"\nResults saved to {results_base}")
    print("Done!")


if __name__ == "__main__":
    main()
