# Adam + per-step collocation resampling baseline on ice-shelf inverse problem.
# Runs across multiple seeds, logs to WandB (offline), saves best checkpoint by B_err.
# Usage: python -m overlap_lbfgs_pinn.scripts.run_adam_resampling_ice_shelf [--budget N] [--seeds 0,1,2]

import argparse
import json
import os
import sys
import copy

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import wandb

from overlap_lbfgs_pinn.models.mlp import PINNMLP
from overlap_lbfgs_pinn.problems import ice_shelf
from overlap_lbfgs_pinn.samplers.collocation import ResampleSampler
from overlap_lbfgs_pinn.optimizers.budget_tracker import BudgetTracker
from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors
from overlap_lbfgs_pinn.trainers.pinn_trainer import train_adam_resampling


def run_seed(seed, config, output_dir):
    device = config["device"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINNMLP(layers=config["layers"]).to(device)
    data = ice_shelf.generate_data(
        N_ob=config["N_ob"], noise_level=config["noise_level"],
        seed=seed, device=device,
    )
    sampler = ResampleSampler(N=config["N_coll"], device=device)
    tracker = BudgetTracker(budget=config["budget"])

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "overlap-lbfgs-pinn"),
        name=f"adam_resample_ice_shelf_seed{seed}",
        config={**config, "seed": seed, "method": "adam_resampling"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_adam_resampling(
        model=model, problem_module=ice_shelf, sampler=sampler,
        budget_tracker=tracker, config=config, data=data, wandb_run=run,
    )

    run.finish()

    seed_out = {
        "seed": seed,
        "final_errors": results["final_errors"],
        "best_step": results["best_step"],
        "best_B_err": results["best_B_err"],
        "total_evals": results["total_evals"],
    }
    with open(os.path.join(output_dir, f"seed_{seed}_metrics.json"), "w") as f:
        json.dump(seed_out, f, indent=2)

    loss_hist = results["loss_history"]
    with open(os.path.join(output_dir, f"seed_{seed}_loss_history.json"), "w") as f:
        json.dump(loss_hist, f)

    eval_hist = results["eval_history"]
    with open(os.path.join(output_dir, f"seed_{seed}_eval_history.json"), "w") as f:
        json.dump(eval_hist, f)

    torch.save(model.state_dict(), os.path.join(output_dir, f"seed_{seed}_best_model.pt"))

    return seed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    gamma_ratio = 0.1
    gamma = gamma_ratio / (1.0 + gamma_ratio)

    config = {
        "layers": [1, 20, 20, 20, 20, 20, 20, 3],
        "lr": 1e-3,
        "gamma": gamma,
        "N_ob": 401,
        "N_coll": 1001,
        "noise_level": 0.3,
        "budget": args.budget,
        "eval_interval": 500,
        "device": args.device,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "outputs", "adam_resampling_ice_shelf")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        res = run_seed(seed, config, output_dir)
        all_results.append(res)

    B_errs = [r["final_errors"]["B_err"] for r in all_results]
    u_errs = [r["final_errors"]["u_err"] for r in all_results]
    h_errs = [r["final_errors"]["h_err"] for r in all_results]

    summary = {
        "method": "adam_resampling",
        "problem": "ice_shelf",
        "config": config,
        "seeds": seeds,
        "per_seed_results": all_results,
        "summary": {
            "B_err_mean": float(np.mean(B_errs)),
            "B_err_std": float(np.std(B_errs)),
            "u_err_mean": float(np.mean(u_errs)),
            "u_err_std": float(np.std(u_errs)),
            "h_err_mean": float(np.mean(h_errs)),
            "h_err_std": float(np.std(h_errs)),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"B_err: {summary['summary']['B_err_mean']:.6e} +/- {summary['summary']['B_err_std']:.6e}")
    print(f"u_err: {summary['summary']['u_err_mean']:.6e} +/- {summary['summary']['u_err_std']:.6e}")
    print(f"h_err: {summary['summary']['h_err_mean']:.6e} +/- {summary['summary']['h_err_std']:.6e}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
