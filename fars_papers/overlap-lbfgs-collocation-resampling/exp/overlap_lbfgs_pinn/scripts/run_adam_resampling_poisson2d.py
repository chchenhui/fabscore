# Adam + per-step collocation resampling baseline on 2D Poisson forward problem.
# Runs across multiple seeds, logs to WandB (offline), saves best checkpoint by rel L2 error.
# Usage: python -m overlap_lbfgs_pinn.scripts.run_adam_resampling_poisson2d [--budget N] [--seeds 0,1,2]

import argparse
import json
import os
import copy
import math

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import wandb

from overlap_lbfgs_pinn.models.mlp import PINNMLP
from overlap_lbfgs_pinn.problems import poisson_2d
from overlap_lbfgs_pinn.samplers.collocation import ResampleSampler2D
from overlap_lbfgs_pinn.optimizers.budget_tracker import BudgetTracker


def train_adam_resampling_poisson2d(model, sampler, budget_tracker, config, wandb_run=None):
    device = config["device"]
    lr = config["lr"]
    lambda_bc = config["lambda_bc"]
    eval_interval = config.get("eval_interval", 500)

    xy_bc = poisson_2d.generate_boundary_points(
        N_per_edge=config["N_per_edge"], device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_history = []
    eval_history = []
    best_rel_l2 = float("inf")
    best_state = None
    best_step = 0
    step = 0

    while not budget_tracker.exhausted():
        xy_coll = sampler.sample()
        optimizer.zero_grad()
        loss, loss_pde, loss_bc = poisson_2d.total_loss(model, xy_coll, xy_bc, lambda_bc)

        if not math.isfinite(loss.item()):
            print(f"[STEP {step}] NaN/Inf loss detected: total={loss.item()}, "
                  f"pde={loss_pde.item()}, bc={loss_bc.item()}")
            if wandb_run is not None:
                wandb_run.log({"train/nan_detected": 1, "step": step})
            break

        loss.backward()
        optimizer.step()
        budget_tracker.increment(1)
        step += 1

        loss_val = loss.item()
        loss_pde_val = loss_pde.item()
        loss_bc_val = loss_bc.item()
        loss_history.append({
            "step": step, "total_loss": loss_val,
            "pde_loss": loss_pde_val, "bc_loss": loss_bc_val,
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/total_loss": loss_val,
                "train/pde_loss": loss_pde_val,
                "train/bc_loss": loss_bc_val,
                "budget/count": budget_tracker.count,
            }, step=step)

        if step % 100 == 0:
            print(f"[STEP {step}/{budget_tracker.budget}] loss={loss_val:.6e}  "
                  f"pde={loss_pde_val:.6e}  bc={loss_bc_val:.6e}")

        if step % eval_interval == 0 or budget_tracker.exhausted():
            rel_l2 = poisson_2d.eval_rel_l2(model, N_grid=100, device=device)
            errs = {"rel_l2": rel_l2, "step": step}
            eval_history.append(errs)
            print(f"  -> rel_L2={rel_l2:.6e}")
            if wandb_run is not None:
                wandb_run.log({"eval/rel_l2": rel_l2}, step=step)
            if rel_l2 < best_rel_l2:
                best_rel_l2 = rel_l2
                best_state = copy.deepcopy(model.state_dict())
                best_step = step

    if best_state is not None:
        model.load_state_dict(best_state)
    final_rel_l2 = poisson_2d.eval_rel_l2(model, N_grid=100, device=device)

    return {
        "loss_history": loss_history,
        "eval_history": eval_history,
        "best_step": best_step,
        "best_rel_l2": best_rel_l2,
        "final_rel_l2": final_rel_l2,
        "total_evals": budget_tracker.count,
    }


def run_seed(seed, config, output_dir):
    device = config["device"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINNMLP(layers=config["layers"]).to(device)
    sampler = ResampleSampler2D(N=config["N_coll"], device=device)
    tracker = BudgetTracker(budget=config["budget"])

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "overlap-lbfgs-pinn"),
        name=f"adam_resample_poisson2d_seed{seed}",
        config={**config, "seed": seed, "method": "adam_resampling", "problem": "poisson_2d"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_adam_resampling_poisson2d(
        model=model, sampler=sampler, budget_tracker=tracker,
        config=config, wandb_run=run,
    )

    run.finish()

    seed_out = {
        "seed": seed,
        "final_rel_l2": results["final_rel_l2"],
        "best_step": results["best_step"],
        "best_rel_l2": results["best_rel_l2"],
        "total_evals": results["total_evals"],
    }
    with open(os.path.join(output_dir, f"seed_{seed}_metrics.json"), "w") as f:
        json.dump(seed_out, f, indent=2)

    with open(os.path.join(output_dir, f"seed_{seed}_loss_history.json"), "w") as f:
        json.dump(results["loss_history"], f)

    with open(os.path.join(output_dir, f"seed_{seed}_eval_history.json"), "w") as f:
        json.dump(results["eval_history"], f)

    torch.save(model.state_dict(), os.path.join(output_dir, f"seed_{seed}_best_model.pt"))

    return seed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    config = {
        "layers": [2, 50, 50, 50, 50, 1],
        "lr": 1e-3,
        "lambda_bc": 1.0,
        "N_coll": 2000,
        "N_per_edge": 200,
        "budget": args.budget,
        "eval_interval": 500,
        "device": args.device,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "outputs", "adam_resampling_poisson2d")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")
        res = run_seed(seed, config, output_dir)
        all_results.append(res)

    rel_l2s = [r["final_rel_l2"] for r in all_results]

    summary = {
        "method": "adam_resampling",
        "problem": "poisson_2d",
        "config": config,
        "seeds": seeds,
        "per_seed_results": all_results,
        "summary": {
            "rel_l2_mean": float(np.mean(rel_l2s)),
            "rel_l2_std": float(np.std(rel_l2s)),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"rel_L2: {summary['summary']['rel_l2_mean']:.6e} +/- {summary['summary']['rel_l2_std']:.6e}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
