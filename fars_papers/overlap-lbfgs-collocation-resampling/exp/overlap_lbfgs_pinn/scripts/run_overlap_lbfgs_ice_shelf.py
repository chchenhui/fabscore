# Adam warmstart + overlap-resampled L-BFGS on ice-shelf inverse problem.
# Three-phase: (1) Adam+resampling, (2) Adam+fixed collocation, (3) OverlapLBFGS+OverlapResampleSampler.
# gamma=0.5 throughout (matching reference implementation).
# Usage: python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_ice_shelf [--budget N] [--adam_budget N] [--seeds 0,1,2]

import argparse
import json
import os

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import wandb

from overlap_lbfgs_pinn.models.mlp import PINNMLP
from overlap_lbfgs_pinn.problems import ice_shelf
from overlap_lbfgs_pinn.samplers.collocation import OverlapResampleSampler
from overlap_lbfgs_pinn.optimizers.budget_tracker import BudgetTracker
from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors
from overlap_lbfgs_pinn.trainers.pinn_trainer import train_adam_then_overlap_lbfgs


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
    overlap_sampler = OverlapResampleSampler(
        N=config["N_coll"], overlap_frac=config["overlap_frac"], device=device,
    )
    tracker = BudgetTracker(budget=config["budget"])

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "overlap-lbfgs-pinn"),
        name=f"{config.get('run_prefix', 'overlap_lbfgs_ice_shelf')}_seed{seed}",
        config={**config, "seed": seed, "method": "overlap_lbfgs"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_adam_then_overlap_lbfgs(
        model=model, problem_module=ice_shelf, overlap_sampler=overlap_sampler,
        budget_tracker=tracker, config=config, data=data, wandb_run=run,
    )

    run.finish()

    seed_out = {
        "seed": seed,
        "final_errors": results["final_errors"],
        "best_step": results["best_step"],
        "best_B_err": results["best_B_err"],
        "total_evals": results["total_evals"],
        "lbfgs_diagnostics": results["lbfgs_diagnostics"],
    }
    with open(os.path.join(output_dir, f"seed_{seed}_metrics.json"), "w") as f:
        json.dump(seed_out, f, indent=2)

    with open(os.path.join(output_dir, f"seed_{seed}_loss_history.json"), "w") as f:
        json.dump(results["loss_history"], f)

    with open(os.path.join(output_dir, f"seed_{seed}_eval_history.json"), "w") as f:
        json.dump(results["eval_history"], f)

    if "lbfgs_step_history" in results:
        with open(os.path.join(output_dir, f"seed_{seed}_lbfgs_step_history.json"), "w") as f:
            json.dump(results["lbfgs_step_history"], f)

    torch.save(model.state_dict(), os.path.join(output_dir, f"seed_{seed}_best_model.pt"))

    return seed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument("--adam_budget", type=int, default=20000)
    parser.add_argument("--adam_fixed_budget", type=int, default=7500)
    parser.add_argument("--overlap_frac", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_prefix", type=str, default="overlap_lbfgs_ice_shelf")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    config = {
        "layers": [1, 20, 20, 20, 20, 20, 20, 3],
        "lr": 1e-3,
        "gamma": args.gamma,
        "N_ob": 401,
        "N_coll": 1001,
        "noise_level": 0.3,
        "budget": args.budget,
        "adam_budget": args.adam_budget,
        "adam_fixed_budget": args.adam_fixed_budget,
        "overlap_frac": args.overlap_frac,
        "eval_interval": 500,
        "device": args.device,
        "run_prefix": args.run_prefix,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    if args.output_dir:
        output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    else:
        output_dir = os.path.join(base_dir, "outputs", "overlap_lbfgs_ice_shelf")
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
        "method": "overlap_lbfgs",
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

    for r in all_results:
        d = r["lbfgs_diagnostics"]
        print(f"  Seed {r['seed']}: adam={d['adam_evals']}, "
              f"olbfgs_steps={d['lbfgs_outer_steps']}, "
              f"ls_evals={d['lbfgs_line_search_evals']}, "
              f"overlap_evals={d['overlap_grad_evals']}, "
              f"cautious_skips={d['cautious_skips']}, "
              f"ls_failures={d['line_search_failures']}, "
              f"total={r['total_evals']}, "
              f"termination={d['termination_reason']}")

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
