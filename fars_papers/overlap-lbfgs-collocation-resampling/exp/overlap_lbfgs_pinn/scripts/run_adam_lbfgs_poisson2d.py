# Adam warmstart + fixed-collocation L-BFGS baseline on 2D Poisson forward problem.
# Two-phase: (1) Adam+resampling for adam_budget evals, (2) L-BFGS+fixed for remaining budget.
# Mirrors run_adam_lbfgs_ice_shelf.py but adapted for 2D Poisson (rel L2 metric, no data loss).
# Usage: python -m overlap_lbfgs_pinn.scripts.run_adam_lbfgs_poisson2d [--budget N] [--adam_budget N] [--seeds 0,1,2]

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


def train_adam_then_lbfgs_poisson2d(model, sampler, budget_tracker, config, wandb_run=None):
    device = config["device"]
    lr = config["lr"]
    lambda_bc = config["lambda_bc"]
    adam_budget = config["adam_budget"]
    eval_interval = config.get("eval_interval", 500)

    xy_bc = poisson_2d.generate_boundary_points(
        N_per_edge=config["N_per_edge"], device=device,
    )

    loss_history = []
    eval_history = []
    best_rel_l2 = float("inf")
    best_state = None
    best_step = 0
    next_eval_at = eval_interval

    def _maybe_eval(budget_count, phase_label):
        nonlocal best_rel_l2, best_state, best_step, next_eval_at
        if budget_count < next_eval_at and not budget_tracker.exhausted():
            return
        rel_l2 = poisson_2d.eval_rel_l2(model, N_grid=100, device=device)
        entry = {"rel_l2": rel_l2, "step": budget_count, "phase": phase_label}
        eval_history.append(entry)
        print(f"  [{phase_label} budget={budget_count}] rel_L2={rel_l2:.6e}")
        if wandb_run is not None:
            wandb_run.log({"eval/rel_l2": rel_l2}, step=budget_count)
        if rel_l2 < best_rel_l2:
            best_rel_l2 = rel_l2
            best_state = copy.deepcopy(model.state_dict())
            best_step = budget_count
        while next_eval_at <= budget_count:
            next_eval_at += eval_interval

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    adam_step = 0
    nan_detected = False

    print(f"=== PHASE 1: Adam + resampling (budget 0 -> {adam_budget}) ===")
    while budget_tracker.count < adam_budget and not budget_tracker.exhausted():
        xy_coll = sampler.sample()
        adam_optimizer.zero_grad()
        loss, loss_pde, loss_bc = poisson_2d.total_loss(model, xy_coll, xy_bc, lambda_bc)

        if not math.isfinite(loss.item()):
            print(f"[ADAM STEP {adam_step}] NaN/Inf loss: total={loss.item()}")
            nan_detected = True
            if wandb_run is not None:
                wandb_run.log({"train/nan_detected": 1, "step": budget_tracker.count})
            break

        loss.backward()
        adam_optimizer.step()
        budget_tracker.increment(1)
        adam_step += 1

        loss_val = loss.item()
        loss_pde_val = loss_pde.item()
        loss_bc_val = loss_bc.item()
        loss_history.append({
            "step": budget_tracker.count, "total_loss": loss_val,
            "pde_loss": loss_pde_val, "bc_loss": loss_bc_val, "phase": "adam",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train/total_loss": loss_val,
                "train/pde_loss": loss_pde_val,
                "train/bc_loss": loss_bc_val,
                "budget/count": budget_tracker.count,
                "train/phase": 0,
            }, step=budget_tracker.count)

        if adam_step % 100 == 0:
            print(f"[ADAM {budget_tracker.count}/{budget_tracker.budget}] loss={loss_val:.6e}  "
                  f"pde={loss_pde_val:.6e}  bc={loss_bc_val:.6e}")

        _maybe_eval(budget_tracker.count, "adam")

    adam_evals = budget_tracker.count
    print(f"=== Phase 1 done: {adam_evals} gradient evals consumed ===")

    lbfgs_diagnostics = {
        "adam_evals": adam_evals,
        "lbfgs_outer_steps": 0,
        "lbfgs_closure_calls": 0,
        "closure_calls_per_step": [],
        "termination_reason": "not_started",
    }

    if nan_detected or budget_tracker.exhausted():
        lbfgs_diagnostics["termination_reason"] = "nan_in_adam" if nan_detected else "budget_exhausted_in_adam"
    else:
        xy_coll_fixed = torch.rand(config["N_coll"], 2, dtype=torch.float64, device=device)
        print(f"=== PHASE 2: L-BFGS + fixed collocation (budget {adam_evals} -> {budget_tracker.budget}) ===")

        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=20,
            tolerance_grad=1e-11,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        lbfgs_nan = False
        closure_count_this_step = 0

        def closure():
            nonlocal lbfgs_nan, closure_count_this_step
            lbfgs_optimizer.zero_grad()
            loss, loss_pde, loss_bc = poisson_2d.total_loss(
                model, xy_coll_fixed, xy_bc, lambda_bc,
            )

            if not math.isfinite(loss.item()):
                lbfgs_nan = True
                return loss

            if not budget_tracker.exhausted():
                loss.backward()
                budget_tracker.increment(1)
                closure_count_this_step += 1

            loss_val = loss.item()
            loss_pde_val = loss_pde.item()
            loss_bc_val = loss_bc.item()
            loss_history.append({
                "step": budget_tracker.count, "total_loss": loss_val,
                "pde_loss": loss_pde_val, "bc_loss": loss_bc_val, "phase": "lbfgs",
            })

            if wandb_run is not None:
                wandb_run.log({
                    "train/total_loss": loss_val,
                    "train/pde_loss": loss_pde_val,
                    "train/bc_loss": loss_bc_val,
                    "budget/count": budget_tracker.count,
                    "train/phase": 1,
                }, step=budget_tracker.count)

            return loss

        prev_loss = None
        while not budget_tracker.exhausted() and not lbfgs_nan:
            closure_count_this_step = 0
            loss_out = lbfgs_optimizer.step(closure)

            lbfgs_diagnostics["lbfgs_outer_steps"] += 1
            lbfgs_diagnostics["closure_calls_per_step"].append(closure_count_this_step)
            lbfgs_diagnostics["lbfgs_closure_calls"] += closure_count_this_step

            current_loss = loss_out.item() if loss_out is not None else float("nan")
            if lbfgs_diagnostics["lbfgs_outer_steps"] % 10 == 0 or lbfgs_diagnostics["lbfgs_outer_steps"] <= 5:
                print(f"[LBFGS step {lbfgs_diagnostics['lbfgs_outer_steps']} | "
                      f"budget={budget_tracker.count}/{budget_tracker.budget}] "
                      f"loss={current_loss:.6e}  closures={closure_count_this_step}")

            _maybe_eval(budget_tracker.count, "lbfgs")

            if lbfgs_nan:
                lbfgs_diagnostics["termination_reason"] = "nan"
                break

            if closure_count_this_step == 0:
                lbfgs_diagnostics["termination_reason"] = "budget_exhausted"
                break

            if prev_loss is not None:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                loss_change = abs(current_loss - prev_loss)
                if grad_norm < 1e-11:
                    lbfgs_diagnostics["termination_reason"] = "gradient_tolerance"
                    print(f"  L-BFGS converged: grad_norm={grad_norm:.2e} < 1e-11")
                    break
                if loss_change < 1e-14:
                    lbfgs_diagnostics["termination_reason"] = "change_tolerance"
                    print(f"  L-BFGS converged: loss_change={loss_change:.2e} < 1e-14")
                    break

            if closure_count_this_step == 1:
                lbfgs_diagnostics["termination_reason"] = "line_search_converged"
                print(f"  L-BFGS: only 1 closure call (line search found minimum immediately)")
                break

            prev_loss = current_loss

        if lbfgs_diagnostics["termination_reason"] == "not_started":
            lbfgs_diagnostics["termination_reason"] = "budget_exhausted"

    _maybe_eval(budget_tracker.count, "final")

    print(f"=== Training complete: {budget_tracker.count} total gradient evals ===")
    print(f"  Adam evals: {lbfgs_diagnostics['adam_evals']}")
    print(f"  L-BFGS closure calls: {lbfgs_diagnostics['lbfgs_closure_calls']}")
    print(f"  L-BFGS outer steps: {lbfgs_diagnostics['lbfgs_outer_steps']}")
    print(f"  Termination: {lbfgs_diagnostics['termination_reason']}")

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
        "lbfgs_diagnostics": lbfgs_diagnostics,
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
        name=f"adam_lbfgs_poisson2d_seed{seed}",
        config={**config, "seed": seed, "method": "adam_then_lbfgs", "problem": "poisson_2d"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_adam_then_lbfgs_poisson2d(
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
        "lbfgs_diagnostics": results["lbfgs_diagnostics"],
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
    parser.add_argument("--adam_budget", type=int, default=15000)
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
        "adam_budget": args.adam_budget,
        "eval_interval": 500,
        "device": args.device,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "outputs", "adam_lbfgs_poisson2d")
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
        "method": "adam_then_lbfgs",
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

    for r in all_results:
        d = r["lbfgs_diagnostics"]
        print(f"  Seed {r['seed']}: adam_evals={d['adam_evals']}, "
              f"lbfgs_steps={d['lbfgs_outer_steps']}, "
              f"lbfgs_closures={d['lbfgs_closure_calls']}, "
              f"total={r['total_evals']}, "
              f"termination={d['termination_reason']}")

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
