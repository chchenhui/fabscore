# Adam warmstart + overlap-resampled L-BFGS on 2D Poisson forward problem.
# Three-phase: (1) Adam+resampling, (2) Adam+fixed collocation, (3) OverlapLBFGS.
# Phase 2 stabilizes the loss landscape before L-BFGS. Higher lambda_bc in
# phases 2-3 ensures boundary conditions are enforced by L-BFGS.
# Usage: python -m overlap_lbfgs_pinn.scripts.run_overlap_lbfgs_poisson2d [--budget N] [--seeds 0,1,2]

import argparse
import copy
import json
import math
import os

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import wandb

from overlap_lbfgs_pinn.models.mlp import PINNMLP
from overlap_lbfgs_pinn.problems import poisson_2d
from overlap_lbfgs_pinn.samplers.collocation import (
    ResampleSampler2D, OverlapResampleSampler2D, FixedSampler2D,
)
from overlap_lbfgs_pinn.optimizers.overlap_lbfgs import OverlapLBFGS
from overlap_lbfgs_pinn.optimizers.budget_tracker import BudgetTracker


def train_overlap_lbfgs_poisson2d(model, adam_sampler, overlap_sampler,
                                   budget_tracker, config, wandb_run=None):
    device = config["device"]
    lr = config["lr"]
    lambda_bc = config["lambda_bc"]
    lambda_bc_fixed = config.get("lambda_bc_fixed", lambda_bc)
    lambda_bc_lbfgs = config.get("lambda_bc_lbfgs", lambda_bc)
    adam_budget = config["adam_budget"]
    adam_fixed_budget = config.get("adam_fixed_budget", 0)
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

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adam_step = 0
    nan_detected = False

    print(f"=== PHASE 1: Adam + resampling (budget 0 -> {adam_budget}) ===")
    while budget_tracker.count < adam_budget and not budget_tracker.exhausted():
        xy_coll = adam_sampler.sample()
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

    adam_fixed_evals = 0
    adam_fixed_end = adam_evals + adam_fixed_budget

    if not nan_detected and not budget_tracker.exhausted() and adam_fixed_budget > 0:
        xy_coll_fixed = torch.rand(config["N_coll"], 2, dtype=torch.float64, device=device)
        print(f"=== PHASE 2: Adam + fixed collocation (budget {adam_evals} -> {adam_fixed_end}, lambda_bc={lambda_bc_fixed}) ===")

        adam_fixed_step = 0
        while budget_tracker.count < adam_fixed_end and not budget_tracker.exhausted():
            adam_optimizer.zero_grad()
            loss, loss_pde, loss_bc = poisson_2d.total_loss(
                model, xy_coll_fixed, xy_bc, lambda_bc_fixed,
            )

            if not math.isfinite(loss.item()):
                print(f"[ADAM-FIXED STEP {adam_fixed_step}] NaN/Inf loss")
                nan_detected = True
                break

            loss.backward()
            adam_optimizer.step()
            budget_tracker.increment(1)
            adam_fixed_step += 1

            loss_val = loss.item()
            loss_pde_val = loss_pde.item()
            loss_bc_val = loss_bc.item()
            loss_history.append({
                "step": budget_tracker.count, "total_loss": loss_val,
                "pde_loss": loss_pde_val, "bc_loss": loss_bc_val, "phase": "adam_fixed",
            })

            if wandb_run is not None:
                wandb_run.log({
                    "train/total_loss": loss_val,
                    "train/pde_loss": loss_pde_val,
                    "train/bc_loss": loss_bc_val,
                    "budget/count": budget_tracker.count,
                    "train/phase": 1,
                }, step=budget_tracker.count)

            if adam_fixed_step % 100 == 0:
                print(f"[ADAM-FIXED {budget_tracker.count}/{budget_tracker.budget}] loss={loss_val:.6e}  "
                      f"pde={loss_pde_val:.6e}  bc={loss_bc_val:.6e}")

            _maybe_eval(budget_tracker.count, "adam_fixed")

        adam_fixed_evals = budget_tracker.count - adam_evals
        print(f"=== Phase 2 done: {adam_fixed_evals} additional gradient evals ===")

    lbfgs_diagnostics = {
        "adam_evals": adam_evals,
        "adam_fixed_evals": adam_fixed_evals,
        "lbfgs_outer_steps": 0,
        "lbfgs_line_search_evals": 0,
        "overlap_grad_evals": 0,
        "cautious_skips": 0,
        "line_search_failures": 0,
        "termination_reason": "not_started",
    }

    if nan_detected or budget_tracker.exhausted():
        lbfgs_diagnostics["termination_reason"] = "nan_in_adam" if nan_detected else "budget_exhausted_in_adam"
    else:
        print(f"=== PHASE 3: Overlap-LBFGS (budget {budget_tracker.count} -> {budget_tracker.budget}, lambda_bc={lambda_bc_lbfgs}) ===")

        olbfgs = OverlapLBFGS(
            model.parameters(),
            history_size=20,
            c1=1e-4,
            c2=0.9,
            max_ls=20,
            cautious_eps=1e-6,
        )

        lbfgs_nan = False

        while not budget_tracker.exhausted() and not lbfgs_nan:
            full_set, overlap_set = overlap_sampler.sample()

            def make_full_closure(xy_full):
                def _closure():
                    olbfgs.zero_grad()
                    loss, loss_pde, loss_bc = poisson_2d.total_loss(
                        model, xy_full, xy_bc, lambda_bc_lbfgs,
                    )
                    if not math.isfinite(loss.item()):
                        nonlocal lbfgs_nan
                        lbfgs_nan = True
                        return (loss.item(), loss_pde.item(), loss_bc.item())
                    loss.backward()
                    return (loss.item(), loss_pde.item(), loss_bc.item())
                return _closure

            def make_overlap_closure(xy_overlap):
                def _closure():
                    olbfgs.zero_grad()
                    loss, loss_pde, loss_bc = poisson_2d.total_loss(
                        model, xy_overlap, xy_bc, lambda_bc_lbfgs,
                    )
                    if not math.isfinite(loss.item()):
                        nonlocal lbfgs_nan
                        lbfgs_nan = True
                        return (loss.item(), loss_pde.item(), loss_bc.item())
                    loss.backward()
                    return (loss.item(), loss_pde.item(), loss_bc.item())
                return _closure

            full_closure = make_full_closure(full_set)
            overlap_closure = make_overlap_closure(overlap_set)

            step_info = olbfgs.step(full_closure, overlap_closure, budget_tracker)

            lbfgs_diagnostics["lbfgs_outer_steps"] += 1
            lbfgs_diagnostics["lbfgs_line_search_evals"] += step_info["ls_evals"]
            lbfgs_diagnostics["overlap_grad_evals"] += step_info["overlap_evals"]
            lbfgs_diagnostics["cautious_skips"] = olbfgs.diagnostics["cautious_skips"]
            lbfgs_diagnostics["line_search_failures"] = olbfgs.diagnostics["line_search_failures"]

            loss_val = step_info["loss"]
            loss_pde_val = step_info["loss_e"]
            loss_bc_val = step_info["loss_d"]
            loss_history.append({
                "step": budget_tracker.count, "total_loss": loss_val,
                "pde_loss": loss_pde_val, "bc_loss": loss_bc_val,
                "phase": "overlap_lbfgs",
            })

            if wandb_run is not None:
                wandb_run.log({
                    "train/total_loss": loss_val,
                    "train/pde_loss": loss_pde_val,
                    "train/bc_loss": loss_bc_val,
                    "budget/count": budget_tracker.count,
                    "train/phase": 2,
                    "lbfgs/cautious_skips": olbfgs.diagnostics["cautious_skips"],
                    "lbfgs/line_search_evals": step_info["ls_evals"],
                    "lbfgs/overlap_grad_evals": olbfgs.diagnostics["overlap_grad_evals"],
                    "lbfgs/alpha": step_info["alpha"],
                    "lbfgs/grad_norm": step_info["grad_norm"],
                }, step=budget_tracker.count)

            outer = lbfgs_diagnostics["lbfgs_outer_steps"]
            if outer % 10 == 0 or outer <= 5:
                print(f"[OL-BFGS step {outer} | budget={budget_tracker.count}/"
                      f"{budget_tracker.budget}] loss={loss_val:.6e}  "
                      f"alpha={step_info['alpha']:.4e}  ls_evals={step_info['ls_evals']}  "
                      f"cautious_skip={step_info['cautious_skip']}")

            _maybe_eval(budget_tracker.count, "overlap_lbfgs")

            if lbfgs_nan or step_info.get("nan_detected", False):
                lbfgs_diagnostics["termination_reason"] = "nan"
                break

            if step_info["grad_norm"] < 1e-11:
                lbfgs_diagnostics["termination_reason"] = "gradient_tolerance"
                print(f"  Overlap-LBFGS converged: grad_norm={step_info['grad_norm']:.2e}")
                break

    if lbfgs_diagnostics["termination_reason"] == "not_started":
        lbfgs_diagnostics["termination_reason"] = "budget_exhausted"

    _maybe_eval(budget_tracker.count, "final")

    print(f"=== Training complete: {budget_tracker.count} total gradient evals ===")
    print(f"  Adam evals: {lbfgs_diagnostics['adam_evals']}")
    print(f"  Adam-fixed evals: {lbfgs_diagnostics['adam_fixed_evals']}")
    print(f"  Overlap-LBFGS outer steps: {lbfgs_diagnostics['lbfgs_outer_steps']}")
    print(f"  Overlap-LBFGS line-search evals: {lbfgs_diagnostics['lbfgs_line_search_evals']}")
    print(f"  Overlap gradient evals: {lbfgs_diagnostics['overlap_grad_evals']}")
    print(f"  Cautious skips: {lbfgs_diagnostics['cautious_skips']}")
    print(f"  Line search failures: {lbfgs_diagnostics['line_search_failures']}")
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
    adam_sampler = ResampleSampler2D(N=config["N_coll"], device=device)
    overlap_sampler = OverlapResampleSampler2D(
        N=config["N_coll"], overlap_frac=config["overlap_frac"], device=device,
    )
    tracker = BudgetTracker(budget=config["budget"])

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "overlap-lbfgs-pinn"),
        name=f"overlap_lbfgs_poisson2d_seed{seed}",
        config={**config, "seed": seed, "method": "overlap_lbfgs", "problem": "poisson_2d"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_overlap_lbfgs_poisson2d(
        model=model, adam_sampler=adam_sampler, overlap_sampler=overlap_sampler,
        budget_tracker=tracker, config=config, wandb_run=run,
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
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument("--adam_budget", type=int, default=15000)
    parser.add_argument("--adam_fixed_budget", type=int, default=5000)
    parser.add_argument("--overlap_frac", type=float, default=0.9)
    parser.add_argument("--lambda_bc", type=float, default=1.0)
    parser.add_argument("--lambda_bc_fixed", type=float, default=5.0)
    parser.add_argument("--lambda_bc_lbfgs", type=float, default=10.0)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    config = {
        "layers": [2, 50, 50, 50, 50, 1],
        "lr": 1e-3,
        "lambda_bc": args.lambda_bc,
        "lambda_bc_fixed": args.lambda_bc_fixed,
        "lambda_bc_lbfgs": args.lambda_bc_lbfgs,
        "N_coll": 2000,
        "N_per_edge": 200,
        "budget": args.budget,
        "adam_budget": args.adam_budget,
        "adam_fixed_budget": args.adam_fixed_budget,
        "overlap_frac": args.overlap_frac,
        "eval_interval": 500,
        "device": args.device,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "outputs", "overlap_lbfgs_poisson2d")
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
        "method": "overlap_lbfgs",
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
              f"adam_fixed_evals={d['adam_fixed_evals']}, "
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
