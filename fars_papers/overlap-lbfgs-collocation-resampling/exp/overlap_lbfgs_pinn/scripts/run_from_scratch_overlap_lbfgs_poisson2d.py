"""Diagnostic: Overlap-resampled L-BFGS from random init (no Adam warmstart) on 2D Poisson.
Skips Phase 1 (Adam+resampling) and Phase 2 (Adam+fixed) entirely.
Runs only overlap-LBFGS from step 0 with OverlapResampleSampler2D(o=0.5).
Same total budget (50000) as the warmstart experiments for fair comparison.
"""
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
from overlap_lbfgs_pinn.samplers.collocation import OverlapResampleSampler2D
from overlap_lbfgs_pinn.optimizers.overlap_lbfgs import OverlapLBFGS
from overlap_lbfgs_pinn.optimizers.budget_tracker import BudgetTracker


def train_from_scratch_overlap_lbfgs_poisson2d(model, overlap_sampler,
                                                budget_tracker, config,
                                                wandb_run=None):
    device = config["device"]
    lambda_bc = config["lambda_bc"]
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

    _maybe_eval(0, "init")

    lbfgs_diagnostics = {
        "adam_evals": 0,
        "adam_fixed_evals": 0,
        "lbfgs_outer_steps": 0,
        "lbfgs_line_search_evals": 0,
        "overlap_grad_evals": 0,
        "cautious_skips": 0,
        "line_search_failures": 0,
        "termination_reason": "not_started",
    }

    print(f"=== PHASE: Overlap-LBFGS from scratch (budget 0 -> {budget_tracker.budget}, lambda_bc={lambda_bc}) ===")

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
                    model, xy_full, xy_bc, lambda_bc,
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
                    model, xy_overlap, xy_bc, lambda_bc,
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
            "phase": "overlap_lbfgs_scratch",
        })

        if wandb_run is not None:
            wb_log = {
                "train/total_loss": loss_val,
                "train/pde_loss": loss_pde_val,
                "train/bc_loss": loss_bc_val,
                "budget/count": budget_tracker.count,
                "train/phase": 0,
                "lbfgs/cautious_skips": olbfgs.diagnostics["cautious_skips"],
                "lbfgs/line_search_evals": step_info["ls_evals"],
                "lbfgs/overlap_grad_evals": olbfgs.diagnostics["overlap_grad_evals"],
                "lbfgs/alpha": step_info["alpha"],
                "lbfgs/grad_norm": step_info["grad_norm"],
            }
            if step_info.get("ys_value") is not None:
                wb_log["lbfgs/ys_value"] = step_info["ys_value"]
                wb_log["lbfgs/y_norm"] = step_info["y_norm"]
                wb_log["lbfgs/s_norm"] = step_info["s_norm"]
            wandb_run.log(wb_log, step=budget_tracker.count)

        outer = lbfgs_diagnostics["lbfgs_outer_steps"]
        if outer % 10 == 0 or outer <= 5:
            print(f"[OL-BFGS step {outer} | budget={budget_tracker.count}/"
                  f"{budget_tracker.budget}] loss={loss_val:.6e}  "
                  f"alpha={step_info['alpha']:.4e}  ls_evals={step_info['ls_evals']}  "
                  f"cautious_skip={step_info['cautious_skip']}")

        _maybe_eval(budget_tracker.count, "overlap_lbfgs_scratch")

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

    overlap_sampler = OverlapResampleSampler2D(
        N=config["N_coll"], overlap_frac=config["overlap_frac"], device=device,
    )
    tracker = BudgetTracker(budget=config["budget"])

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "overlap-lbfgs-pinn"),
        name=f"from_scratch_olbfgs_poisson2d_s{seed}",
        config={**config, "seed": seed, "method": "from_scratch_overlap_lbfgs", "problem": "poisson_2d"},
        mode=os.environ.get("WANDB_MODE", "offline"),
        reinit=True,
    )

    results = train_from_scratch_overlap_lbfgs_poisson2d(
        model=model, overlap_sampler=overlap_sampler,
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
    parser.add_argument("--overlap_frac", type=float, default=0.5)
    parser.add_argument("--lambda_bc", type=float, default=10.0)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    config = {
        "layers": [2, 50, 50, 50, 50, 1],
        "lambda_bc": args.lambda_bc,
        "N_coll": 2000,
        "N_per_edge": 200,
        "budget": args.budget,
        "overlap_frac": args.overlap_frac,
        "eval_interval": 500,
        "device": args.device,
    }

    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(base_dir, "outputs", "from_scratch_overlap_lbfgs_poisson2d")
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
        "method": "from_scratch_overlap_lbfgs",
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
        print(f"  Seed {r['seed']}: "
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
