# Unified PINN trainer with pluggable optimizer and sampler.
# Supports: (1) Adam with resampled collocation, (2) Adam warmstart then fixed-collocation L-BFGS,
#           (3) Adam warmstart then overlap-resampled L-BFGS.
# Tracks gradient-evaluation budget, logs to WandB, saves best checkpoint by B_err.
# NaN/Inf detection: stops immediately if loss becomes non-finite.

import copy
import math
import torch


def train_adam_resampling(model, problem_module, sampler, budget_tracker, config, data, wandb_run=None):
    device = config["device"]
    gamma = config["gamma"]
    lr = config["lr"]
    eval_interval = config.get("eval_interval", 500)

    x_data = data["x_data"]
    u_data = data["u_data"]
    h_data = data["h_data"]
    x_eval = data["x_data"]
    u_true = data["u_true"]
    h_true = data["h_true"]
    B_true = data["B_true"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors

    loss_history = []
    eval_history = []
    best_B_err = float("inf")
    best_state = None
    best_step = 0

    step = 0
    while not budget_tracker.exhausted():
        x_coll = sampler.sample()

        optimizer.zero_grad()
        loss, loss_e, loss_d = problem_module.total_loss(model, x_coll, x_data, u_data, h_data, gamma)

        if not math.isfinite(loss.item()):
            print(f"[STEP {step}] NaN/Inf loss detected: total={loss.item()}, pde={loss_e.item()}, data={loss_d.item()}")
            if wandb_run is not None:
                wandb_run.log({"train/nan_detected": 1, "step": step})
            break

        loss.backward()
        optimizer.step()
        budget_tracker.increment(1)
        step += 1

        loss_val = loss.item()
        loss_e_val = loss_e.item()
        loss_d_val = loss_d.item()
        loss_history.append({"step": step, "total_loss": loss_val, "pde_loss": loss_e_val, "data_loss": loss_d_val})

        if wandb_run is not None:
            wandb_run.log({
                "train/total_loss": loss_val,
                "train/pde_loss": loss_e_val,
                "train/data_loss": loss_d_val,
                "budget/count": budget_tracker.count,
            }, step=step)

        if step % 100 == 0:
            print(f"[STEP {step}/{budget_tracker.budget}] loss={loss_val:.6e}  pde={loss_e_val:.6e}  data={loss_d_val:.6e}")

        if step % eval_interval == 0 or budget_tracker.exhausted():
            errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)
            errs["step"] = step
            eval_history.append(errs)
            print(f"  -> B_err={errs['B_err']:.6e}  u_err={errs['u_err']:.6e}  h_err={errs['h_err']:.6e}")

            if wandb_run is not None:
                wandb_run.log({
                    "eval/B_err": errs["B_err"],
                    "eval/u_err": errs["u_err"],
                    "eval/h_err": errs["h_err"],
                }, step=step)

            if errs["B_err"] < best_B_err:
                best_B_err = errs["B_err"]
                best_state = copy.deepcopy(model.state_dict())
                best_step = step

    if best_state is not None:
        model.load_state_dict(best_state)
    final_errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)

    return {
        "loss_history": loss_history,
        "eval_history": eval_history,
        "best_step": best_step,
        "best_B_err": best_B_err,
        "final_errors": final_errs,
        "total_evals": budget_tracker.count,
    }


def train_adam_then_lbfgs(model, problem_module, sampler, budget_tracker, config, data, wandb_run=None):
    device = config["device"]
    gamma = config["gamma"]
    lr = config["lr"]
    adam_budget = config["adam_budget"]
    eval_interval = config.get("eval_interval", 500)

    x_data = data["x_data"]
    u_data = data["u_data"]
    h_data = data["h_data"]
    x_eval = data["x_data"]
    u_true = data["u_true"]
    h_true = data["h_true"]
    B_true = data["B_true"]

    from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors

    loss_history = []
    eval_history = []
    best_B_err = float("inf")
    best_state = None
    best_step = 0
    next_eval_at = eval_interval

    def _maybe_eval(budget_count, phase_label):
        nonlocal best_B_err, best_state, best_step, next_eval_at
        if budget_count < next_eval_at and not budget_tracker.exhausted():
            return
        errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)
        errs["step"] = budget_count
        errs["phase"] = phase_label
        eval_history.append(errs)
        print(f"  [{phase_label} budget={budget_count}] B_err={errs['B_err']:.6e}  u_err={errs['u_err']:.6e}  h_err={errs['h_err']:.6e}")
        if wandb_run is not None:
            wandb_run.log({"eval/B_err": errs["B_err"], "eval/u_err": errs["u_err"], "eval/h_err": errs["h_err"]}, step=budget_count)
        if errs["B_err"] < best_B_err:
            best_B_err = errs["B_err"]
            best_state = copy.deepcopy(model.state_dict())
            best_step = budget_count
        while next_eval_at <= budget_count:
            next_eval_at += eval_interval

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    adam_step = 0
    nan_detected = False

    print(f"=== PHASE 1: Adam + resampling (budget 0 -> {adam_budget}) ===")
    while budget_tracker.count < adam_budget and not budget_tracker.exhausted():
        x_coll = sampler.sample()
        adam_optimizer.zero_grad()
        loss, loss_e, loss_d = problem_module.total_loss(model, x_coll, x_data, u_data, h_data, gamma)

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
        loss_e_val = loss_e.item()
        loss_d_val = loss_d.item()
        loss_history.append({"step": budget_tracker.count, "total_loss": loss_val, "pde_loss": loss_e_val, "data_loss": loss_d_val, "phase": "adam"})

        if wandb_run is not None:
            wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e_val, "train/data_loss": loss_d_val, "budget/count": budget_tracker.count, "train/phase": 0}, step=budget_tracker.count)

        if adam_step % 100 == 0:
            print(f"[ADAM {budget_tracker.count}/{budget_tracker.budget}] loss={loss_val:.6e}  pde={loss_e_val:.6e}  data={loss_d_val:.6e}")

        _maybe_eval(budget_tracker.count, "adam")

    adam_evals = budget_tracker.count
    print(f"=== Phase 1 done: {adam_evals} gradient evals consumed ===")

    lbfgs_step_history = []
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
        x_coll_fixed = sampler.sample().detach().clone()
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
            loss, loss_e, loss_d = problem_module.total_loss(model, x_coll_fixed, x_data, u_data, h_data, gamma)

            if not math.isfinite(loss.item()):
                lbfgs_nan = True
                return loss

            if not budget_tracker.exhausted():
                loss.backward()
                budget_tracker.increment(1)
                closure_count_this_step += 1

            loss_val = loss.item()
            loss_e_val = loss_e.item()
            loss_d_val = loss_d.item()
            loss_history.append({"step": budget_tracker.count, "total_loss": loss_val, "pde_loss": loss_e_val, "data_loss": loss_d_val, "phase": "lbfgs"})

            if wandb_run is not None:
                wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e_val, "train/data_loss": loss_d_val, "budget/count": budget_tracker.count, "train/phase": 1}, step=budget_tracker.count)

            return loss

        def _gather_flat_params():
            return torch.cat([p.data.view(-1) for p in model.parameters()])

        prev_loss = None
        while not budget_tracker.exhausted() and not lbfgs_nan:
            closure_count_this_step = 0
            x_before = _gather_flat_params().clone()
            loss_out = lbfgs_optimizer.step(closure)
            x_after = _gather_flat_params()

            lbfgs_diagnostics["lbfgs_outer_steps"] += 1
            lbfgs_diagnostics["closure_calls_per_step"].append(closure_count_this_step)
            lbfgs_diagnostics["lbfgs_closure_calls"] += closure_count_this_step

            current_loss = loss_out.item() if loss_out is not None else float("nan")
            s_vec = x_after - x_before
            s_norm = s_vec.norm().item()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            lbfgs_step_history.append({
                "iter": lbfgs_diagnostics["lbfgs_outer_steps"],
                "grad_norm": grad_norm,
                "alpha": None,
                "ls_evals": closure_count_this_step,
                "ls_failed": False,
                "cautious_skip": None,
                "ys_value": None,
                "y_norm": None,
                "s_norm": s_norm,
            })

            if wandb_run is not None:
                wandb_run.log({
                    "lbfgs/grad_norm": grad_norm,
                    "lbfgs/s_norm": s_norm,
                    "lbfgs/closure_calls": closure_count_this_step,
                }, step=budget_tracker.count)

            if lbfgs_diagnostics["lbfgs_outer_steps"] % 10 == 0 or lbfgs_diagnostics["lbfgs_outer_steps"] <= 5:
                print(f"[LBFGS step {lbfgs_diagnostics['lbfgs_outer_steps']} | budget={budget_tracker.count}/{budget_tracker.budget}] loss={current_loss:.6e}  closures={closure_count_this_step}  grad_norm={grad_norm:.2e}")

            _maybe_eval(budget_tracker.count, "lbfgs")

            if lbfgs_nan:
                lbfgs_diagnostics["termination_reason"] = "nan"
                break

            if closure_count_this_step == 0:
                lbfgs_diagnostics["termination_reason"] = "budget_exhausted"
                break

            if prev_loss is not None:
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
    final_errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)

    return {
        "loss_history": loss_history,
        "eval_history": eval_history,
        "best_step": best_step,
        "best_B_err": best_B_err,
        "final_errors": final_errs,
        "total_evals": budget_tracker.count,
        "lbfgs_diagnostics": lbfgs_diagnostics,
        "lbfgs_step_history": lbfgs_step_history,
    }



def train_adam_then_overlap_lbfgs(model, problem_module, overlap_sampler,
                                  budget_tracker, config, data, wandb_run=None):
    device = config["device"]
    gamma = config["gamma"]
    lr = config["lr"]
    adam_budget = config["adam_budget"]
    adam_fixed_budget = config.get("adam_fixed_budget", 0)
    eval_interval = config.get("eval_interval", 500)

    x_data = data["x_data"]
    u_data = data["u_data"]
    h_data = data["h_data"]
    x_eval = data["x_data"]
    u_true = data["u_true"]
    h_true = data["h_true"]
    B_true = data["B_true"]

    from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors
    from overlap_lbfgs_pinn.samplers.collocation import ResampleSampler
    from overlap_lbfgs_pinn.optimizers.overlap_lbfgs import OverlapLBFGS

    loss_history = []
    eval_history = []
    best_B_err = float("inf")
    best_state = None
    best_step = 0
    next_eval_at = eval_interval

    def _maybe_eval(budget_count, phase_label):
        nonlocal best_B_err, best_state, best_step, next_eval_at
        if budget_count < next_eval_at and not budget_tracker.exhausted():
            return
        errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)
        errs["step"] = budget_count
        errs["phase"] = phase_label
        eval_history.append(errs)
        print(f"  [{phase_label} budget={budget_count}] B_err={errs['B_err']:.6e}  "
              f"u_err={errs['u_err']:.6e}  h_err={errs['h_err']:.6e}")
        if wandb_run is not None:
            wandb_run.log({"eval/B_err": errs["B_err"], "eval/u_err": errs["u_err"],
                           "eval/h_err": errs["h_err"]}, step=budget_count)
        if errs["B_err"] < best_B_err:
            best_B_err = errs["B_err"]
            best_state = copy.deepcopy(model.state_dict())
            best_step = budget_count
        while next_eval_at <= budget_count:
            next_eval_at += eval_interval

    adam_sampler = ResampleSampler(N=config["N_coll"], device=device)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    adam_step = 0
    nan_detected = False

    print(f"=== PHASE 1: Adam + resampling (budget 0 -> {adam_budget}) ===")
    while budget_tracker.count < adam_budget and not budget_tracker.exhausted():
        x_coll = adam_sampler.sample()
        adam_optimizer.zero_grad()
        loss, loss_e, loss_d = problem_module.total_loss(model, x_coll, x_data, u_data, h_data, gamma)

        if not math.isfinite(loss.item()):
            print(f"[ADAM STEP {adam_step}] NaN/Inf loss: total={loss.item()}")
            nan_detected = True
            break

        loss.backward()
        adam_optimizer.step()
        budget_tracker.increment(1)
        adam_step += 1

        loss_val = loss.item()
        loss_e_val = loss_e.item()
        loss_d_val = loss_d.item()
        loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                             "pde_loss": loss_e_val, "data_loss": loss_d_val,
                             "phase": "adam"})

        if wandb_run is not None:
            wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e_val,
                           "train/data_loss": loss_d_val,
                           "budget/count": budget_tracker.count,
                           "train/phase": 0}, step=budget_tracker.count)

        if adam_step % 100 == 0:
            print(f"[ADAM {budget_tracker.count}/{budget_tracker.budget}] "
                  f"loss={loss_val:.6e}  pde={loss_e_val:.6e}  data={loss_d_val:.6e}")

        _maybe_eval(budget_tracker.count, "adam")

    adam_evals = budget_tracker.count
    print(f"=== Phase 1 done: {adam_evals} gradient evals consumed ===")

    lbfgs_step_history = []
    lbfgs_diagnostics = {
        "adam_evals": adam_evals,
        "adam_fixed_evals": 0,
        "lbfgs_outer_steps": 0,
        "lbfgs_line_search_evals": 0,
        "overlap_grad_evals": 0,
        "cautious_skips": 0,
        "line_search_failures": 0,
        "termination_reason": "not_started",
    }

    if nan_detected or budget_tracker.exhausted():
        lbfgs_diagnostics["termination_reason"] = (
            "nan_in_adam" if nan_detected else "budget_exhausted_in_adam"
        )
    else:
        adam_fixed_end = budget_tracker.count + adam_fixed_budget
        if adam_fixed_budget > 0 and not budget_tracker.exhausted():
            full_set_init, _ = overlap_sampler.sample()
            x_coll_fixed = full_set_init.detach().clone()
            print(f"=== PHASE 2: Adam + fixed collocation (budget {budget_tracker.count} -> {adam_fixed_end}) ===")
            adam_fixed_step = 0
            while budget_tracker.count < adam_fixed_end and not budget_tracker.exhausted():
                adam_optimizer.zero_grad()
                loss, loss_e, loss_d = problem_module.total_loss(
                    model, x_coll_fixed, x_data, u_data, h_data, gamma)
                if not math.isfinite(loss.item()):
                    print(f"[ADAM-FIXED STEP {adam_fixed_step}] NaN/Inf loss")
                    nan_detected = True
                    break
                loss.backward()
                adam_optimizer.step()
                budget_tracker.increment(1)
                adam_fixed_step += 1
                loss_val = loss.item()
                loss_e_val = loss_e.item()
                loss_d_val = loss_d.item()
                loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                                     "pde_loss": loss_e_val, "data_loss": loss_d_val,
                                     "phase": "adam_fixed"})
                if wandb_run is not None:
                    wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e_val,
                                   "train/data_loss": loss_d_val,
                                   "budget/count": budget_tracker.count, "train/phase": 0.5},
                                  step=budget_tracker.count)
                if adam_fixed_step % 100 == 0:
                    print(f"[ADAM-FIXED {budget_tracker.count}/{budget_tracker.budget}] "
                          f"loss={loss_val:.6e}  pde={loss_e_val:.6e}  data={loss_d_val:.6e}")
                _maybe_eval(budget_tracker.count, "adam_fixed")
            lbfgs_diagnostics["adam_fixed_evals"] = adam_fixed_step
            print(f"=== Phase 2 done: {budget_tracker.count} gradient evals consumed ===")

        if nan_detected or budget_tracker.exhausted():
            lbfgs_diagnostics["termination_reason"] = (
                "nan_in_adam_fixed" if nan_detected else "budget_exhausted_in_adam_fixed"
            )
        else:
            print(f"=== PHASE 3: Overlap-resampled L-BFGS (budget {budget_tracker.count} -> "
                  f"{budget_tracker.budget}, gamma={gamma:.4f}) ===")

            olbfgs = OverlapLBFGS(model.parameters(), history_size=20,
                                  c1=1e-4, c2=0.9, max_ls=20, cautious_eps=1e-6)

            lbfgs_nan = False

            while not budget_tracker.exhausted() and not lbfgs_nan:
                full_set, overlap_set = overlap_sampler.sample()

                def make_full_closure(x_coll_):
                    def _closure():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                        loss, loss_e, loss_d = problem_module.total_loss(
                            model, x_coll_, x_data, u_data, h_data, gamma)
                        if not math.isfinite(loss.item()):
                            nonlocal lbfgs_nan
                            lbfgs_nan = True
                            return (loss.item(), loss_e.item(), loss_d.item())
                        loss.backward()
                        return (loss.item(), loss_e.item(), loss_d.item())
                    return _closure

                def make_overlap_closure(x_overlap_):
                    def _closure():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                        residual = problem_module.pde_residual(model, x_overlap_)
                        loss_e_overlap = torch.mean(residual ** 2)
                        loss_d = problem_module.data_loss(model, x_data, u_data, h_data)
                        loss = gamma * loss_e_overlap + (1.0 - gamma) * loss_d
                        if not math.isfinite(loss.item()):
                            nonlocal lbfgs_nan
                            lbfgs_nan = True
                            return (loss.item(), loss_e_overlap.item(), loss_d.item())
                        loss.backward()
                        return (loss.item(), loss_e_overlap.item(), loss_d.item())
                    return _closure

                full_closure = make_full_closure(full_set)
                overlap_closure = make_overlap_closure(overlap_set)

                step_info = olbfgs.step(full_closure, overlap_closure, budget_tracker)

                lbfgs_diagnostics["lbfgs_outer_steps"] += 1
                lbfgs_diagnostics["lbfgs_line_search_evals"] += step_info["ls_evals"]
                lbfgs_diagnostics["overlap_grad_evals"] += step_info["overlap_evals"]
                lbfgs_diagnostics["cautious_skips"] = olbfgs.diagnostics["cautious_skips"]
                lbfgs_diagnostics["line_search_failures"] = olbfgs.diagnostics["line_search_failures"]

                lbfgs_step_history.append({
                    "iter": lbfgs_diagnostics["lbfgs_outer_steps"],
                    "grad_norm": step_info["grad_norm"],
                    "alpha": step_info["alpha"],
                    "ls_evals": step_info["ls_evals"],
                    "ls_failed": step_info["ls_failed"],
                    "cautious_skip": step_info["cautious_skip"],
                    "ys_value": step_info.get("ys_value"),
                    "y_norm": step_info.get("y_norm"),
                    "s_norm": step_info.get("s_norm"),
                })

                loss_val = step_info["loss"]
                loss_e_val = step_info["loss_e"]
                loss_d_val = step_info["loss_d"]
                loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                                     "pde_loss": loss_e_val, "data_loss": loss_d_val,
                                     "phase": "overlap_lbfgs"})

                if wandb_run is not None:
                    wb_log = {
                        "train/total_loss": loss_val, "train/pde_loss": loss_e_val,
                        "train/data_loss": loss_d_val,
                        "budget/count": budget_tracker.count, "train/phase": 1,
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
    print(f"  Adam resampled evals: {lbfgs_diagnostics['adam_evals']}")
    print(f"  Adam fixed evals: {lbfgs_diagnostics.get('adam_fixed_evals', 0)}")
    print(f"  Overlap-LBFGS outer steps: {lbfgs_diagnostics['lbfgs_outer_steps']}")
    print(f"  Overlap-LBFGS line-search evals: {lbfgs_diagnostics['lbfgs_line_search_evals']}")
    print(f"  Overlap gradient evals: {lbfgs_diagnostics['overlap_grad_evals']}")
    print(f"  Cautious skips: {lbfgs_diagnostics['cautious_skips']}")
    print(f"  Line search failures: {lbfgs_diagnostics['line_search_failures']}")
    print(f"  Termination: {lbfgs_diagnostics['termination_reason']}")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)

    return {
        "loss_history": loss_history,
        "eval_history": eval_history,
        "best_step": best_step,
        "best_B_err": best_B_err,
        "final_errors": final_errs,
        "total_evals": budget_tracker.count,
        "lbfgs_diagnostics": lbfgs_diagnostics,
        "lbfgs_step_history": lbfgs_step_history,
    }


def train_adam_then_naive_resampled_lbfgs(model, problem_module, sampler,
                                          budget_tracker, config, data, wandb_run=None):
    device = config["device"]
    gamma = config["gamma"]
    lr = config["lr"]
    adam_budget = config["adam_budget"]
    adam_fixed_budget = config.get("adam_fixed_budget", 0)
    eval_interval = config.get("eval_interval", 500)

    x_data = data["x_data"]
    u_data = data["u_data"]
    h_data = data["h_data"]
    x_eval = data["x_data"]
    u_true = data["u_true"]
    h_true = data["h_true"]
    B_true = data["B_true"]

    from overlap_lbfgs_pinn.evaluation.metrics import relative_l2_errors
    from overlap_lbfgs_pinn.samplers.collocation import ResampleSampler
    from overlap_lbfgs_pinn.optimizers.overlap_lbfgs import OverlapLBFGS

    loss_history = []
    eval_history = []
    best_B_err = float("inf")
    best_state = None
    best_step = 0
    next_eval_at = eval_interval

    def _maybe_eval(budget_count, phase_label):
        nonlocal best_B_err, best_state, best_step, next_eval_at
        if budget_count < next_eval_at and not budget_tracker.exhausted():
            return
        errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)
        errs["step"] = budget_count
        errs["phase"] = phase_label
        eval_history.append(errs)
        print(f"  [{phase_label} budget={budget_count}] B_err={errs['B_err']:.6e}  "
              f"u_err={errs['u_err']:.6e}  h_err={errs['h_err']:.6e}")
        if wandb_run is not None:
            wandb_run.log({"eval/B_err": errs["B_err"], "eval/u_err": errs["u_err"],
                           "eval/h_err": errs["h_err"]}, step=budget_count)
        if errs["B_err"] < best_B_err:
            best_B_err = errs["B_err"]
            best_state = copy.deepcopy(model.state_dict())
            best_step = budget_count
        while next_eval_at <= budget_count:
            next_eval_at += eval_interval

    adam_sampler = ResampleSampler(N=config["N_coll"], device=device)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    adam_step = 0
    nan_detected = False

    print(f"=== PHASE 1: Adam + resampling (budget 0 -> {adam_budget}) ===")
    while budget_tracker.count < adam_budget and not budget_tracker.exhausted():
        x_coll = adam_sampler.sample()
        adam_optimizer.zero_grad()
        loss, loss_e, loss_d = problem_module.total_loss(model, x_coll, x_data, u_data, h_data, gamma)
        if not math.isfinite(loss.item()):
            print(f"[ADAM STEP {adam_step}] NaN/Inf loss: total={loss.item()}")
            nan_detected = True
            break
        loss.backward()
        adam_optimizer.step()
        budget_tracker.increment(1)
        adam_step += 1
        loss_val = loss.item()
        loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                             "pde_loss": loss_e.item(), "data_loss": loss_d.item(), "phase": "adam"})
        if wandb_run is not None:
            wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e.item(),
                           "train/data_loss": loss_d.item(), "budget/count": budget_tracker.count,
                           "train/phase": 0}, step=budget_tracker.count)
        if adam_step % 100 == 0:
            print(f"[ADAM {budget_tracker.count}/{budget_tracker.budget}] "
                  f"loss={loss_val:.6e}  pde={loss_e.item():.6e}  data={loss_d.item():.6e}")
        _maybe_eval(budget_tracker.count, "adam")

    adam_evals = budget_tracker.count
    print(f"=== Phase 1 done: {adam_evals} gradient evals consumed ===")

    lbfgs_step_history = []
    lbfgs_diagnostics = {
        "adam_evals": adam_evals,
        "adam_fixed_evals": 0,
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
        if adam_fixed_budget > 0:
            x_coll_fixed = adam_sampler.sample().detach().clone()
            fixed_target = adam_evals + adam_fixed_budget
            print(f"=== PHASE 2: Adam + fixed collocation (budget {adam_evals} -> {fixed_target}) ===")
            while budget_tracker.count < fixed_target and not budget_tracker.exhausted():
                adam_optimizer.zero_grad()
                loss, loss_e, loss_d = problem_module.total_loss(
                    model, x_coll_fixed, x_data, u_data, h_data, gamma)
                if not math.isfinite(loss.item()):
                    nan_detected = True
                    break
                loss.backward()
                adam_optimizer.step()
                budget_tracker.increment(1)
                adam_step += 1
                loss_val = loss.item()
                loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                                     "pde_loss": loss_e.item(), "data_loss": loss_d.item(),
                                     "phase": "adam_fixed"})
                if wandb_run is not None:
                    wandb_run.log({"train/total_loss": loss_val, "train/pde_loss": loss_e.item(),
                                   "train/data_loss": loss_d.item(), "budget/count": budget_tracker.count,
                                   "train/phase": 0.5}, step=budget_tracker.count)
                if adam_step % 100 == 0:
                    print(f"[ADAM-FIXED {budget_tracker.count}/{budget_tracker.budget}] loss={loss_val:.6e}")
                _maybe_eval(budget_tracker.count, "adam_fixed")
            lbfgs_diagnostics["adam_fixed_evals"] = budget_tracker.count - adam_evals
            print(f"=== Phase 2 done: {lbfgs_diagnostics['adam_fixed_evals']} fixed evals ===")

        if not nan_detected and not budget_tracker.exhausted():
            print(f"=== PHASE 3: Naive-resampled L-BFGS o=0 (budget {budget_tracker.count} -> {budget_tracker.budget}) ===")
            olbfgs = OverlapLBFGS(model.parameters(), history_size=20, c1=1e-4, c2=0.9, max_ls=20)
            lbfgs_nan = False

            while not budget_tracker.exhausted() and not lbfgs_nan:
                fresh_batch = sampler.sample()

                def make_closure(x_coll_):
                    def _closure():
                        nonlocal lbfgs_nan
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                        loss, loss_e, loss_d = problem_module.total_loss(
                            model, x_coll_, x_data, u_data, h_data, gamma)
                        if not math.isfinite(loss.item()):
                            lbfgs_nan = True
                            return (loss.item(), loss_e.item(), loss_d.item())
                        loss.backward()
                        return (loss.item(), loss_e.item(), loss_d.item())
                    return _closure

                full_closure = make_closure(fresh_batch)
                overlap_closure = make_closure(fresh_batch)

                step_info = olbfgs.step(full_closure, overlap_closure, budget_tracker)

                lbfgs_diagnostics["lbfgs_outer_steps"] += 1
                lbfgs_diagnostics["lbfgs_line_search_evals"] += step_info["ls_evals"]
                lbfgs_diagnostics["overlap_grad_evals"] += step_info["overlap_evals"]
                lbfgs_diagnostics["cautious_skips"] = olbfgs.diagnostics["cautious_skips"]
                lbfgs_diagnostics["line_search_failures"] = olbfgs.diagnostics["line_search_failures"]

                lbfgs_step_history.append({
                    "iter": lbfgs_diagnostics["lbfgs_outer_steps"],
                    "grad_norm": step_info["grad_norm"],
                    "alpha": step_info["alpha"],
                    "ls_evals": step_info["ls_evals"],
                    "ls_failed": step_info["ls_failed"],
                    "cautious_skip": step_info["cautious_skip"],
                    "ys_value": step_info.get("ys_value"),
                    "y_norm": step_info.get("y_norm"),
                    "s_norm": step_info.get("s_norm"),
                })

                loss_val = step_info["loss"]
                loss_e_val = step_info["loss_e"]
                loss_d_val = step_info["loss_d"]
                loss_history.append({"step": budget_tracker.count, "total_loss": loss_val,
                                     "pde_loss": loss_e_val, "data_loss": loss_d_val,
                                     "phase": "naive_lbfgs"})

                if wandb_run is not None:
                    wb_log = {
                        "train/total_loss": loss_val, "train/pde_loss": loss_e_val,
                        "train/data_loss": loss_d_val,
                        "budget/count": budget_tracker.count, "train/phase": 2,
                        "lbfgs/cautious_skips": olbfgs.diagnostics["cautious_skips"],
                        "lbfgs/line_search_evals": step_info["ls_evals"],
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
                    print(f"[NAIVE-LBFGS step {outer} | budget={budget_tracker.count}/"
                          f"{budget_tracker.budget}] loss={loss_val:.6e}  "
                          f"alpha={step_info['alpha']:.4e}  ls_evals={step_info['ls_evals']}  "
                          f"cautious_skip={step_info['cautious_skip']}  ls_failed={step_info['ls_failed']}")

                _maybe_eval(budget_tracker.count, "naive_lbfgs")

                if lbfgs_nan or step_info.get("nan_detected", False):
                    lbfgs_diagnostics["termination_reason"] = "nan"
                    break
                if step_info["grad_norm"] < 1e-11:
                    lbfgs_diagnostics["termination_reason"] = "gradient_tolerance"
                    print(f"  Naive-LBFGS converged: grad_norm={step_info['grad_norm']:.2e}")
                    break

            if lbfgs_diagnostics["termination_reason"] == "not_started":
                lbfgs_diagnostics["termination_reason"] = "budget_exhausted"

    _maybe_eval(budget_tracker.count, "final")

    print(f"=== Training complete: {budget_tracker.count} total gradient evals ===")
    print(f"  Adam resampled evals: {lbfgs_diagnostics['adam_evals']}")
    print(f"  Adam fixed evals: {lbfgs_diagnostics.get('adam_fixed_evals', 0)}")
    print(f"  Naive-LBFGS outer steps: {lbfgs_diagnostics['lbfgs_outer_steps']}")
    print(f"  Line-search evals: {lbfgs_diagnostics['lbfgs_line_search_evals']}")
    print(f"  Cautious skips: {lbfgs_diagnostics['cautious_skips']}")
    print(f"  Line search failures: {lbfgs_diagnostics['line_search_failures']}")
    print(f"  Termination: {lbfgs_diagnostics['termination_reason']}")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_errs = relative_l2_errors(model, x_eval, u_true, h_true, B_true)

    return {
        "loss_history": loss_history,
        "eval_history": eval_history,
        "best_step": best_step,
        "best_B_err": best_B_err,
        "final_errors": final_errs,
        "total_evals": budget_tracker.count,
        "lbfgs_diagnostics": lbfgs_diagnostics,
        "lbfgs_step_history": lbfgs_step_history,
    }
