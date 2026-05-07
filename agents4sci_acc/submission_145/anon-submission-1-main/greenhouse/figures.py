import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .runners import eval_rule, eval_rule_linreg
from .optimizers import baseline_rules
from .dsl import Rule

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_history(hist, path):
    plt.figure(figsize=(6,4))
    plt.plot(hist["gen"], hist["best_loss_train"], label="Best (train)")
    plt.plot(hist["gen"], hist["mean_loss_train"], label="Mean (train)")
    plt.xlabel("Generation"); plt.ylabel("Loss")
    plt.title("Evolutionary Progress (Train=Rastrigin)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def plot_bench_curves(comp: Dict, bench: str, path: str):
    plt.figure(figsize=(6,4))
    for name, data in comp[bench].items():
        plt.plot(data["curves"], label=name)
    plt.xlabel("Steps"); plt.ylabel("Mean Loss (3 seeds)")
    plt.title(f"Bench: {bench}")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def plot_bench_with_err(rule_extra: Rule, bench: str, path: str):
    rules = baseline_rules(); rules["Evolved_v02"] = rule_extra
    fig, ax = plt.subplots(figsize=(6,4))
    for name, r in rules.items():
        _, stack = eval_rule(r, bench, dim=10, steps=300, seeds=(0,1,2))
        mean = stack.mean(axis=0); std = stack.std(axis=0); xs = np.arange(len(mean))
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2)
    ax.set_xlabel("Steps"); ax.set_ylabel("Loss (mean ± std)")
    ax.set_title(f"Bench: {bench} (with error bands)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

def plot_linreg_with_err(rule_extra: Rule, path: str):
    rules = baseline_rules(); rules["Evolved_v02"] = rule_extra
    fig, ax = plt.subplots(figsize=(6,4))
    for name, r in rules.items():
        _, stack = eval_rule_linreg(r, steps=300, seeds=(0,1,2))
        mean = stack.mean(axis=0); std = stack.std(axis=0); xs = np.arange(len(mean))
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2)
    ax.set_xlabel("Steps"); ax.set_ylabel("MSE (mean ± std)")
    ax.set_title("Linear Regression (with error bands)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

def plot_bench_lines(rule_extra: Rule, bench: str, path: str):
    rules = baseline_rules(); rules["Evolved_v02"] = rule_extra
    plt.figure(figsize=(6,4))
    for name, r in rules.items():
        _, stack = eval_rule(r, bench, dim=10, steps=300, seeds=(0,1,2))
        plt.plot(stack.mean(axis=0), label=name)
    plt.xlabel("Steps"); plt.ylabel("Loss (mean over 3 seeds)")
    plt.title(f"Bench: {bench}"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def plot_linreg_lines(rule_extra: Rule, path: str):
    rules = baseline_rules(); rules["Evolved_v02"] = rule_extra
    plt.figure(figsize=(6,4))
    for name, r in rules.items():
        _, stack = eval_rule_linreg(r, steps=300, seeds=(0,1,2))
        plt.plot(stack.mean(axis=0), label=name)
    plt.xlabel("Steps"); plt.ylabel("MSE (mean over 3 seeds)")
    plt.title("Linear Regression"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def plot_pareto(archive, path):
    xs = [a["rule_train_loss"] for a in archive]
    ys = [a["test_loss"] for a in archive]
    gens = [a["gen"] for a in archive]
    plt.figure(figsize=(5,5))
    sc = plt.scatter(xs, ys, c=gens, s=18)
    plt.xlabel("Train loss (Rastrigin)")
    plt.ylabel("Cross-bench loss (Ackley)")
    plt.title("Pareto Cloud of Elite Rules Across Generations")
    cbar = plt.colorbar(sc); cbar.set_label("Generation")
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def plot_token_heatmap(archive, path, bm, bv, a1, a2, p, eta):
    import numpy as np
    gens = sorted(list(set([a["gen"] for a in archive])))
    keys = ["bm","bv","a1","a2","p","eta"]
    value_spaces = {"bm": bm, "bv": bv, "a1": a1, "a2": a2, "p": p, "eta": eta}
    block_imgs = []
    for key in keys:
        vs = value_spaces[key]
        mat = np.zeros((len(vs), len(gens)))
        for j, g in enumerate(gens):
            elites = [a for a in archive if a["gen"]==g]
            for i, val in enumerate(vs):
                mat[i,j] = sum(1 for a in elites if a["rule"][key]==val) / max(1, len(elites))
        block_imgs.append(mat)
    total_rows = sum(m.shape[0] for m in block_imgs)
    canvas = np.zeros((total_rows, len(gens)))
    row = 0; labels = []
    for key, mat in zip(keys, block_imgs):
        r = mat.shape[0]
        canvas[row:row+r, :] = mat
        labels.append((key, row, row+r))
        row += r
    plt.figure(figsize=(8, max(4, total_rows*0.4)))
    plt.imshow(canvas, aspect='auto', interpolation='nearest')
    plt.xlabel("Generation"); plt.ylabel("Token groups (stacked)")
    plt.title("Elite Token Frequencies Across Generations")
    yticks = []; ylabels = []
    for key, r0, r1 in labels:
        mid = (r0 + r1 - 1)/2
        yticks.append(mid); ylabels.append(key)
        plt.hlines([r1-0.5], xmin=-0.5, xmax=canvas.shape[1]-0.5, colors='white', linewidth=0.5)
    plt.yticks(yticks, ylabels)
    plt.colorbar(label="Frequency")
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()
