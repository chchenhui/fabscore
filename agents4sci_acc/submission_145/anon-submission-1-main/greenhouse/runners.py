import numpy as np
from typing import Tuple
from .dsl import Rule
from .benches import BENCHES

def _safe_update(x, step):
    x_new = x - step
    if not np.all(np.isfinite(x_new)):
        x_new = x - 0.1 * step
        if not np.all(np.isfinite(x_new)):
            x_new = x.copy()
    return x_new

def run_optimizer(rule: Rule, f, g, dim=10, steps=300, seed=0, clip_grad=10.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(dim,))
    m = np.zeros_like(x); v = np.zeros_like(x)
    losses = []
    for _ in range(steps):
        grad = g(x)
        grad = np.clip(grad, -clip_grad, clip_grad)
        m = rule.bm * m + (1 - rule.bm) * (grad ** rule.am)
        v = rule.bv * v + (1 - rule.bv) * ((grad ** 2) ** rule.av)
        denom = (np.sqrt(np.maximum(v, 0.0)) + rule.eps) ** rule.p
        step = rule.eta * (rule.a1 * grad + rule.a2 * m) / (denom + 1e-12)
        step = np.clip(step, -clip_grad, clip_grad)
        x = _safe_update(x, step)
        loss = f(x)
        if not np.isfinite(loss):
            x = np.clip(x, -1.0, 1.0)
            loss = f(x)
        losses.append(loss)
    return np.array(losses)

def eval_rule(rule: Rule, bench_name: str, dim=10, steps=300, seeds=(0,1,2)):
    f, g = BENCHES[bench_name]
    curves = []
    finals = []
    for s in seeds:
        c = run_optimizer(rule, f, g, dim=dim, steps=steps, seed=s)
        curves.append(c)
        finals.append(c[-1])
    return float(np.mean(finals)), np.stack(curves, axis=0)

# Linear regression
def make_linreg_data(n=200, d=20, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d,))
    y = X @ w_true + rng.normal(scale=noise, size=(n,))
    return X, y

def linreg_loss_grad(w, X, y):
    n = X.shape[0]
    residual = X @ w - y
    loss = (residual @ residual) / n
    grad = (2.0 / n) * (X.T @ residual)
    return loss, grad

def run_optimizer_linreg(rule: Rule, X, y, steps=300, seed=0, clip_grad=10.0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    w = rng.normal(scale=0.5, size=(d,))
    m = np.zeros_like(w); v = np.zeros_like(w)
    losses = []
    for _ in range(steps):
        loss, grad = linreg_loss_grad(w, X, y)
        grad = np.clip(grad, -clip_grad, clip_grad)
        m = rule.bm * m + (1 - rule.bm) * (grad ** rule.am)
        v = rule.bv * v + (1 - rule.bv) * ((grad ** 2) ** rule.av)
        denom = (np.sqrt(np.maximum(v, 0.0)) + rule.eps) ** rule.p
        step = rule.eta * (rule.a1 * grad + rule.a2 * m) / (denom + 1e-12)
        step = np.clip(step, -clip_grad, clip_grad)
        w = w - step
        losses.append(loss)
    return np.array(losses)

def eval_rule_linreg(rule: Rule, steps=300, seeds=(0,1,2)):
    curves = []
    finals = []
    for s in seeds:
        X, y = make_linreg_data(seed=s)
        c = run_optimizer_linreg(rule, X, y, steps=steps, seed=s)
        curves.append(c)
        finals.append(c[-1])
    return float(np.mean(finals)), np.stack(curves, axis=0)
