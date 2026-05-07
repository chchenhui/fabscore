
# multiway_scoreboard.py
# Evaluate CFR, MCCFR, DeepCFR, NFSP, Random on synthetic multiway decision states (k=3..6)
# using a heuristic multiway proxy q_k. Models are trained on HUEnv.
import importlib.util, numpy as np, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Load modules from the existing codebase
def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(BASE/relpath))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

env      = _load("poker_env",      "env.py")
cfr      = _load("poker_cfr",      "cfr.py")
deep_cfr = _load("poker_deep_cfr", "deep_cfr.py")
nfsp     = _load("poker_nfsp",     "nfsp.py")

EPS = 1e-12
ACTIONS = ["check/call","bet/raise","fold"]

def top1(P,Q): return float((P.argmax(1)==Q.argmax(1)).mean())
def kl_pq(P,Q): return float(np.mean(np.sum(P*(np.log(P+EPS)-np.log(Q+EPS)),axis=1)))
def ce_qp(Q,P): return float(np.mean(-np.sum(Q*np.log(P+EPS),axis=1)))

def policy_from_table(table):
    def f(obs):
        I = cfr.infoset_key(obs)
        dist = table.get(I, {})
        v = np.array([float(dist.get(a,0.0)) for a in ACTIONS], dtype=float)
        s = v.sum()
        return v/s if s>0 else np.ones(3)/3
    return f

def multiway_proxy_policy(k):
    t_raise = min(0.85, 0.60 + 0.05*(k-2))
    t_fold  = max(0.10, 0.35 - 0.03*(k-2))
    slope = 10.0
    def sigma(x): return 1.0/(1.0+np.exp(-slope*x))
    def f(obs):
        e = float(obs.get("equity", 0.5))
        e_k = e**(k-1)               # proxy: prob to beat k-1 opponents (independent)
        pr = sigma(e_k - t_raise)
        pf = sigma(t_fold - e_k)
        pc = max(0.0, 1.0 - max(pr, pf))
        vec = np.array([pc, pr, pf], dtype=float)
        s = vec.sum()
        return vec/s if s>0 else np.ones(3)/3
    return f

def sample_obs(env_inst, n, seed=7):
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(n):
        env_inst.reset(); env_inst.deal()
        env_inst.to_act = int(rng.integers(0,2))
        obs.append(env_inst.observe())
    return obs

def train_cfr(env_inst, T):
    m = cfr.CFRTabular(); m.iterate(env_inst, T=T)
    return policy_from_table(m.average_policy())

def train_mccfr(env_inst, T):
    m = cfr.MCCFRExternal(); m.iterate(env_inst, T=T)
    return policy_from_table(m.average_policy())

def train_deepcfr(env_inst, T):
    d = deep_cfr.DeepCFR(seed=0)
    d.collect_traversals(env_inst, n=T)
    d.train(steps=max(1, T//50))
    return lambda obs: np.array(d.action_probs(obs), dtype=float)

def train_nfsp(env_inst, T):
    n = nfsp.NFSP()
    if hasattr(n, "train"):
        n.train(env_inst, T=T)
    return lambda obs: np.array(n.avg_action_probs(deep_cfr.obs_to_vec(obs)), dtype=float)

def run_multiway_scoreboard(proxy_iters=1000, model_iters=1000, n_per_k=1000, seed=7, out_tex="multiway_gto_accuracy.tex"):
    e = env.HUEnv()
    # Train HU proxy (not used directly for q_k, but keeps parity with HU training budget)
    proxy = cfr.MCCFRExternal(); proxy.iterate(e, T=proxy_iters)

    models = {
        "CFR":     train_cfr(e, model_iters),
        "MCCFR":   train_mccfr(e, model_iters),
        "DeepCFR": train_deepcfr(e, model_iters),
        "NFSP":    train_nfsp(e, model_iters),
        "Random":  (lambda _obs: np.array([1/3,1/3,1/3], dtype=float)),
    }

    rng = np.random.default_rng(seed)
    rows = []
    for k in [3,4,5,6]:
        obs_list = sample_obs(e, n_per_k, seed=int(rng.integers(0, 2**31-1)))
        qk = multiway_proxy_policy(k)
        Q = np.vstack([qk(o) for o in obs_list])
        for name, pi in models.items():
            P = np.vstack([pi(o) for o in obs_list])
            rows.append({
                "Players": k,
                "Model": name,
                "Top-1 \\uparrow": top1(P,Q),
                "KL(p\\|\\,q) \\downarrow": kl_pq(P,Q),
                "CE(q, p) \\downarrow": ce_qp(Q,P),
            })
    df = pd.DataFrame(rows).sort_values(["Players","Model"])

    caption = ("Multiway (k=3--6) accuracy relative to a heuristic multiway proxy $q_k$ on synthetic NLHE decision states. "
               "Top-1 is argmax agreement; KL$(p\\|q)$ and CE$(q,p)$ quantify distributional distance (lower is better). "
               f"Per k, n={n_per_k} states; models trained on HU for {model_iters} iterations; MCCFR trained for {proxy_iters} iterations.")
    tex = df.to_latex(index=False, float_format="%.3f", escape=False, column_format="r l c c c",
                      caption=caption, label="tab:multiway-accuracy", bold_rows=False)
    Path(out_tex).write_text("\\begin{table}[h!]\n\\centering\n" + tex.replace("\\begin{table}","").replace("\\end{table}","") + "\n\\end{table}\n")
    print("Saved:", out_tex)
    return df

if __name__ == "__main__":
    run_multiway_scoreboard()
