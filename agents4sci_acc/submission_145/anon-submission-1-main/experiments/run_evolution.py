import os, json, time
from greenhouse.evo import evolve
from greenhouse.dsl import BM_CHOICES, BV_CHOICES, A1_CHOICES, A2_CHOICES, P_CHOICES, ETA_CHOICES
from greenhouse.figures import ensure_dir, plot_history, plot_pareto, plot_token_heatmap

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "artifacts"); ensure_dir(ART)

cfg = json.load(open(os.path.join(HERE, "config.json")))
start = time.time()
best_rule, best_train, best_test, hist, archive = evolve(
    bench_train=cfg["bench_train"], bench_test=cfg["bench_test"],
    dim=cfg["dim"], steps=cfg["steps"], seeds=tuple(cfg["seeds_evo"]),
    pop_size=cfg["population"], elites=cfg["elites"], gens=cfg["generations"], mutate_prob=cfg["mutate_prob"]
)
elapsed = time.time() - start

json.dump({
    "best_rule": best_rule.to_dict(),
    "pretty": best_rule.pretty(),
    "train_loss": best_train,
    "test_loss": best_test,
    "elapsed_sec": elapsed
}, open(os.path.join(ART, "best_rule_v02.json"), "w"), indent=2)

json.dump(archive, open(os.path.join(ART, "archive_v02.json"), "w"), indent=2)

plot_history(hist, os.path.join(ART, "evo_history_v02.png"))
plot_pareto(archive, os.path.join(ART, "pareto_v02.png"))
plot_token_heatmap(
    archive, os.path.join(ART, "token_heatmap_v02.png"),
    bm=BM_CHOICES, bv=BV_CHOICES, a1=A1_CHOICES, a2=A2_CHOICES, p=P_CHOICES, eta=ETA_CHOICES
)

print("Saved: best_rule_v02.json, archive_v02.json and figures in artifacts/")
