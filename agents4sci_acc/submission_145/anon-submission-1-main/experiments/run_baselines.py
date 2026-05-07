import os, json
from greenhouse.optimizers import baseline_rules
from greenhouse.runners import eval_rule
from greenhouse.figures import ensure_dir, plot_bench_curves
from greenhouse.dsl import Rule
from greenhouse.benches import BENCHES

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "artifacts"); ensure_dir(ART)

best = json.load(open(os.path.join(ART, "best_rule_v02.json")))
best_rule = Rule(**best["best_rule"])

def compare_on_benches(steps=300):
    out = {}
    rules = baseline_rules()
    rules["Evolved_v02"] = best_rule
    for bench in BENCHES.keys():
        res = {}
        for name, r in rules.items():
            loss, curves = eval_rule(r, bench, dim=10, steps=steps, seeds=(0,1,2))
            res[name] = {"final_loss": loss, "curves": curves.mean(axis=0).tolist()}
        out[bench] = res
    return out

comp = compare_on_benches(steps=300)
json.dump(comp, open(os.path.join(ART, "comparison_v02.json"), "w"), indent=2)

# Lines-only main-text figures
for bench in BENCHES.keys():
    plot_bench_curves(comp, bench, os.path.join(ART, f"bench_{bench}_v02_lines.png"))

print("Saved comparison_v02.json and lines-only figures in artifacts/")
