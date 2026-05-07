import os, json
from greenhouse.optimizers import baseline_rules
from greenhouse.runners import eval_rule_linreg
from greenhouse.figures import ensure_dir
from greenhouse.dsl import Rule

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "artifacts"); ensure_dir(ART)

best = json.load(open(os.path.join(ART, "best_rule_v02.json")))
best_rule = Rule(**best["best_rule"])

def compare_linreg(steps=300):
    out = {}
    rules = baseline_rules()
    rules["Evolved_v02"] = best_rule
    res = {}
    for name, r in rules.items():
        loss, curves = eval_rule_linreg(r, steps=steps, seeds=(0,1,2))
        res[name] = {"final_loss": loss, "curves": curves.mean(axis=0).tolist()}
    out["linreg"] = res
    return out

comp_lin = compare_linreg(steps=300)
json.dump(comp_lin, open(os.path.join(ART, "comparison_linreg_v02.json"), "w"), indent=2)

print("Saved comparison_linreg_v02.json in artifacts/")
