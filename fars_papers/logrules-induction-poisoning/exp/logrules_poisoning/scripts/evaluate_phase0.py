"""Evaluate Phase-0 proceed criterion for BGL seed=0.
Computes template disagreement and PA drop for each poisoned config vs clean.
Outputs results/phase0_summary.csv and prints rule diffs.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.metrics import compute_pa, normalize_template
from logrules_poisoning.src.evaluation.diagnostics import template_disagreement, print_rule_diff

DATASET = "BGL"
SEED = 0


def load_predictions(pred_path):
    records = []
    with open(pred_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def load_rules(rules_path):
    with open(rules_path) as f:
        data = json.load(f)
    return data.get("ranked_rules", [])


def main():
    clean_variance_path = PROJECT_ROOT / "results" / "clean_variance.json"
    with open(clean_variance_path) as f:
        variance = json.load(f)
    threshold_x = variance[DATASET]["threshold_X"]
    print(f"BGL threshold_X = {threshold_x:.6f}")

    c0_pred_path = PROJECT_ROOT / "outputs" / "predictions" / "c0_clean" / DATASET / f"seed_{SEED}" / "canary_predictions.jsonl"
    c0_records = load_predictions(c0_pred_path)
    c0_templates = [r["predicted_template"] for r in c0_records]
    c0_gts = [r["template"] for r in c0_records]
    c0_pa = compute_pa(c0_templates, c0_gts)
    print(f"C0 canary PA = {c0_pa:.4f}")

    c0_rules = load_rules(
        PROJECT_ROOT / "outputs" / "rules" / "c0_clean" / DATASET / f"seed_{SEED}" / "ranked_rules.json"
    )

    configs = []
    for payload in ["A", "B", "C"]:
        for k in [1, 3]:
            configs.append((payload, k))

    rows = []

    for payload, k in configs:
        config_name = f"{payload}_k{k}"
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        c1_pred_path = (
            PROJECT_ROOT / "outputs" / "predictions" / "phase0" / config_name
            / DATASET / f"seed_{SEED}" / "canary_predictions.jsonl"
        )
        c1_records = load_predictions(c1_pred_path)
        c1_templates = [r["predicted_template"] for r in c1_records]
        c1_gts = [r["template"] for r in c1_records]
        c1_pa = compute_pa(c1_templates, c1_gts)

        td = template_disagreement(c0_templates, c1_templates)
        pa_drop = c0_pa - c1_pa

        c1_rules = load_rules(
            PROJECT_ROOT / "outputs" / "rules" / f"phase0/{payload}_k{k}" / DATASET / f"seed_{SEED}" / "ranked_rules.json"
        )
        raw_rules_path = PROJECT_ROOT / "outputs" / "rules" / f"phase0/{payload}_k{k}" / DATASET / f"seed_{SEED}" / "raw_rules.json"
        with open(raw_rules_path) as f:
            raw_data = json.load(f)
        model_used = raw_data.get("model_used", "gpt-4o-mini")

        passes_td = td >= 0.10
        passes_pa = pa_drop >= threshold_x
        passes = passes_td and passes_pa

        print(f"  C0 canary PA:  {c0_pa:.4f}")
        print(f"  C1 canary PA:  {c1_pa:.4f}")
        print(f"  PA drop:       {pa_drop:.4f}  (threshold: {threshold_x:.4f})  {'PASS' if passes_pa else 'FAIL'}")
        print(f"  Template disagreement: {td:.4f}  (threshold: 0.10)  {'PASS' if passes_td else 'FAIL'}")
        print(f"  Model used:    {model_used}")
        print(f"  Overall:       {'PASS' if passes else 'FAIL'}")

        print(f"\n  Rule diff:")
        print_rule_diff(c0_rules, c1_rules)

        rows.append({
            "payload": payload,
            "k": k,
            "canary_PA_C0": round(c0_pa, 4),
            "canary_PA_C1": round(c1_pa, 4),
            "PA_drop": round(pa_drop, 4),
            "template_disagreement": round(td, 4),
            "model_used": model_used,
            "passes_criterion": passes,
        })

    csv_path = PROJECT_ROOT / "results" / "phase0_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPhase-0 summary saved to {csv_path}")

    any_pass = any(r["passes_criterion"] for r in rows)
    print(f"\n{'='*60}")
    if any_pass:
        passing = [f"{r['payload']}_k{r['k']}" for r in rows if r["passes_criterion"]]
        print(f"PROCEED: {len(passing)} config(s) pass criteria: {', '.join(passing)}")
    else:
        print("NO CONFIG PASSES. Induction-stage poisoning is NOT effective under this threat model.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
