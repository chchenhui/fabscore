# Ablation: Arelle extracted_value + regex calculated_value.
# Tests whether Arelle's benefit comes from reliable fact extraction or
# from structural recomputation via calculation linkbases. Merges saved
# per-instance results from the Arelle and regex baselines, then evaluates
# the hybrid on the executable subset and compares all three approaches.

import json
from collections import Counter
from pathlib import Path

from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR
from executable_finmr.evaluation.deterministic_judge import evaluate


def _load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main():
    arelle_path = OUTPUT_DIR / "arelle_baseline_results.jsonl"
    regex_path = OUTPUT_DIR / "regex_baseline_results.jsonl"
    exec_ids_path = RESULTS_DIR / "executable_finmr_ids.json"

    arelle_results = _load_jsonl(arelle_path)
    regex_results = _load_jsonl(regex_path)
    with open(exec_ids_path) as f:
        exec_ids = set(json.load(f)["executable_ids"])

    arelle_by_id = {r["id"]: r for r in arelle_results}
    regex_by_id = {r["id"]: r for r in regex_results}

    print(f"Loaded {len(arelle_results)} Arelle results, {len(regex_results)} regex results")
    print(f"Executable subset: {len(exec_ids)} instances")

    hybrid_results = []
    for inst_id in sorted(exec_ids):
        ar = arelle_by_id[inst_id]
        rr = regex_by_id.get(inst_id, {})

        arelle_pred = ar.get("prediction", {})
        regex_pred = rr.get("prediction", {})

        extracted_value = arelle_pred.get("extracted_value")
        calculated_value = regex_pred.get("calculated_value")

        prediction = {}
        if extracted_value is not None:
            prediction["extracted_value"] = extracted_value
            if calculated_value is not None:
                prediction["calculated_value"] = calculated_value

        hybrid_results.append({
            "id": inst_id,
            "dqc_id": ar["dqc_id"],
            "prediction": prediction,
            "gold": ar["gold"],
        })

    hybrid_eval = evaluate(hybrid_results)
    hybrid_metrics = hybrid_eval["metrics"]
    hybrid_per = hybrid_eval["per_instance"]

    print(f"\nHYBRID (Arelle EV + Regex CV) on executable subset (N={len(hybrid_results)}):")
    print(f"  ACC = {hybrid_metrics['ACC']:.4f}  ({hybrid_metrics['N_A']}/{hybrid_metrics['N']})")
    print(f"  SER = {hybrid_metrics['SER']:.4f}  ({hybrid_metrics['N_S']}/{hybrid_metrics['N']})")
    print(f"  EER = {hybrid_metrics['EER']:.4f}  ({hybrid_metrics['N_E']}/{hybrid_metrics['N']})")
    print(f"  CER = {hybrid_metrics['CER']:.4f}  ({hybrid_metrics['N_C']}/{hybrid_metrics['N']})")

    arelle_exec = [r for r in arelle_results if r["id"] in exec_ids]
    arelle_exec_eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in arelle_exec]
    arelle_exec_eval = evaluate(arelle_exec_eval_input)
    arelle_exec_metrics = arelle_exec_eval["metrics"]

    regex_exec = [r for r in regex_results if r["id"] in exec_ids]
    regex_exec_eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in regex_exec]
    regex_exec_eval = evaluate(regex_exec_eval_input)
    regex_exec_metrics = regex_exec_eval["metrics"]

    print(f"\nFULL ARELLE on executable subset (N={arelle_exec_metrics['N']}):")
    print(f"  ACC = {arelle_exec_metrics['ACC']:.4f}  ({arelle_exec_metrics['N_A']}/{arelle_exec_metrics['N']})")
    print(f"  EER = {arelle_exec_metrics['EER']:.4f}  ({arelle_exec_metrics['N_E']}/{arelle_exec_metrics['N']})")
    print(f"  CER = {arelle_exec_metrics['CER']:.4f}  ({arelle_exec_metrics['N_C']}/{arelle_exec_metrics['N']})")

    print(f"\nREGEX BASELINE on executable subset (N={regex_exec_metrics['N']}):")
    print(f"  ACC = {regex_exec_metrics['ACC']:.4f}  ({regex_exec_metrics['N_A']}/{regex_exec_metrics['N']})")
    print(f"  SER = {regex_exec_metrics['SER']:.4f}  ({regex_exec_metrics['N_S']}/{regex_exec_metrics['N']})")
    print(f"  EER = {regex_exec_metrics['EER']:.4f}  ({regex_exec_metrics['N_E']}/{regex_exec_metrics['N']})")
    print(f"  CER = {regex_exec_metrics['CER']:.4f}  ({regex_exec_metrics['N_C']}/{regex_exec_metrics['N']})")

    per_dqc_hybrid = {}
    for r in hybrid_per:
        dqc = r["dqc_id"]
        if dqc not in per_dqc_hybrid:
            per_dqc_hybrid[dqc] = {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0}
        per_dqc_hybrid[dqc][r["label"]] += 1
        per_dqc_hybrid[dqc]["total"] += 1

    per_dqc_arelle = {}
    for r in arelle_exec_eval["per_instance"]:
        dqc = r["dqc_id"] if "dqc_id" in r else arelle_by_id[r["id"]]["dqc_id"]
        if dqc not in per_dqc_arelle:
            per_dqc_arelle[dqc] = {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0}
        per_dqc_arelle[dqc][r["label"]] += 1
        per_dqc_arelle[dqc]["total"] += 1

    per_dqc_regex = {}
    for r in regex_exec_eval["per_instance"]:
        dqc = r["dqc_id"] if "dqc_id" in r else regex_by_id[r["id"]]["dqc_id"]
        if dqc not in per_dqc_regex:
            per_dqc_regex[dqc] = {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0}
        per_dqc_regex[dqc][r["label"]] += 1
        per_dqc_regex[dqc]["total"] += 1

    print(f"\n{'='*70}")
    print("THREE-WAY COMPARISON ON EXECUTABLE SUBSET")
    print(f"{'='*70}")
    header = f"{'Method':<30} {'ACC':>7} {'SER':>7} {'EER':>7} {'CER':>7} {'N':>5}"
    print(header)
    print("-" * len(header))

    def _row(name, m):
        print(f"{name:<30} {m['ACC']:>7.4f} {m['SER']:>7.4f} {m['EER']:>7.4f} {m['CER']:>7.4f} {m['N']:>5}")

    _row("Full Arelle", arelle_exec_metrics)
    _row("Arelle EV + Regex CV", hybrid_metrics)
    _row("Regex (both)", regex_exec_metrics)

    for dqc in sorted(set(list(per_dqc_hybrid.keys()) + list(per_dqc_arelle.keys()) + list(per_dqc_regex.keys()))):
        print(f"\n  --- {dqc} ---")
        for label, data in [("Full Arelle", per_dqc_arelle), ("Arelle+MsgCalc", per_dqc_hybrid), ("Regex", per_dqc_regex)]:
            d = data.get(dqc, {"A": 0, "S": 0, "E": 0, "C": 0, "total": 0})
            n = d["total"] if d["total"] > 0 else 1
            print(f"    {label:<20} ACC={d['A']/n:.4f}  EER={d['E']/n:.4f}  CER={d['C']/n:.4f}  N={d['total']}")

    output_path = OUTPUT_DIR / "ablation_message_calc_results.jsonl"
    with open(output_path, "w") as f:
        for r in hybrid_per:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-instance results: {output_path}")

    def _per_dqc_dict(data):
        out = {}
        for dqc, d in sorted(data.items()):
            n = d["total"]
            out[dqc] = {
                "N": n,
                "ACC": round(d["A"] / n, 4) if n else 0,
                "SER": round(d["S"] / n, 4) if n else 0,
                "EER": round(d["E"] / n, 4) if n else 0,
                "CER": round(d["C"] / n, 4) if n else 0,
            }
        return out

    comparison = {
        "ablation": "message_calc_vs_structural_recomputation",
        "description": "Arelle for extracted_value, regex for calculated_value -- tests whether Arelle benefit is from fact extraction or structural recomputation",
        "executable_subset_N": len(exec_ids),
        "full_arelle": {
            "method": "Arelle for both extracted_value and calculated_value",
            "aggregate": {k: round(v, 4) for k, v in arelle_exec_metrics.items() if k in ("ACC", "SER", "EER", "CER")},
            "counts": {k: v for k, v in arelle_exec_metrics.items() if k.startswith("N")},
            "per_dqc": _per_dqc_dict(per_dqc_arelle),
        },
        "arelle_ev_regex_cv": {
            "method": "Arelle for extracted_value, regex for calculated_value",
            "aggregate": {k: round(v, 4) for k, v in hybrid_metrics.items() if k in ("ACC", "SER", "EER", "CER")},
            "counts": {k: v for k, v in hybrid_metrics.items() if k.startswith("N")},
            "per_dqc": _per_dqc_dict(per_dqc_hybrid),
        },
        "regex_both": {
            "method": "Regex for both extracted_value and calculated_value",
            "aggregate": {k: round(v, 4) for k, v in regex_exec_metrics.items() if k in ("ACC", "SER", "EER", "CER")},
            "counts": {k: v for k, v in regex_exec_metrics.items() if k.startswith("N")},
            "per_dqc": _per_dqc_dict(per_dqc_regex),
        },
    }

    results_path = RESULTS_DIR / "ablation_message_calc.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved: {results_path}")


if __name__ == "__main__":
    main()
