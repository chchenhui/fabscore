"""Cross-model comparison script for Qwen2.5-7B-Instruct replication.
Reads metric files for Llama-3.1-8B and Qwen2.5-7B, computes deltas,
and outputs qwen_replication.csv."""

import csv
import json
import os

PROJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
OUTPUTS_DIR = os.path.join(PROJ_DIR, "eacp", "outputs")
RESULTS_DIR = os.path.join(PROJ_DIR, "eacp", "results")

METRIC_FILES = {
    ("Llama-3.1-8B-Instruct", "A"): "A_llama31_8b_confiqa_mc_metrics.json",
    ("Llama-3.1-8B-Instruct", "B"): "B_llama31_8b_confiqa_mc_metrics.json",
    ("Llama-3.1-8B-Instruct", "C"): "C_llama31_8b_confiqa_mc_sc_full_metrics.json",
    ("Qwen2.5-7B-Instruct", "A"): "A_qwen25_7b_confiqa_mc_metrics.json",
    ("Qwen2.5-7B-Instruct", "B"): "B_qwen25_7b_confiqa_mc_metrics.json",
    ("Qwen2.5-7B-Instruct", "C"): "C_qwen25_7b_confiqa_mc_sc_metrics.json",
}


def load_metrics():
    all_metrics = {}
    for (model, cond), fname in METRIC_FILES.items():
        path = os.path.join(OUTPUTS_DIR, fname)
        with open(path) as f:
            all_metrics[(model, cond)] = json.load(f)
    return all_metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = load_metrics()

    rows = []
    for model in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]:
        for cond in ["A", "B", "C"]:
            m = metrics[(model, cond)]
            rows.append({
                "Model": model,
                "Condition": cond,
                "Pc": m["Pc"],
                "Po": m["Po"],
                "MR": m["MR"],
                "EM": m["EM"],
            })

    for model in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]:
        b = metrics[(model, "B")]
        c = metrics[(model, "C")]
        rows.append({
            "Model": model,
            "Condition": "C-B delta",
            "Pc": round(c["Pc"] - b["Pc"], 2),
            "Po": round(c["Po"] - b["Po"], 2),
            "MR": round(c["MR"] - b["MR"], 2),
            "EM": round(c["EM"] - b["EM"], 2),
        })

    out_path = os.path.join(RESULTS_DIR, "qwen_replication.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Condition", "Pc", "Po", "MR", "EM"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")

    print("\n=== Cross-Model Comparison (ConFiQA-MC, 6000 examples) ===")
    print(f"{'Model':<28} {'Condition':<12} {'Pc':>8} {'Po':>8} {'MR':>8} {'EM':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['Model']:<28} {r['Condition']:<12} {r['Pc']:>8} {r['Po']:>8} {r['MR']:>8} {r['EM']:>8}")


if __name__ == "__main__":
    main()
