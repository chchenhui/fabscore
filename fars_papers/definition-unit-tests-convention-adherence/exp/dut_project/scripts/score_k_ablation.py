"""K-ablation analysis: compare k=1 vs k=3 discriminative checks on both models.
Loads A, B, C(k=3), C(k=1) outputs for Qwen and Llama, scores them,
computes paired bootstrap CI for (k=3 - k=1), and generates a bar chart.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dut_project.inference.parse_outputs import (
    _normalize_answer,
    extract_answer_robust,
    extract_boxed_answer,
)
from dut_project.evaluation.statistics import paired_bootstrap_ci

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_PATH = os.path.join(PROJ, "dut_project/data/erdos_conventions_bench.jsonl")
FAMILIES = ["asymptotics", "completeness", "convolution"]


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def extract_qwen(raw_output, family=None):
    text = raw_output.strip()
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    first_line = text.split("\n")[0].strip()
    return _normalize_answer(first_line)


def score_outputs(outputs, bench_map, extractor):
    correct_map = {}
    per_fam = {}
    alt_count = 0
    total = 0

    for o in outputs:
        item = bench_map[o["item_id"]]
        gt = _normalize_answer(item["ground_truth_glossary"])
        gt_alt = _normalize_answer(item["ground_truth_alternate"])
        ans = extractor(o["raw_output"], family=o["family"])
        ok = (ans == gt) if ans else False
        correct_map[o["item_id"]] = ok
        if ans and ans == gt_alt:
            alt_count += 1
        total += 1

        fam = o["family"]
        if fam not in per_fam:
            per_fam[fam] = {"correct": 0, "total": 0}
        per_fam[fam]["total"] += 1
        if ok:
            per_fam[fam]["correct"] += 1

    overall_correct = sum(pf["correct"] for pf in per_fam.values())
    return {
        "overall_accuracy": overall_correct / total if total > 0 else 0.0,
        "overall_correct": overall_correct,
        "total": total,
        "per_family": {
            fam: {
                "accuracy": pf["correct"] / pf["total"] if pf["total"] > 0 else 0.0,
                "correct": pf["correct"],
                "total": pf["total"],
            }
            for fam, pf in per_fam.items()
        },
        "alt_rate": alt_count / total if total > 0 else 0.0,
        "correct_map": correct_map,
    }


def score_model(model_key, extractor, baseline_dir=None, k1_dir=None):
    bench_items = load_jsonl(BENCH_PATH)
    bench_map = {item["item_id"]: item for item in bench_items}
    if baseline_dir is None:
        baseline_dir = os.path.join(PROJ, f"dut_project/outputs/{model_key}")
    if k1_dir is None:
        k1_dir = baseline_dir

    conditions = {}
    for cond_name, dirname, fname in [
        ("A", baseline_dir, "condition_a.jsonl"),
        ("B", baseline_dir, "condition_b.jsonl"),
        ("C_k3", baseline_dir, "condition_c.jsonl"),
        ("C_k1", k1_dir, "condition_c_k1.jsonl"),
    ]:
        path = os.path.join(dirname, fname)
        outputs = load_jsonl(path)
        conditions[cond_name] = score_outputs(outputs, bench_map, extractor)

    all_ids = sorted(
        set(conditions["A"]["correct_map"].keys())
        & set(conditions["B"]["correct_map"].keys())
        & set(conditions["C_k3"]["correct_map"].keys())
        & set(conditions["C_k1"]["correct_map"].keys())
    )

    ci_k3_minus_k1 = paired_bootstrap_ci(
        [conditions["C_k3"]["correct_map"][i] for i in all_ids],
        [conditions["C_k1"]["correct_map"][i] for i in all_ids],
        n_resamples=10000, ci_level=0.95, seed=42,
    )

    ci_k1_minus_b = paired_bootstrap_ci(
        [conditions["C_k1"]["correct_map"][i] for i in all_ids],
        [conditions["B"]["correct_map"][i] for i in all_ids],
        n_resamples=10000, ci_level=0.95, seed=42,
    )

    ci_k1_minus_a = paired_bootstrap_ci(
        [conditions["C_k1"]["correct_map"][i] for i in all_ids],
        [conditions["A"]["correct_map"][i] for i in all_ids],
        n_resamples=10000, ci_level=0.95, seed=42,
    )

    return {
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "correct_map"} for k, v in conditions.items()},
        "bootstrap_ci_k3_minus_k1": {
            "observed_diff": ci_k3_minus_k1["observed_diff"],
            "ci_lower": ci_k3_minus_k1["ci_lower"],
            "ci_upper": ci_k3_minus_k1["ci_upper"],
            "excludes_zero": ci_k3_minus_k1["excludes_zero"],
            "n_resamples": ci_k3_minus_k1["n_resamples"],
        },
        "bootstrap_ci_k1_minus_b": {
            "observed_diff": ci_k1_minus_b["observed_diff"],
            "ci_lower": ci_k1_minus_b["ci_lower"],
            "ci_upper": ci_k1_minus_b["ci_upper"],
            "excludes_zero": ci_k1_minus_b["excludes_zero"],
            "n_resamples": ci_k1_minus_b["n_resamples"],
        },
        "bootstrap_ci_k1_minus_a": {
            "observed_diff": ci_k1_minus_a["observed_diff"],
            "ci_lower": ci_k1_minus_a["ci_lower"],
            "ci_upper": ci_k1_minus_a["ci_upper"],
            "excludes_zero": ci_k1_minus_a["excludes_zero"],
            "n_resamples": ci_k1_minus_a["n_resamples"],
        },
        "raw_correct_maps": {k: v["correct_map"] for k, v in conditions.items()},
    }


def make_bar_chart(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    conditions_order = ["A", "B", "C_k1", "C_k3"]
    labels = ["A\n(Glossary)", "B\n(Neutral k=3)", "C\n(DUT k=1)", "C\n(DUT k=3)"]
    colors = ["#7fb3d8", "#aad4aa", "#f7b267", "#e85d75"]

    for ax, (model_key, model_label) in zip(axes, [
        ("qwen25_math_7b", "Qwen2.5-Math-7B-Instruct"),
        ("llama31_8b", "Llama-3.1-8B-Instruct"),
    ]):
        model_data = results[model_key]
        accs = [model_data["conditions"][c]["overall_accuracy"] * 100 for c in conditions_order]
        bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Main Accuracy (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("K-Ablation: Effect of Number of Discriminative Checks", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    print("=== K-ABLATION ANALYSIS ===\n")

    qwen_extractor = lambda raw, family=None: extract_qwen(raw, family)
    llama_extractor = lambda raw, family=None: extract_answer_robust(raw, family=family)

    model_results = {}
    for model_key, model_name, extractor, bdir, k1dir in [
        ("qwen25_math_7b", "Qwen/Qwen2.5-Math-7B-Instruct", qwen_extractor,
         os.path.join(PROJ, "dut_project/outputs/qwen25_math_7b_v3"),
         os.path.join(PROJ, "dut_project/outputs/qwen25_math_7b")),
        ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", llama_extractor,
         os.path.join(PROJ, "dut_project/outputs/llama31_8b"),
         os.path.join(PROJ, "dut_project/outputs/llama31_8b")),
    ]:
        print(f"--- {model_name} ---")
        result = score_model(model_key, extractor, baseline_dir=bdir, k1_dir=k1dir)
        model_results[model_key] = result

        for cond_label, cond_key in [("A", "A"), ("B (neutral k=3)", "B"), ("C (DUT k=1)", "C_k1"), ("C (DUT k=3)", "C_k3")]:
            r = result["conditions"][cond_key]
            print(f"  {cond_label}: {r['overall_accuracy']:.1%} ({r['overall_correct']}/{r['total']}), alt_rate={r['alt_rate']:.1%}")
            for fam in FAMILIES:
                if fam in r["per_family"]:
                    pf = r["per_family"][fam]
                    print(f"    {fam}: {pf['accuracy']:.0%} ({pf['correct']}/{pf['total']})")

        ci = result["bootstrap_ci_k3_minus_k1"]
        print(f"  k3-k1: {ci['observed_diff']:+.1%}, CI [{ci['ci_lower']:+.1%}, {ci['ci_upper']:+.1%}], excludes_zero={ci['excludes_zero']}")
        ci1b = result["bootstrap_ci_k1_minus_b"]
        print(f"  k1-B: {ci1b['observed_diff']:+.1%}, CI [{ci1b['ci_lower']:+.1%}, {ci1b['ci_upper']:+.1%}], excludes_zero={ci1b['excludes_zero']}")
        ci1a = result["bootstrap_ci_k1_minus_a"]
        print(f"  k1-A: {ci1a['observed_diff']:+.1%}, CI [{ci1a['ci_lower']:+.1%}, {ci1a['ci_upper']:+.1%}], excludes_zero={ci1a['excludes_zero']}")
        print()

    output_json = {
        "analysis": "K-ablation: k=1 vs k=3 discriminative checks",
        "models": {}
    }
    for model_key, model_name in [
        ("qwen25_math_7b", "Qwen/Qwen2.5-Math-7B-Instruct"),
        ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ]:
        r = model_results[model_key]
        conds = r["conditions"]
        output_json["models"][model_key] = {
            "model": model_name,
            "overall_accuracy": {
                "A": conds["A"]["overall_accuracy"],
                "B_neutral_k3": conds["B"]["overall_accuracy"],
                "C_k1": conds["C_k1"]["overall_accuracy"],
                "C_k3": conds["C_k3"]["overall_accuracy"],
            },
            "per_family_accuracy": {
                fam: {
                    "A": conds["A"]["per_family"].get(fam, {}).get("accuracy"),
                    "B_neutral_k3": conds["B"]["per_family"].get(fam, {}).get("accuracy"),
                    "C_k1": conds["C_k1"]["per_family"].get(fam, {}).get("accuracy"),
                    "C_k3": conds["C_k3"]["per_family"].get(fam, {}).get("accuracy"),
                }
                for fam in FAMILIES
            },
            "alternate_convention_rate": {
                "A": conds["A"]["alt_rate"],
                "B_neutral_k3": conds["B"]["alt_rate"],
                "C_k1": conds["C_k1"]["alt_rate"],
                "C_k3": conds["C_k3"]["alt_rate"],
            },
            "bootstrap_ci_k3_minus_k1": r["bootstrap_ci_k3_minus_k1"],
            "bootstrap_ci_k1_minus_b": r["bootstrap_ci_k1_minus_b"],
            "bootstrap_ci_k1_minus_a": r["bootstrap_ci_k1_minus_a"],
        }

    json_path = os.path.join(PROJ, "results/analysis/k_ablation.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Results saved to {json_path}")

    chart_path = os.path.join(PROJ, "results/figures/k_ablation.png")
    make_bar_chart(model_results, chart_path)


if __name__ == "__main__":
    main()
