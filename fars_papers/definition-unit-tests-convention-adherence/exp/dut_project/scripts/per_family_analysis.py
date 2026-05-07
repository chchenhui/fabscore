"""Per-family accuracy table, error categorization, and visualization.

Produces:
  results/analysis/per_family_table.json  - per family x model x condition accuracy
  results/analysis/error_breakdown.json   - error types for Condition C (convention/arithmetic/parsing)
  results/figures/per_family_accuracy.png - grouped bar charts
  results/figures/error_breakdown.png     - stacked bar charts
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

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_PATH = os.path.join(PROJ, "dut_project/data/erdos_conventions_bench.jsonl")
FAMILIES = ["asymptotics", "completeness", "convolution"]
CONDITIONS = ["A", "B", "C"]

MODEL_CONFIGS = {
    "qwen25_math_7b": {
        "label": "Qwen2.5-Math-7B-Instruct",
        "output_dir": os.path.join(PROJ, "dut_project/outputs/qwen25_math_7b_v3"),
    },
    "llama31_8b": {
        "label": "Llama-3.1-8B-Instruct",
        "output_dir": os.path.join(PROJ, "dut_project/outputs/llama31_8b"),
    },
}


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


def get_extractor(model_key):
    if model_key == "qwen25_math_7b":
        return extract_qwen
    return extract_answer_robust


def score_all(outputs, bench_map, extractor, condition):
    """Score outputs, returning per-item detail for error analysis."""
    items_detail = []
    for o in outputs:
        item = bench_map[o["item_id"]]
        gt = _normalize_answer(item["ground_truth_glossary"])
        gt_alt = _normalize_answer(item["ground_truth_alternate"])
        ans = extractor(o["raw_output"], family=o["family"])
        ok = (ans == gt) if ans else False
        alt_match = (ans == gt_alt) if ans else False

        detail = {
            "item_id": o["item_id"],
            "family": o["family"],
            "condition": condition,
            "predicted": ans,
            "ground_truth_glossary": gt,
            "ground_truth_alternate": gt_alt,
            "main_correct": ok,
            "alt_match": alt_match,
        }

        if condition in ("B", "C"):
            detail["check_accuracy_note"] = ("N/A: V3 verified-facts design provides check "
                                              "answers as demonstrations, not model-generated")

        items_detail.append(detail)
    return items_detail


def compute_per_family_metrics(items_detail, condition):
    per_fam = {}
    for d in items_detail:
        fam = d["family"]
        if fam not in per_fam:
            per_fam[fam] = {
                "correct": 0, "total": 0, "alt_matches": 0,
            }
        pf = per_fam[fam]
        pf["total"] += 1
        if d["main_correct"]:
            pf["correct"] += 1
        if d["alt_match"]:
            pf["alt_matches"] += 1

    result = {}
    for fam in FAMILIES:
        pf = per_fam.get(fam, {"correct": 0, "total": 0, "alt_matches": 0})
        entry = {
            "main_accuracy": pf["correct"] / pf["total"] if pf["total"] > 0 else 0.0,
            "alt_match_rate": pf["alt_matches"] / pf["total"] if pf["total"] > 0 else 0.0,
            "n": pf["total"],
        }
        if condition in ("B", "C"):
            entry["check_accuracy"] = "N/A (verified-facts design: checks provided as demonstrations)"
        result[fam] = entry
    return result


def compute_error_breakdown(items_detail):
    per_fam = {}
    for d in items_detail:
        fam = d["family"]
        if fam not in per_fam:
            per_fam[fam] = {
                "total": 0, "correct": 0,
                "convention_error": 0, "arithmetic_error": 0, "parsing_failure": 0,
            }
        pf = per_fam[fam]
        pf["total"] += 1
        if d["main_correct"]:
            pf["correct"] += 1
        else:
            if d["predicted"] is None:
                pf["parsing_failure"] += 1
            elif d["alt_match"]:
                pf["convention_error"] += 1
            else:
                pf["arithmetic_error"] += 1

    result = {}
    for fam in FAMILIES:
        pf = per_fam.get(fam, {"total": 0, "correct": 0,
                                 "convention_error": 0, "arithmetic_error": 0,
                                 "parsing_failure": 0})
        t = pf["total"] if pf["total"] > 0 else 1
        result[fam] = {
            "total": pf["total"],
            "correct": pf["correct"],
            "correct_rate": pf["correct"] / t,
            "convention_error": pf["convention_error"],
            "convention_error_rate": pf["convention_error"] / t,
            "arithmetic_error": pf["arithmetic_error"],
            "arithmetic_error_rate": pf["arithmetic_error"] / t,
            "parsing_failure": pf["parsing_failure"],
            "parsing_failure_rate": pf["parsing_failure"] / t,
        }
    return result


def verify_against_existing(per_family_table):
    ref_path = os.path.join(PROJ, "results/effectiveness_evaluation.json")
    with open(ref_path) as f:
        ref = json.load(f)

    model_map = {
        "qwen25_math_7b": "qwen25_math_7b",
        "llama31_8b": "llama31_8b",
    }
    cond_map = {"A": "A", "B": "B", "C": "C"}

    mismatches = []
    for model_key, ref_key in model_map.items():
        ref_fam = ref["consolidated_results"][ref_key]["per_family_accuracy"]
        for fam in FAMILIES:
            for cond, ref_cond in cond_map.items():
                ref_val = ref_fam[fam][ref_cond]
                computed_val = per_family_table[model_key][fam][cond]["main_accuracy"]
                if abs(ref_val - computed_val) > 0.001:
                    mismatches.append(
                        f"{model_key}/{fam}/{cond}: computed={computed_val:.4f} vs ref={ref_val:.4f}"
                    )

    if mismatches:
        print("VERIFICATION FAILED - mismatches found:")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)
    else:
        print("Verification PASSED: all per-family main accuracy values match existing results.")


def plot_per_family_accuracy(per_family_table, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.25
    colors = {"A": "#5B9BD5", "B": "#ED7D31", "C": "#70AD47"}

    for ax_idx, (model_key, model_cfg) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[ax_idx]
        for i, cond in enumerate(CONDITIONS):
            vals = [per_family_table[model_key][fam][cond]["main_accuracy"] * 100
                    for fam in FAMILIES]
            bars = ax.bar(x + i * width, vals, width, label=f"Condition {cond}",
                          color=colors[cond], edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Convention Family")
        ax.set_ylabel("Main Accuracy (%)")
        ax.set_title(model_cfg["label"])
        ax.set_xticks(x + width)
        ax.set_xticklabels([f.capitalize() for f in FAMILIES])
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-family accuracy chart to {out_path}")


def plot_error_breakdown(error_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.5
    colors = {
        "correct": "#70AD47",
        "convention_error": "#ED7D31",
        "arithmetic_error": "#5B9BD5",
        "parsing_failure": "#A5A5A5",
    }
    labels = {
        "correct": "Correct",
        "convention_error": "Convention Error",
        "arithmetic_error": "Arithmetic/Reasoning Error",
        "parsing_failure": "Parsing Failure",
    }

    for ax_idx, (model_key, model_cfg) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[ax_idx]
        bottom = np.zeros(len(FAMILIES))
        for cat_key in ["correct", "convention_error", "arithmetic_error", "parsing_failure"]:
            vals = [error_data[model_key][fam][f"{cat_key}_rate"] * 100 for fam in FAMILIES]
            ax.bar(x, vals, width, bottom=bottom, label=labels[cat_key],
                   color=colors[cat_key], edgecolor="black", linewidth=0.5)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:
                    ax.text(x[j], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold")
            bottom += np.array(vals)

        ax.set_xlabel("Convention Family")
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"{model_cfg['label']} (Condition C)")
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in FAMILIES])
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error breakdown chart to {out_path}")


def main():
    bench_items = load_jsonl(BENCH_PATH)
    bench_map = {item["item_id"]: item for item in bench_items}

    per_family_table = {}
    all_condition_c_details = {}

    for model_key, model_cfg in MODEL_CONFIGS.items():
        extractor = get_extractor(model_key)
        per_family_table[model_key] = {}

        for cond in CONDITIONS:
            output_path = os.path.join(
                model_cfg["output_dir"], f"condition_{cond.lower()}.jsonl"
            )
            outputs = load_jsonl(output_path)
            items_detail = score_all(outputs, bench_map, extractor, cond)
            metrics = compute_per_family_metrics(items_detail, cond)

            for fam in FAMILIES:
                if fam not in per_family_table[model_key]:
                    per_family_table[model_key][fam] = {}
                per_family_table[model_key][fam][cond] = metrics[fam]

            if cond == "C":
                all_condition_c_details[model_key] = items_detail

    print("\n=== Per-Family Accuracy Table ===")
    for model_key in MODEL_CONFIGS:
        print(f"\n{MODEL_CONFIGS[model_key]['label']}:")
        for fam in FAMILIES:
            row = per_family_table[model_key][fam]
            parts = []
            for cond in CONDITIONS:
                acc = row[cond]["main_accuracy"]
                alt = row[cond]["alt_match_rate"]
                chk = row[cond].get("check_accuracy", None)
                s = f"{cond}: {acc:.0%} (alt={alt:.1%}"
                if chk is not None and isinstance(chk, (int, float)):
                    s += f", chk={chk:.0%}"
                elif chk is not None:
                    s += ", chk=N/A"
                s += ")"
                parts.append(s)
            print(f"  {fam}: {', '.join(parts)}")

    print("\n=== Verification ===")
    verify_against_existing(per_family_table)

    analysis_dir = os.path.join(PROJ, "results/analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    table_path = os.path.join(analysis_dir, "per_family_table.json")
    with open(table_path, "w") as f:
        json.dump(per_family_table, f, indent=2)
    print(f"\nSaved per-family table to {table_path}")

    error_data = {}
    for model_key, details in all_condition_c_details.items():
        error_data[model_key] = compute_error_breakdown(details)

    print("\n=== Error Breakdown (Condition C) ===")
    for model_key in MODEL_CONFIGS:
        print(f"\n{MODEL_CONFIGS[model_key]['label']}:")
        for fam in FAMILIES:
            eb = error_data[model_key][fam]
            print(f"  {fam} (n={eb['total']}): correct={eb['correct_rate']:.0%}, "
                  f"convention_err={eb['convention_error_rate']:.1%}, "
                  f"arith_err={eb['arithmetic_error_rate']:.1%}, "
                  f"parse_fail={eb['parsing_failure_rate']:.1%}")

    error_path = os.path.join(analysis_dir, "error_breakdown.json")
    with open(error_path, "w") as f:
        json.dump(error_data, f, indent=2)
    print(f"\nSaved error breakdown to {error_path}")

    fig_dir = os.path.join(PROJ, "results/figures")
    plot_per_family_accuracy(per_family_table, os.path.join(fig_dir, "per_family_accuracy.png"))
    plot_error_breakdown(error_data, os.path.join(fig_dir, "error_breakdown.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
