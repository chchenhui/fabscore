# Per-DQC rule family breakdown analysis.
# Loads Arelle and Regex per-instance results, groups by DQC rule family,
# computes executability, ACC/SER/EER/CER for full and exec subsets,
# failure taxonomy, and generates comparison visualizations.

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RULE_NAMES = {
    "0015": "DQC_0015 (sign/negativity)",
    "0117": "DQC_0117 (dimensional cross-check)",
    "0126": "DQC_0126 (calc consistency)",
}
RULE_ORDER = ["0015", "0117", "0126"]
LABEL_ORDER = ["A", "S", "E", "C"]
LABEL_COLORS = {"A": "#2ca02c", "S": "#d62728", "E": "#ff7f0e", "C": "#1f77b4"}
FAILURE_CATS = ["external_dependency", "missing_dts_artifact", "malformed_xml", "unknown"]
FAILURE_COLORS = {
    "external_dependency": "#e377c2",
    "missing_dts_artifact": "#7f7f7f",
    "malformed_xml": "#bcbd22",
    "unknown": "#17becf",
}


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _extract_family(dqc_id: str) -> str:
    return dqc_id.replace("DQC_US_", "")


def _label_counts(labels):
    c = Counter(labels)
    return {l: c.get(l, 0) for l in LABEL_ORDER}


def _rates(label_counts, n):
    if n == 0:
        return {"ACC": 0.0, "SER": 0.0, "EER": 0.0, "CER": 0.0}
    return {
        "ACC": round(label_counts["A"] / n, 4),
        "SER": round(label_counts["S"] / n, 4),
        "EER": round(label_counts["E"] / n, 4),
        "CER": round(label_counts["C"] / n, 4),
    }


def compute_breakdown():
    arelle_rows = _load_jsonl(OUTPUT_DIR / "arelle_baseline_results.jsonl")
    regex_rows = _load_jsonl(OUTPUT_DIR / "regex_baseline_results.jsonl")

    regex_by_id = {r["id"]: r for r in regex_rows}

    arelle_exec_ids = {r["id"] for r in arelle_rows if r.get("executable")}

    grouped = defaultdict(list)
    for ar in arelle_rows:
        family = ar.get("dqc_rule_family") or _extract_family(ar["dqc_id"])
        rr = regex_by_id.get(ar["id"], {})
        grouped[family].append({"arelle": ar, "regex": rr})

    results = {}
    for family in RULE_ORDER:
        items = grouped[family]
        n = len(items)

        arelle_labels = [it["arelle"].get("judge_label", "S") for it in items]
        arelle_lc = _label_counts(arelle_labels)

        exec_items = [it for it in items if it["arelle"].get("executable")]
        n_exec = len(exec_items)
        arelle_exec_labels = [it["arelle"].get("judge_label", "S") for it in exec_items]
        arelle_exec_lc = _label_counts(arelle_exec_labels)

        regex_labels = [it["regex"].get("label", "S") for it in items]
        regex_lc = _label_counts(regex_labels)

        regex_exec_labels = [it["regex"].get("label", "S") for it in exec_items]
        regex_exec_lc = _label_counts(regex_exec_labels)

        non_exec = [it for it in items if not it["arelle"].get("executable")]
        failure_counts = Counter(it["arelle"].get("failure_label", "unknown") for it in non_exec)
        failure_dist = {cat: failure_counts.get(cat, 0) for cat in FAILURE_CATS}

        results[family] = {
            "rule_name": RULE_NAMES[family],
            "N": n,
            "N_executable": n_exec,
            "executability_pct": round(n_exec / n * 100, 2) if n else 0,
            "arelle_full": {**_rates(arelle_lc, n), **{f"N_{l}": arelle_lc[l] for l in LABEL_ORDER}},
            "arelle_exec": {**_rates(arelle_exec_lc, n_exec), **{f"N_{l}": arelle_exec_lc[l] for l in LABEL_ORDER}},
            "regex_full": {**_rates(regex_lc, n), **{f"N_{l}": regex_lc[l] for l in LABEL_ORDER}},
            "regex_exec": {**_rates(regex_exec_lc, n_exec), **{f"N_{l}": regex_exec_lc[l] for l in LABEL_ORDER}},
            "failure_taxonomy": failure_dist,
        }

    return results


def make_stacked_bar(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Judge Label Distribution by DQC Rule Family", fontsize=14, fontweight="bold")

    for idx, family in enumerate(RULE_ORDER):
        ax = axes[idx]
        data = results[family]
        n = data["N"]

        arelle_fracs = [data["arelle_full"][f"N_{l}"] / n for l in LABEL_ORDER]
        regex_fracs = [data["regex_full"][f"N_{l}"] / n for l in LABEL_ORDER]

        x = np.array([0, 1])
        width = 0.5
        bottoms_a = np.zeros(1)
        bottoms_r = np.zeros(1)

        for i, l in enumerate(LABEL_ORDER):
            ax.bar(0, arelle_fracs[i], width, bottom=bottoms_a[0], color=LABEL_COLORS[l], edgecolor="white", linewidth=0.5)
            ax.bar(1, regex_fracs[i], width, bottom=bottoms_r[0], color=LABEL_COLORS[l], edgecolor="white", linewidth=0.5)
            bottoms_a[0] += arelle_fracs[i]
            bottoms_r[0] += regex_fracs[i]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Arelle", "Regex"], fontsize=10)
        ax.set_title(f"DQC_{family}\n(N={n})", fontsize=11)
        ax.set_ylim(0, 1.05)
        if idx == 0:
            ax.set_ylabel("Proportion", fontsize=11)

    patches = [mpatches.Patch(color=LABEL_COLORS[l], label=f"{l} ({'Accurate' if l=='A' else 'Struct Err' if l=='S' else 'Extract Err' if l=='E' else 'Calc Err'})") for l in LABEL_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = FIGURES_DIR / "dqc_rule_judge_labels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def make_executability_pies(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Executability by DQC Rule Family", fontsize=14, fontweight="bold")

    for idx, family in enumerate(RULE_ORDER):
        ax = axes[idx]
        data = results[family]
        n_exec = data["N_executable"]
        ft = data["failure_taxonomy"]
        n_non_exec = sum(ft.values())

        sizes = [n_exec]
        labels_list = [f"Executable\n({n_exec})"]
        colors = ["#2ca02c"]

        for cat in FAILURE_CATS:
            if ft[cat] > 0:
                sizes.append(ft[cat])
                nice = cat.replace("_", " ").title()
                labels_list.append(f"{nice}\n({ft[cat]})")
                colors.append(FAILURE_COLORS[cat])

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_list, autopct="%1.0f%%",
            colors=colors, startangle=90, pctdistance=0.75,
            textprops={"fontsize": 8},
        )
        for t in autotexts:
            t.set_fontsize(7)
        ax.set_title(f"DQC_{family}\n(N={data['N']})", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIGURES_DIR / "dqc_rule_executability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    results = compute_breakdown()

    out_path = RESULTS_DIR / "per_dqc_rule_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    print("\n=== Per-DQC Rule Family Breakdown ===\n")
    header = f"{'Rule':<12} {'N':>4} {'Exec':>5} {'Exec%':>6} | {'Ar-ACC':>7} {'Ar-SER':>7} {'Ar-EER':>7} {'Ar-CER':>7} | {'Rx-ACC':>7} {'Rx-SER':>7} {'Rx-EER':>7} {'Rx-CER':>7}"
    print(header)
    print("-" * len(header))
    for family in RULE_ORDER:
        d = results[family]
        af = d["arelle_full"]
        rf = d["regex_full"]
        print(f"DQC_{family:<7} {d['N']:>4} {d['N_executable']:>5} {d['executability_pct']:>5.1f}% | "
              f"{af['ACC']:>6.1%} {af['SER']:>6.1%} {af['EER']:>6.1%} {af['CER']:>6.1%} | "
              f"{rf['ACC']:>6.1%} {rf['SER']:>6.1%} {rf['EER']:>6.1%} {rf['CER']:>6.1%}")

    print("\n--- Executable Subset ---\n")
    header2 = f"{'Rule':<12} {'N_ex':>5} | {'Ar-ACC':>7} {'Ar-EER':>7} {'Ar-CER':>7} | {'Rx-ACC':>7} {'Rx-EER':>7} {'Rx-CER':>7}"
    print(header2)
    print("-" * len(header2))
    for family in RULE_ORDER:
        d = results[family]
        ae = d["arelle_exec"]
        re_ = d["regex_exec"]
        print(f"DQC_{family:<7} {d['N_executable']:>5} | "
              f"{ae['ACC']:>6.1%} {ae['EER']:>6.1%} {ae['CER']:>6.1%} | "
              f"{re_['ACC']:>6.1%} {re_['EER']:>6.1%} {re_['CER']:>6.1%}")

    print("\n--- Failure Taxonomy (non-executable instances) ---\n")
    for family in RULE_ORDER:
        d = results[family]
        ft = d["failure_taxonomy"]
        total = sum(ft.values())
        print(f"DQC_{family} (N_fail={total}):")
        for cat in FAILURE_CATS:
            if ft[cat] > 0:
                print(f"  {cat}: {ft[cat]} ({ft[cat]/total*100:.1f}%)")

    make_stacked_bar(results)
    make_executability_pies(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
