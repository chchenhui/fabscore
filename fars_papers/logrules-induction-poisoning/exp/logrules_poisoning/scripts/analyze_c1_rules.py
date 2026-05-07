"""Analyze C1 poisoned rules vs Phase-0v2 clean rules (same induction session).
Generates human-readable rule diffs and flags over-general rules.
Outputs rule_diff.txt per config and results/c1_rule_analysis.csv.
Usage: python scripts/analyze_c1_rules.py
"""

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.diagnostics import print_rule_diff

OVER_GENERAL_PATTERNS = [
    re.compile(r"replace\s+all\s+alphabetic", re.IGNORECASE),
    re.compile(r"replace\s+all\s+words", re.IGNORECASE),
    re.compile(r"replace\s+all\s+non[- ]?numeric", re.IGNORECASE),
    re.compile(r"replace\s+all\s+[a-z]*\s*tokens?\s+with\s+<\*>", re.IGNORECASE),
    re.compile(r"every\s+(word|token|string)\s+(should\s+be|becomes?|is)\s+<\*>", re.IGNORECASE),
    re.compile(r"all\s+(words|tokens|strings)\s+(are|become|should)\s+<\*>", re.IGNORECASE),
    re.compile(r"wildcard\s+everything", re.IGNORECASE),
    re.compile(r"abstract\s+all", re.IGNORECASE),
    re.compile(r"mark\s+(all|every)\s+\w+\s+(as\s+)?(<\*>|wildcard)", re.IGNORECASE),
]


def is_over_general(rule: str) -> bool:
    for pat in OVER_GENERAL_PATTERNS:
        if pat.search(rule):
            return True
    return False


def load_rules(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("ranked_rules", [])


def main():
    config_path = PROJECT_ROOT / "results" / "c1_config.json"
    with open(config_path) as f:
        config = json.load(f)

    payloads = config["qualifying_payloads"]
    k_values = config["k_values"]
    datasets = config["datasets"]
    seeds = config["seeds"]

    rows = []

    for payload in payloads:
        for k in k_values:
            for dataset in datasets:
                for seed in seeds:
                    print(f"Analyzing {payload}/k{k}/{dataset}/seed_{seed}...")

                    c0_rules_path = (
                        PROJECT_ROOT / "outputs" / "rules" / "phase0v2_clean"
                        / dataset / f"seed_{seed}" / "ranked_rules.json"
                    )
                    c1_rules_path = (
                        PROJECT_ROOT / "outputs" / "rules" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}" / "ranked_rules.json"
                    )

                    c0_rules = load_rules(c0_rules_path)
                    c1_rules = load_rules(c1_rules_path)

                    clean_set = set(c0_rules)
                    poisoned_set = set(c1_rules)
                    shared = clean_set & poisoned_set
                    removed = clean_set - poisoned_set
                    added = poisoned_set - clean_set

                    over_general_rules = [r for r in c1_rules if is_over_general(r)]
                    num_over_general = len(over_general_rules)

                    diff_dir = (
                        PROJECT_ROOT / "outputs" / "rules" / "c1_poisoned"
                        / payload / dataset / f"k{k}" / f"seed_{seed}"
                    )
                    diff_dir.mkdir(parents=True, exist_ok=True)
                    diff_path = diff_dir / "rule_diff.txt"

                    lines = []
                    lines.append(f"Rule Diff: C0 (clean) vs C1 (poisoned)")
                    lines.append(f"Config: payload={payload}, k={k}, dataset={dataset}, seed={seed}")
                    lines.append(f"")
                    lines.append(f"Clean rules: {len(c0_rules)}")
                    lines.append(f"Poisoned rules: {len(c1_rules)}")
                    lines.append(f"Shared: {len(shared)}")
                    lines.append(f"Removed (clean only): {len(removed)}")
                    lines.append(f"Added (poisoned only): {len(added)}")
                    lines.append(f"Over-general rules: {num_over_general}")
                    lines.append("")

                    if removed:
                        lines.append("--- Removed (in clean, not in poisoned) ---")
                        for r in sorted(removed):
                            lines.append(f"  - {r}")
                        lines.append("")

                    if added:
                        lines.append("+++ Added (in poisoned, not in clean) +++")
                        for r in sorted(added):
                            flag = " [OVER-GENERAL]" if is_over_general(r) else ""
                            lines.append(f"  + {r}{flag}")
                        lines.append("")

                    if shared:
                        lines.append(f"=== Shared ({len(shared)} rules) ===")
                        for r in sorted(shared):
                            flag = " [OVER-GENERAL]" if is_over_general(r) else ""
                            lines.append(f"  = {r}{flag}")
                        lines.append("")

                    if over_general_rules:
                        lines.append("!!! Over-General Rules Detected !!!")
                        for r in over_general_rules:
                            lines.append(f"  ! {r}")
                        lines.append("")

                    with open(diff_path, "w") as f:
                        f.write("\n".join(lines))

                    rows.append({
                        "payload": payload,
                        "k": k,
                        "dataset": dataset,
                        "seed": seed,
                        "num_clean_rules": len(c0_rules),
                        "num_poisoned_rules": len(c1_rules),
                        "num_shared": len(shared),
                        "num_removed": len(removed),
                        "num_added": len(added),
                        "num_over_general": num_over_general,
                    })

                    status = "FLAGGED" if num_over_general > 0 else "OK"
                    print(f"  clean={len(c0_rules)} poisoned={len(c1_rules)} "
                          f"shared={len(shared)} removed={len(removed)} added={len(added)} "
                          f"over_general={num_over_general} [{status}]")

    csv_path = PROJECT_ROOT / "results" / "c1_rule_analysis.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRule analysis saved to {csv_path}")

    total_over_general = sum(r["num_over_general"] for r in rows)
    configs_with_og = sum(1 for r in rows if r["num_over_general"] > 0)
    print(f"\nSummary: {total_over_general} over-general rules across {configs_with_og}/{len(rows)} configs")


if __name__ == "__main__":
    main()
