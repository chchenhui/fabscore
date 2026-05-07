# Main pipeline runner for the Arelle symbolic baseline on all 332 FinMR instances.
# Reconstructs XBRL packages, loads with Arelle, executes DQC rules, evaluates,
# and produces executability audit + metrics.

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

from executable_finmr.configs.settings import OUTPUT_DIR, RESULTS_DIR
from executable_finmr.data.load_finmr import load_finmr
from executable_finmr.data.target_extractor import extract_target
from executable_finmr.data.xbrl_reconstructor import reconstruct_xbrl_package
from executable_finmr.engine.arelle_runner import load_and_analyze
from executable_finmr.engine.failure_classifier import classify_failure
from executable_finmr.evaluation.deterministic_judge import evaluate
from executable_finmr.rules import dqc_0015, dqc_0117, dqc_0126

RULE_EXECUTORS = {
    "0015": dqc_0015.execute,
    "0126": dqc_0126.execute,
    "0117": dqc_0117.execute,
}

XBRL_PKG_DIR = OUTPUT_DIR / "xbrl_packages"


def run_single(inst, pkg_dir: str) -> dict:
    target = extract_target(inst)
    pkg_info = reconstruct_xbrl_package(inst, pkg_dir)

    if pkg_info["errors"]:
        return {
            "id": inst.id,
            "dqc_id": inst.dqc_id,
            "dqc_rule_family": target.dqc_rule_family,
            "executable": False,
            "failure_label": "missing_dts_artifact",
            "prediction": {},
            "gold": inst.gold_answer,
            "error_log": str(pkg_info["errors"]),
        }

    arelle_result = load_and_analyze(pkg_info["instance_path"], target)

    executor = RULE_EXECUTORS.get(target.dqc_rule_family)
    rule_result = None

    if arelle_result.executable and executor:
        try:
            rule_result = executor(arelle_result, target)
        except Exception as e:
            rule_result = {"success": False, "reason": f"exception:{e}"}
    elif arelle_result.executable and not executor:
        rule_result = {"success": False, "reason": f"unknown_rule:{target.dqc_rule_family}"}

    prediction = {}
    executable = False
    failure_label = None

    if rule_result and rule_result.get("success"):
        executable = True
        prediction = {
            "extracted_value": rule_result["extracted_value"],
            "calculated_value": rule_result["calculated_value"],
        }
    else:
        executable = False
        failure_label = classify_failure(
            arelle_result.error_log,
            rule_result,
        )

    arelle_result.close()

    return {
        "id": inst.id,
        "dqc_id": inst.dqc_id,
        "dqc_rule_family": target.dqc_rule_family,
        "executable": executable,
        "failure_label": failure_label,
        "prediction": prediction,
        "gold": inst.gold_answer,
        "error_log": arelle_result.error_log[:500] if arelle_result.error_log else "",
        "n_facts": arelle_result.n_facts,
        "n_concepts": arelle_result.n_concepts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", type=int, default=0, help="Run on N instances only")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated instance IDs")
    args = parser.parse_args()

    print("Loading FinMR dataset...")
    instances = load_finmr()
    print(f"Loaded {len(instances)} instances")

    if args.ids:
        target_ids = set(int(x) for x in args.ids.split(","))
        instances = [i for i in instances if i.id in target_ids]
        print(f"Filtered to {len(instances)} instances by ID")
    elif args.sanity > 0:
        instances = instances[: args.sanity]
        print(f"Sanity mode: running on {len(instances)} instances")

    dqc_counts = Counter(i.dqc_id for i in instances)
    print(f"DQC distribution: {dict(sorted(dqc_counts.items()))}")

    XBRL_PKG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx, inst in enumerate(instances):
        pkg_dir = str(XBRL_PKG_DIR / str(inst.id))
        try:
            result = run_single(inst, pkg_dir)
        except Exception as e:
            print(f"  ERROR on instance {inst.id}: {e}")
            traceback.print_exc()
            result = {
                "id": inst.id,
                "dqc_id": inst.dqc_id,
                "dqc_rule_family": inst.dqc_rule_family,
                "executable": False,
                "failure_label": "unknown",
                "prediction": {},
                "gold": inst.gold_answer,
                "error_log": str(e)[:500],
            }
        all_results.append(result)

        status = "OK" if result["executable"] else f"FAIL({result.get('failure_label', '?')})"
        if idx % 20 == 0 or idx == len(instances) - 1:
            n_exec = sum(1 for r in all_results if r["executable"])
            print(f"  [{idx+1}/{len(instances)}] id={inst.id} {inst.dqc_id} -> {status}  (exec so far: {n_exec}/{len(all_results)})")

    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in all_results]
    eval_output = evaluate(eval_input)
    metrics = eval_output["metrics"]
    per_instance_eval = eval_output["per_instance"]

    for r, ev in zip(all_results, per_instance_eval):
        r["judge_label"] = ev["label"]

    n_exec = sum(1 for r in all_results if r["executable"])
    n_total = len(all_results)
    executability = n_exec / n_total if n_total else 0

    print(f"\nFULL-SET METRICS (N={n_total}):")
    print(f"  ACC = {metrics['ACC']:.4f}  ({metrics['N_A']}/{n_total})")
    print(f"  SER = {metrics['SER']:.4f}  ({metrics['N_S']}/{n_total})")
    print(f"  EER = {metrics['EER']:.4f}  ({metrics['N_E']}/{n_total})")
    print(f"  CER = {metrics['CER']:.4f}  ({metrics['N_C']}/{n_total})")

    exec_results = [r for r in all_results if r["executable"]]
    if exec_results:
        exec_eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in exec_results]
        exec_eval_output = evaluate(exec_eval_input)
        exec_metrics = exec_eval_output["metrics"]
        print(f"\nEXECUTABLE-SUBSET METRICS (N={n_exec}):")
        print(f"  ACC = {exec_metrics['ACC']:.4f}  ({exec_metrics['N_A']}/{n_exec})")
        print(f"  SER = {exec_metrics['SER']:.4f}  ({exec_metrics['N_S']}/{n_exec})")
        print(f"  EER = {exec_metrics['EER']:.4f}  ({exec_metrics['N_E']}/{n_exec})")
        print(f"  CER = {exec_metrics['CER']:.4f}  ({exec_metrics['N_C']}/{n_exec})")
    else:
        exec_metrics = {"ACC": 0, "SER": 0, "EER": 0, "CER": 0, "N": 0}

    print(f"\nEXECUTABILITY COVERAGE: {executability:.4f}  ({n_exec}/{n_total})")

    failure_counts = Counter(r.get("failure_label") for r in all_results if not r["executable"])
    print(f"\nFAILURE TAXONOMY:")
    for label, count in failure_counts.most_common():
        print(f"  {label}: {count}")

    per_dqc = {}
    for r in all_results:
        dqc = r["dqc_id"]
        if dqc not in per_dqc:
            per_dqc[dqc] = {"A": 0, "S": 0, "E": 0, "C": 0, "exec": 0, "total": 0}
        per_dqc[dqc][r["judge_label"]] += 1
        per_dqc[dqc]["total"] += 1
        if r["executable"]:
            per_dqc[dqc]["exec"] += 1

    print(f"\nPER-DQC BREAKDOWN:")
    for dqc in sorted(per_dqc):
        d = per_dqc[dqc]
        n = d["total"]
        print(f"  {dqc}: N={n} exec={d['exec']}/{n} ACC={d['A']/n:.4f} SER={d['S']/n:.4f} EER={d['E']/n:.4f} CER={d['C']/n:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "arelle_baseline_results.jsonl"
    with open(output_path, "w") as f:
        for r in all_results:
            row = {
                "id": r["id"],
                "dqc_id": r["dqc_id"],
                "dqc_rule_family": r["dqc_rule_family"],
                "executable": r["executable"],
                "failure_label": r.get("failure_label"),
                "prediction": r["prediction"],
                "gold": r["gold"],
                "judge_label": r["judge_label"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"\nPer-instance results: {output_path}")

    executable_ids = sorted([r["id"] for r in all_results if r["executable"]])
    exec_ids_path = RESULTS_DIR / "executable_finmr_ids.json"
    with open(exec_ids_path, "w") as f:
        json.dump({"executable_ids": executable_ids, "count": len(executable_ids)}, f, indent=2)
    print(f"Executable IDs: {exec_ids_path}")

    metrics_output = {
        "method": "arelle_symbolic_baseline",
        "dataset": "TheFinAI/FinMR",
        "n_instances": n_total,
        "full_set": {
            "ACC": round(metrics["ACC"], 4),
            "SER": round(metrics["SER"], 4),
            "EER": round(metrics["EER"], 4),
            "CER": round(metrics["CER"], 4),
            "N_A": metrics["N_A"],
            "N_S": metrics["N_S"],
            "N_E": metrics["N_E"],
            "N_C": metrics["N_C"],
        },
        "executable_subset": {
            "N": n_exec,
            "ACC": round(exec_metrics["ACC"], 4) if exec_results else 0,
            "SER": round(exec_metrics.get("SER", 0), 4),
            "EER": round(exec_metrics.get("EER", 0), 4),
            "CER": round(exec_metrics.get("CER", 0), 4),
        },
        "executability_coverage": round(executability, 4),
        "failure_taxonomy": dict(failure_counts),
        "per_dqc": {},
    }
    for dqc in sorted(per_dqc):
        d = per_dqc[dqc]
        n = d["total"]
        metrics_output["per_dqc"][dqc] = {
            "N": n,
            "N_executable": d["exec"],
            "ACC": round(d["A"] / n, 4) if n else 0,
            "SER": round(d["S"] / n, 4) if n else 0,
            "EER": round(d["E"] / n, 4) if n else 0,
            "CER": round(d["C"] / n, 4) if n else 0,
        }

    metrics_path = RESULTS_DIR / "arelle_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Aggregate metrics: {metrics_path}")


if __name__ == "__main__":
    main()
