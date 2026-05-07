# Ablation study: measure the contribution of the href-rewriting step in the
# XBRL reconstruction pipeline. Runs the full Arelle symbolic pipeline with
# skip_href_rewrite=True and compares to the normal (with-rewrite) baseline.

import argparse
import json
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

NO_REWRITE_PKG_DIR = OUTPUT_DIR / "xbrl_packages_no_rewrite"


def run_single_no_rewrite(inst, pkg_dir: str) -> dict:
    target = extract_target(inst)
    pkg_info = reconstruct_xbrl_package(inst, pkg_dir, skip_href_rewrite=True)

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


def load_with_rewrite_results() -> list[dict]:
    path = OUTPUT_DIR / "arelle_baseline_results.jsonl"
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", type=int, default=0)
    parser.add_argument("--ids", type=str, default="")
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

    NO_REWRITE_PKG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx, inst in enumerate(instances):
        pkg_dir = str(NO_REWRITE_PKG_DIR / str(inst.id))
        try:
            result = run_single_no_rewrite(inst, pkg_dir)
        except Exception as e:
            print(f"  ERROR on instance {inst.id}: {e}")
            traceback.print_exc()
            result = {
                "id": inst.id,
                "dqc_id": inst.dqc_id,
                "dqc_rule_family": getattr(inst, "dqc_rule_family", ""),
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

    eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in all_results]
    eval_output = evaluate(eval_input)
    for r, ev in zip(all_results, eval_output["per_instance"]):
        r["judge_label"] = ev["label"]

    n_total = len(all_results)
    n_exec_no_rewrite = sum(1 for r in all_results if r["executable"])
    exec_coverage_no_rewrite = n_exec_no_rewrite / n_total if n_total else 0

    exec_results_no_rewrite = [r for r in all_results if r["executable"]]
    if exec_results_no_rewrite:
        exec_eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in exec_results_no_rewrite]
        exec_eval_output = evaluate(exec_eval_input)
        acc_exec_no_rewrite = exec_eval_output["metrics"]["ACC"]
    else:
        acc_exec_no_rewrite = 0.0

    failure_counts_no_rewrite = Counter(r.get("failure_label") for r in all_results if not r["executable"])

    print(f"\n{'='*60}")
    print("NO-REWRITE ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"Executability: {n_exec_no_rewrite}/{n_total} ({exec_coverage_no_rewrite:.4f})")
    print(f"ACC on executable subset: {acc_exec_no_rewrite:.4f}")
    print(f"Failure causes:")
    for label, count in failure_counts_no_rewrite.most_common():
        print(f"  {label}: {count}")

    with_rewrite = load_with_rewrite_results()
    wr_ids = set(r["id"] for r in with_rewrite)
    nr_ids = set(r["id"] for r in all_results)
    common_ids = wr_ids & nr_ids

    wr_by_id = {r["id"]: r for r in with_rewrite}
    nr_by_id = {r["id"]: r for r in all_results}

    wr_common = [wr_by_id[i] for i in sorted(common_ids)]
    nr_common = [nr_by_id[i] for i in sorted(common_ids)]

    n_common = len(common_ids)
    n_exec_wr = sum(1 for r in wr_common if r["executable"])
    n_exec_nr = sum(1 for r in nr_common if r["executable"])

    exec_wr_results = [r for r in wr_common if r["executable"]]
    if exec_wr_results:
        wr_eval_input = [{"id": r["id"], "prediction": r["prediction"], "gold": r["gold"]} for r in exec_wr_results]
        wr_eval_output = evaluate(wr_eval_input)
        acc_exec_wr = wr_eval_output["metrics"]["ACC"]
    else:
        acc_exec_wr = 0.0

    newly_failed_ids = set()
    for i in sorted(common_ids):
        if wr_by_id[i]["executable"] and not nr_by_id[i]["executable"]:
            newly_failed_ids.add(i)
    newly_failed_causes = Counter(nr_by_id[i].get("failure_label") for i in newly_failed_ids)

    print(f"\n{'='*60}")
    print("COMPARISON (common {0} instances)".format(n_common))
    print(f"{'='*60}")
    print(f"With rewrite:    exec={n_exec_wr}/{n_common} ({n_exec_wr/n_common:.4f})  ACC_exec={acc_exec_wr:.4f}")
    print(f"Without rewrite: exec={n_exec_nr}/{n_common} ({n_exec_nr/n_common:.4f})  ACC_exec={acc_exec_no_rewrite:.4f}")
    print(f"Newly failed (exec->non-exec): {len(newly_failed_ids)}")
    print(f"Newly failed causes:")
    for label, count in newly_failed_causes.most_common():
        print(f"  {label}: {count}")

    dominant_failure = failure_counts_no_rewrite.most_common(1)[0][0] if failure_counts_no_rewrite else "N/A"

    comparison_table = [
        {
            "variant": "With href rewrite (baseline)",
            "executability_coverage": round(n_exec_wr / n_common, 4) if n_common else 0,
            "n_executable": n_exec_wr,
            "n_total": n_common,
            "acc_on_executable_subset": round(acc_exec_wr, 4),
            "dominant_failure_cause": "N/A",
        },
        {
            "variant": "Without href rewrite (ablation)",
            "executability_coverage": round(exec_coverage_no_rewrite, 4),
            "n_executable": n_exec_no_rewrite,
            "n_total": n_total,
            "acc_on_executable_subset": round(acc_exec_no_rewrite, 4),
            "dominant_failure_cause": dominant_failure,
        },
    ]

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Variant':<40} {'Exec Coverage':<18} {'ACC (Exec)':<15} {'Dominant Failure'}")
    print("-" * 100)
    for row in comparison_table:
        print(f"{row['variant']:<40} {row['executability_coverage']:<18.4f} {row['acc_on_executable_subset']:<15.4f} {row['dominant_failure_cause']}")

    output = {
        "experiment": "ablation_no_href_rewrite",
        "dataset": "TheFinAI/FinMR",
        "n_instances": n_total,
        "with_rewrite": {
            "n_executable": n_exec_wr,
            "executability_coverage": round(n_exec_wr / n_common, 4) if n_common else 0,
            "acc_on_executable_subset": round(acc_exec_wr, 4),
        },
        "without_rewrite": {
            "n_executable": n_exec_no_rewrite,
            "executability_coverage": round(exec_coverage_no_rewrite, 4),
            "acc_on_executable_subset": round(acc_exec_no_rewrite, 4),
        },
        "failure_causes_without_rewrite": dict(failure_counts_no_rewrite),
        "newly_failed_count": len(newly_failed_ids),
        "newly_failed_causes": dict(newly_failed_causes),
        "comparison_table": comparison_table,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "ablation_no_rewrite.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    per_instance_path = OUTPUT_DIR / "ablation_no_rewrite_results.jsonl"
    with open(per_instance_path, "w") as f:
        for r in all_results:
            row = {
                "id": r["id"],
                "dqc_id": r["dqc_id"],
                "dqc_rule_family": r["dqc_rule_family"],
                "executable": r["executable"],
                "failure_label": r.get("failure_label"),
                "prediction": r["prediction"],
                "gold": r["gold"],
                "judge_label": r.get("judge_label"),
            }
            f.write(json.dumps(row) + "\n")
    print(f"Per-instance results: {per_instance_path}")


if __name__ == "__main__":
    main()
