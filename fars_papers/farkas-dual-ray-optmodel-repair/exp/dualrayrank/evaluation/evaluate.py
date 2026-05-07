"""Evaluate attempt-0 (or any attempt) LP generation results against ground truth.

Solves each .lp file with HiGHS, compares objective to ground truth, and computes
pass@1 rates overall and per-difficulty. Saves infeasible instance list for downstream
repair experiments.

Usage:
    python -m dualrayrank.evaluation.evaluate --attempt-dir dualrayrank/outputs/attempt0
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dualrayrank.data.load_mamo import load_mamo
from dualrayrank.solver.highs_wrapper import HiGHSWrapper


def parse_ground_truth(answer_str: str) -> float | None:
    """Extract numeric value from MAMO answer string."""
    cleaned = str(answer_str).strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "").replace("\\$", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def classify_result(
    lp_path: Path,
    ground_truth: float | None,
    wrapper: HiGHSWrapper,
) -> dict:
    """Solve an LP file and classify the result."""
    record = {
        "lp_path": str(lp_path),
        "solver_status": "error",
        "objective": None,
        "ground_truth": ground_truth,
        "classification": "fail-error",
        "message": "",
    }

    if not lp_path.exists():
        record["message"] = "LP file not found"
        return record

    lp_text = lp_path.read_text(encoding="utf-8").strip()
    if not lp_text:
        record["message"] = "LP file is empty"
        return record

    try:
        result = wrapper.solve_lp(str(lp_path))
    except Exception as e:
        record["message"] = f"Solver exception: {e}"
        return record

    record["solver_status"] = result.status
    record["message"] = result.message

    if result.status == "error":
        record["classification"] = "fail-error"
    elif result.status == "infeasible":
        record["classification"] = "fail-infeasible"
    elif result.status == "unbounded":
        record["classification"] = "fail-unbounded"
    elif result.status == "optimal":
        record["objective"] = result.objective
        if ground_truth is not None and result.objective is not None:
            if HiGHSWrapper.check_objective(result.objective, ground_truth):
                record["classification"] = "pass"
            else:
                record["classification"] = "fail-wrong-objective"
        else:
            record["classification"] = "fail-error"
            record["message"] = "Could not parse ground truth or objective"
    else:
        record["classification"] = "fail-error"

    return record


def run_evaluation(attempt_dir: Path, results_dir: Path):
    """Run full evaluation pipeline on an attempt directory."""
    wrapper = HiGHSWrapper()
    instances = load_mamo()
    instance_map = {inst["id"]: inst for inst in instances}

    results = []
    for inst in instances:
        instance_id = inst["id"]
        difficulty = inst["difficulty"]
        lp_path = attempt_dir / f"{difficulty}_{instance_id}.lp"
        ground_truth = parse_ground_truth(inst["Answer"])

        record = classify_result(lp_path, ground_truth, wrapper)
        record["instance_id"] = instance_id
        record["difficulty"] = difficulty
        results.append(record)

    total = len(results)
    status_counts = Counter(r["classification"] for r in results)
    pass_count = status_counts.get("pass", 0)

    easy = [r for r in results if r["difficulty"] == "EasyLP"]
    complex_ = [r for r in results if r["difficulty"] == "ComplexLP"]

    easy_pass = sum(1 for r in easy if r["classification"] == "pass")
    complex_pass = sum(1 for r in complex_ if r["classification"] == "pass")

    easy_infeasible = sum(1 for r in easy if r["classification"] == "fail-infeasible")
    complex_infeasible = sum(1 for r in complex_ if r["classification"] == "fail-infeasible")
    total_infeasible = status_counts.get("fail-infeasible", 0)

    summary = {
        "total_instances": total,
        "pass_at_1": pass_count / total if total > 0 else 0,
        "pass_count": pass_count,
        "status_distribution": dict(status_counts),
        "EasyLP": {
            "total": len(easy),
            "pass_count": easy_pass,
            "pass_at_1": easy_pass / len(easy) if easy else 0,
            "infeasible_count": easy_infeasible,
            "infeasible_rate": easy_infeasible / len(easy) if easy else 0,
        },
        "ComplexLP": {
            "total": len(complex_),
            "pass_count": complex_pass,
            "pass_at_1": complex_pass / len(complex_) if complex_ else 0,
            "infeasible_count": complex_infeasible,
            "infeasible_rate": complex_infeasible / len(complex_) if complex_ else 0,
        },
        "infeasible_total": total_infeasible,
        "infeasible_rate": total_infeasible / total if total > 0 else 0,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "attempt0_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_instance": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}")

    infeasible_instances = [
        {"instance_id": r["instance_id"], "difficulty": r["difficulty"]}
        for r in results
        if r["classification"] == "fail-infeasible"
    ]
    infeasible_path = attempt_dir / "infeasible_instances.json"
    with open(infeasible_path, "w", encoding="utf-8") as f:
        json.dump(infeasible_instances, f, indent=2, ensure_ascii=False)
    print(f"Infeasible instances ({len(infeasible_instances)}) saved to {infeasible_path}")

    print(f"\n{'='*60}")
    print(f"ATTEMPT-0 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total instances: {total}")
    print(f"Overall pass@1:  {summary['pass_at_1']:.4f} ({pass_count}/{total})")
    print(f"EasyLP pass@1:   {summary['EasyLP']['pass_at_1']:.4f} ({easy_pass}/{len(easy)})")
    print(f"ComplexLP pass@1: {summary['ComplexLP']['pass_at_1']:.4f} ({complex_pass}/{len(complex_)})")
    print(f"\nStatus distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/total:.4f})")
    print(f"\nInfeasible rate: {summary['infeasible_rate']:.4f} ({total_infeasible}/{total})")
    print(f"  EasyLP:    {summary['EasyLP']['infeasible_rate']:.4f} ({easy_infeasible}/{len(easy)})")
    print(f"  ComplexLP: {summary['ComplexLP']['infeasible_rate']:.4f} ({complex_infeasible}/{len(complex_)})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LP generation results")
    parser.add_argument("--attempt-dir", default="dualrayrank/outputs/attempt0")
    parser.add_argument("--results-dir", default="dualrayrank/results")
    args = parser.parse_args()

    run_evaluation(Path(args.attempt_dir), Path(args.results_dir))


if __name__ == "__main__":
    main()
