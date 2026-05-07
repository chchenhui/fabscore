import os.path as osp
import json
import logging
import argparse
from typing import Dict, Set

from fabscore.utils.utils import _get_bucket_items


def _empty_results() -> Dict:
    return {
        "tables": [],
        "figures": [],
        "results_section": [],
        "summary": {
            "total": 0,
            "verified": 0,
            "data_fabrication": 0,
            "experiment_fabrication": 0,
            "result_fabrication": 0,
            "no_code_files": 0,
            "insufficient_evidence": 0,
        },
    }


def _verdict_from_fabrication_type(fabrication_type: str) -> str:
    mapping = {
        "data_fabrication": "Data Fabrication",
        "experiment_fabrication": "Experiment Fabrication",
        "result_fabrication": "Result Fabrication",
    }
    return mapping.get(fabrication_type, "Experiment Fabrication")


def _normalize_category(category: str) -> str:
    mapping = {
        "table": "tables",
        "tables": "tables",
        "figure": "figures",
        "figures": "figures",
        "results_section": "results_section",
    }
    return mapping.get(category, "results_section")


def _format_stage_explanation(stage: str, explanation: str, fallback: str) -> str:
    clean_explanation = (explanation or "").strip()
    prefix = f"{stage} stage:"
    if clean_explanation.lower().startswith(prefix.lower()):
        return clean_explanation
    if not clean_explanation:
        clean_explanation = fallback
    return f"{prefix} {clean_explanation}"


def _execution_detection_mode(item: Dict) -> str:
    detection_mode = item.get("detection_mode")
    if detection_mode in {
        "code_execution",
        "execution_artifact_analysis",
        "static_code_analysis",
    }:
        return detection_mode

    verification_process = item.get("verification_process", {}) or {}
    commands_executed = verification_process.get("commands_executed", []) or []
    if commands_executed:
        return "code_execution"

    return "execution_artifact_analysis"


def _normalize_execution_verdict(item: Dict) -> str:
    verdict = (item.get("verdict") or "").strip()
    if verdict == "Mismatch":
        return "Error"
    if verdict in {
        "Verified",
        "Data Fabrication",
        "Experiment Fabrication",
        "Result Fabrication",
        "Insufficient Evidence",
        "Error",
    }:
        return verdict
    return "Error"


def _entry_identity(category: str, entry: Dict) -> str:
    claim_index = entry.get("claim_index")
    if claim_index is not None:
        return f"{category}:{claim_index}"
    return f"{category}:{entry.get('claim', '')}"


def _append_unique(results: Dict, category: str, entry: Dict, seen: Set[str]) -> None:
    key = _entry_identity(category, entry)
    if key in seen:
        return
    results[category].append(entry)
    seen.add(key)


def _append_analysis_findings(results: Dict, analysis: Dict, seen: Set[str]) -> None:
    for item in _get_bucket_items(analysis, "no_code_files"):
        category = _normalize_category(item.get("category", "results_section"))
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": "No Code Files",
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "No relevant code files or experimental artifacts found during static analysis.",
                ),
                "evidence": "; ".join(item.get("code_evidence", [])) or "N/A",
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "no_code_files",
                },
                "verification_process": {
                    "files_analyzed": item.get("code_evidence", []),
                    "commands_executed": [],
                },
            },
            seen,
        )

    for item in _get_bucket_items(analysis, "obvious_hallucination"):
        category = _normalize_category(item.get("category", "results_section"))
        fabrication_type = item.get("fabrication_type", "result_fabrication")
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": _verdict_from_fabrication_type(fabrication_type),
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "Static analysis found an obvious fabrication issue.",
                ),
                "evidence": "; ".join(item.get("code_evidence", [])) or "N/A",
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "obvious_hallucination",
                    "fabrication_type": fabrication_type,
                },
                "verification_process": {
                    "files_analyzed": item.get("code_evidence", []),
                    "commands_executed": [],
                },
            },
            seen,
        )

    for item in _get_bucket_items(analysis, "static_verifiable"):
        category = _normalize_category(item.get("category", "results_section"))
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": "Verified",
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "Static analysis was sufficient to confirm this claim.",
                ),
                "evidence": "; ".join(item.get("code_evidence", [])) or "N/A",
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "static_verifiable",
                },
                "verification_process": {
                    "files_analyzed": item.get("code_evidence", []),
                    "commands_executed": [],
                },
            },
            seen,
        )

    for item in _get_bucket_items(analysis, "insufficient_evidence"):
        category = _normalize_category(item.get("category", "results_section"))
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": "Insufficient Evidence",
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "Static analysis found claim-relevant evidence, but it remained insufficient for a reliable conclusion.",
                ),
                "evidence": "; ".join(item.get("code_evidence", [])) or "N/A",
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "insufficient_evidence",
                },
                "verification_process": {
                    "files_analyzed": item.get("code_evidence", []),
                    "commands_executed": [],
                },
            },
            seen,
        )

    for item in _get_bucket_items(analysis, "error"):
        category = _normalize_category(item.get("category", "results_section"))
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": "Error",
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "Analysis output was incomplete or malformed for this claim.",
                ),
                "evidence": "; ".join(item.get("code_evidence", [])) or "N/A",
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "error",
                },
                "verification_process": {
                    "files_analyzed": item.get("code_evidence", []),
                    "commands_executed": [],
                },
            },
            seen,
        )


def _append_execution_results(results: Dict, execution_results: Dict, seen: Set[str]) -> None:
    for category in ("tables", "figures", "results_section"):
        for item in execution_results.get(category, []):
            normalized_item = dict(item)
            normalized_item["verdict"] = _normalize_execution_verdict(item)
            normalized_item["decision_stage"] = "execution"
            normalized_item.pop("detection_source", None)
            normalized_item["detection_mode"] = _execution_detection_mode(item)
            normalized_item["explanation"] = _format_stage_explanation(
                "Execution",
                item.get("explanation", ""),
                f"Execution stage classified this claim as {normalized_item.get('verdict', 'Unknown')}.",
            )
            verification_process = dict(item.get("verification_process", {}) or {})
            normalized_item["verification_process"] = {
                "files_analyzed": verification_process.get("files_analyzed", []),
                "commands_executed": verification_process.get("commands_executed", []),
            }
            _append_unique(results, category, normalized_item, seen)


def _append_missing_execution_required(results: Dict, analysis: Dict, seen: Set[str]) -> None:
    for item in _get_bucket_items(analysis, "execution_required"):
        category = _normalize_category(item.get("category", "results_section"))
        files = list(item.get("candidate_files") or [])
        for p in item.get("code_evidence") or []:
            if p not in files:
                files.append(p)
        eps = item.get("suggested_entrypoints") or []
        ev_parts = ["; ".join(map(str, files))] if files else []
        if eps:
            ev_parts.append("entrypoints: " + "; ".join(map(str, eps)))
        evidence = " | ".join(ev_parts) if ev_parts else "N/A"
        _append_unique(
            results,
            category,
            {
                "claim_index": item.get("index"),
                "claim": item.get("content", ""),
                "verdict": "Error",
                "explanation": _format_stage_explanation(
                    "Analysis",
                    item.get("reason", ""),
                    "This claim required execution, but no execution-stage result was available during summarization.",
                ),
                "evidence": evidence,
                "detection_mode": "static_code_analysis",
                "decision_stage": "analysis",
                "analysis_classification": {
                    "bucket": "execution_required",
                },
                "verification_process": {
                    "files_analyzed": files,
                    "commands_executed": [],
                },
            },
            seen,
        )


def _empty_verdict_breakdown() -> Dict:
    return {
        "verified": 0,
        "no_code_files": 0,
        "data_fabrication": 0,
        "experiment_fabrication": 0,
        "result_fabrication": 0,
        "insufficient_evidence": 0,
    }


def _increment_verdict_breakdown(breakdown: Dict, verdict: str) -> None:
    if verdict == "Verified":
        breakdown["verified"] += 1
    elif verdict == "Data Fabrication":
        breakdown["data_fabrication"] += 1
    elif verdict == "Experiment Fabrication":
        breakdown["experiment_fabrication"] += 1
    elif verdict == "Result Fabrication":
        breakdown["result_fabrication"] += 1
    elif verdict == "Insufficient Evidence":
        breakdown["insufficient_evidence"] += 1
    elif verdict == "No Code Files":
        breakdown["no_code_files"] += 1
    else:
        breakdown["insufficient_evidence"] += 1


def _recompute_summary(results: Dict) -> Dict:
    summary = {
        "total": 0,
        "verified": 0,
        "data_fabrication": 0,
        "experiment_fabrication": 0,
        "result_fabrication": 0,
        "no_code_files": 0,
        "insufficient_evidence": 0,
    }

    code_execution_breakdown = _empty_verdict_breakdown()
    execution_artifact_analysis_breakdown = _empty_verdict_breakdown()
    static_code_analysis_breakdown = _empty_verdict_breakdown()

    for category in ("tables", "figures", "results_section"):
        for item in results.get(category, []):
            summary["total"] += 1
            verdict = item.get("verdict", "")
            decision_stage = item.get("decision_stage") or ("analysis" if item.get("analysis_classification") else "execution")

            if verdict == "Verified":
                summary["verified"] += 1
            elif verdict == "Data Fabrication":
                summary["data_fabrication"] += 1
            elif verdict == "Experiment Fabrication":
                summary["experiment_fabrication"] += 1
            elif verdict == "Result Fabrication":
                summary["result_fabrication"] += 1
            elif verdict == "Insufficient Evidence":
                summary["insufficient_evidence"] += 1
            elif verdict == "No Code Files":
                summary["no_code_files"] += 1
            else:
                summary["insufficient_evidence"] += 1

            detection_mode = item.get("detection_mode")
            if detection_mode == "code_execution":
                _increment_verdict_breakdown(code_execution_breakdown, verdict)
                continue

            if detection_mode == "execution_artifact_analysis":
                _increment_verdict_breakdown(execution_artifact_analysis_breakdown, verdict)
                continue

            if detection_mode == "static_code_analysis":
                _increment_verdict_breakdown(static_code_analysis_breakdown, verdict)
                continue

            if decision_stage == "analysis":
                _increment_verdict_breakdown(static_code_analysis_breakdown, verdict)
            else:
                _increment_verdict_breakdown(execution_artifact_analysis_breakdown, verdict)

    results["summary"] = summary
    results["final_assessment"] = {
        "fab_score": round(summary["verified"] / summary["total"], 4) if summary["total"] else 0.0,
        "total_claims": summary["total"],
        "verified": summary["verified"],
        "data_fabrication": summary["data_fabrication"],
        "experiment_fabrication": summary["experiment_fabrication"],
        "result_fabrication": summary["result_fabrication"],
        "no_code_files": summary["no_code_files"],
        "insufficient_evidence": summary["insufficient_evidence"],
        "static_code_analysis_breakdown": static_code_analysis_breakdown,
        "execution_artifact_analysis_breakdown": execution_artifact_analysis_breakdown,
        "code_execution_breakdown": code_execution_breakdown,
    }
    return results


def generate_final_summary(analysis: Dict, execution_results: Dict) -> Dict:
    results = _empty_results()
    seen: Set[str] = set()

    _append_analysis_findings(results, analysis, seen)
    _append_execution_results(results, execution_results, seen)
    _append_missing_execution_required(results, analysis, seen)

    results["analysis_summary"] = analysis.get("summary", {})
    results["execution_summary"] = execution_results.get("summary", {})
    return _recompute_summary(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate the final FabScore summary from analysis and execution outputs.")
    parser.add_argument("--verify_dir", required=True, help="Path to the fabscore_{judge} directory.")
    args = parser.parse_args()

    analysis_path = osp.join(args.verify_dir, "fs_analysis.json")
    execution_path = osp.join(args.verify_dir, "fs_execution.json")
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    with open(execution_path, "r", encoding="utf-8") as f:
        execution_results = json.load(f)

    final = generate_final_summary(analysis, execution_results)
    print(json.dumps(final.get("final_assessment"), indent=4, ensure_ascii=False))