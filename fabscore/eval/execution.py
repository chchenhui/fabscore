import os
import os.path as osp
import json
import logging
import argparse
import signal
import subprocess
import time
import re
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv

from fabscore.utils.utils import SUPPORTED_JUDGES, _get_bucket_items


load_dotenv()


EXECUTION_AGENT_TIMEOUT_SECONDS = 86400
EXECUTION_AGENT_TIMEOUT_RETRY_SECONDS = 86400
MAX_TIMEOUT_RETRIES = 1

_VERDICT_SUMMARY_KEYS = {
    "Verified": "verified",
    "Data Fabrication": "data_fabrication",
    "Experiment Fabrication": "experiment_fabrication",
    "Result Fabrication": "result_fabrication",
    "No Code Files": "no_code_files",
    "Insufficient Evidence": "insufficient_evidence",
    "Error": "error",
}

_FAB_TYPE_TO_VERDICT_LABEL = {
    "data_fabrication": "Data Fabrication",
    "experiment_fabrication": "Experiment Fabrication",
    "result_fabrication": "Result Fabrication",
}
_VERDICT_LABEL_TO_FAB_TYPE = {v: k for k, v in _FAB_TYPE_TO_VERDICT_LABEL.items()}

_VALID_EXECUTION_VERDICTS = frozenset(_VERDICT_SUMMARY_KEYS.keys())


def _fabscore_judge_dir(task_path: str, judge_type: str) -> str:
    return osp.join(task_path, f"fabscore_{judge_type}")


def _claim_lookup_from_extracted(extracted_data: Dict) -> Dict[int, Dict]:
    lookup: Dict[int, Dict] = {}
    idx_counter = 1
    for category_key, category_name in (
        ("tables", "tables"),
        ("figures", "figures"),
        ("results_section", "results_section"),
    ):
        for item in extracted_data.get(category_key, []):
            lookup[idx_counter] = {
                "index": idx_counter,
                "category": category_name,
                "content": item,
            }
            idx_counter += 1
    return lookup


def _load_extracted_data(task_path: str, judge_type: str, extracted_results_path: Optional[str]) -> Dict:
    if extracted_results_path is None:
        extracted_results_path = osp.join(_fabscore_judge_dir(task_path, judge_type), "fs_extracted.json")
    with open(extracted_results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _empty_execution_result() -> Dict:
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
            "error": 0,
        },
    }


def _normalize_result_category(category: str) -> str:
    mapping = {
        "table": "tables",
        "tables": "tables",
        "figure": "figures",
        "figures": "figures",
        "results_section": "results_section",
    }
    return mapping.get(category, "results_section")


def _recompute_summary(results: Dict) -> None:
    summary = dict.fromkeys(["total", *_VERDICT_SUMMARY_KEYS.values()], 0)
    for category in ("tables", "figures", "results_section"):
        for item in results.get(category, []):
            summary["total"] += 1
            key = _VERDICT_SUMMARY_KEYS.get(item.get("verdict"))
            if key:
                summary[key] += 1
    results["summary"] = summary


def _save_progress(progress_path: str, aggregated: Dict, completed_claim_indices: List[int]) -> None:
    progress = {
        "completed_claim_indices": completed_claim_indices,
        "aggregated": aggregated,
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=4, ensure_ascii=False)


def _load_progress(progress_path: str) -> Optional[Dict]:
    if not osp.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logging.warning(f"Corrupted progress file {progress_path}, starting fresh: {exc}")
        return None


def _claim_command_output_path(workspace_dir: str, claim_index: int) -> str:
    return osp.join(workspace_dir, f"claim_{claim_index}_command_output.txt")


def _append_execution_log(log_path: str, entry: Dict) -> None:
    payload = {"entries": []}

    if osp.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                payload = existing
            elif isinstance(existing, list):
                payload["entries"] = existing
        except (json.JSONDecodeError, IOError):
            logging.warning(f"Corrupted execution log {log_path}, overwriting it.")

    payload["entries"].append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _build_evidence_text(evidence: str, code_reference: str) -> str:
    clean_evidence = (evidence or "").strip()
    clean_code_reference = (code_reference or "").strip()

    if clean_evidence and clean_code_reference and clean_code_reference not in clean_evidence:
        return f"{clean_evidence} | Code: {clean_code_reference}"
    if clean_evidence:
        return clean_evidence
    if clean_code_reference:
        return clean_code_reference
    return "N/A"


def _parse_code_references(code_reference: str) -> List[str]:
    if not code_reference:
        return []

    refs: List[str] = []
    for part in re.split(r"[;\n]", code_reference):
        clean = part.strip()
        if clean and clean != "N/A" and clean not in refs:
            refs.append(clean)
    return refs


def _extract_code_references_from_text(text: str) -> List[str]:
    if not text:
        return []

    refs: List[str] = []
    for path, line in re.findall(r"([A-Za-z0-9_./-]+\.py):(\d+(?:-\d+)?)", text):
        ref = f"{path}:{line}"
        if ref not in refs:
            refs.append(ref)
    for path, line in re.findall(r"([A-Za-z0-9_./-]+\.py) line (\d+(?:-\d+)?)", text):
        ref = f"{path}:{line}"
        if ref not in refs:
            refs.append(ref)
    return refs


def _normalize_code_reference(verdict_data: Dict) -> str:
    explicit_reference = verdict_data.get("code_reference", verdict_data.get("code_evidence", "")) or ""
    refs = _parse_code_references(explicit_reference)
    if refs:
        return "; ".join(refs)

    extracted_refs = _extract_code_references_from_text(
        " ".join(
            filter(
                None,
                [
                    verdict_data.get("explanation", ""),
                    verdict_data.get("evidence_extracted", ""),
                    verdict_data.get("evidence", ""),
                ],
            )
        )
    )
    return "; ".join(extracted_refs) if extracted_refs else "N/A"


# First-token prefixes: "code execution" is a subset (script/build/runtime); the rest are still
# logged commands for evidence / command_output policy.
_CODE_EXECUTION_COMMANDS = frozenset(
    {
        "python",
        "python3",
        "bash",
        "sh",
        "node",
        "npm",
        "npx",
        "pytest",
        "make",
        "cmake",
        "uv",
        "conda",
        "poetry",
        "pipenv",
        "micromamba",
        "mamba",
    }
)
_INSPECTION_TOOL_COMMANDS = frozenset(
    {
        "git",
        "rg",
        "grep",
        "sed",
        "awk",
        "cat",
        "head",
        "tail",
        "wc",
        "ls",
        "find",
        "env",
        "timeout",
    }
)
_COMMAND_PATH_PREFIXES = ("./", "../", "/")
_REAL_COMMANDS = _CODE_EXECUTION_COMMANDS | _INSPECTION_TOOL_COMMANDS


def _has_command_prefix(first_token: str, commands: Set[str]) -> bool:
    if first_token in commands:
        return True
    return first_token.startswith(_COMMAND_PATH_PREFIXES)


def _is_real_command(command: str) -> bool:
    clean_command = (command or "").strip()
    if not clean_command:
        return False
    # Relaxed restriction: keep all non-empty commands
    return True


def _is_code_execution_command(command: str) -> bool:
    clean_command = (command or "").strip()
    if not clean_command:
        return False
    return _has_command_prefix(clean_command.split()[0], _CODE_EXECUTION_COMMANDS)


def _extract_commands_executed(verdict_data: Dict) -> List[str]:
    raw_commands = verdict_data.get("command_executed", "")

    if isinstance(raw_commands, list):
        commands = [str(command).strip() for command in raw_commands]
    else:
        commands = [str(raw_commands).strip()]

    return [command for command in commands if _is_real_command(command)]


def _apply_command_output_file_policy(command_output_path: str, verdict_data: Dict) -> None:
    """Keep claim_*_command_output.txt consistent with recorded commands (agent-written file).

    Removes accidental empty files when command_executed lists no runnable commands; warns on
    mismatches. Python never creates this file.
    """
    if not command_output_path or not isinstance(verdict_data, dict):
        return
    commands = _extract_commands_executed(verdict_data)
    if not osp.exists(command_output_path):
        if commands:
            logging.warning(
                "Expected command output file (commands were run) but missing: %s",
                command_output_path,
            )
        return
    try:
        size = osp.getsize(command_output_path)
    except OSError:
        return
    if not commands:
        if size == 0:
            try:
                os.remove(command_output_path)
            except OSError:
                pass
        else:
            logging.warning(
                "Command output file has content but command_executed lists no runnable command: %s",
                command_output_path,
            )


def _infer_execution_classification(verdict_data: Dict) -> Optional[str]:
    existing = verdict_data.get("execution_classification")
    if isinstance(existing, dict) and existing.get("fabrication_type"):
        return existing.get("fabrication_type")

    text = " ".join(
        filter(
            None,
            [
                verdict_data.get("explanation", ""),
                verdict_data.get("evidence_extracted", ""),
                verdict_data.get("evidence", ""),
                verdict_data.get("code_reference", ""),
                verdict_data.get("command_executed", ""),
            ],
        )
    ).lower()

    # Evidence context
    evidence = verdict_data.get("evidence_extracted", verdict_data.get("evidence", "")).lower()
    code_ref = verdict_data.get("code_reference", "").lower()
    command = verdict_data.get("command_executed", "").lower()

    _issue_base = [
        "not found",
        "cannot load",
        "can't load",
        "failed to load",
        "cannot import",
        "can't import",
        "failed to import",
        "cannot resolve",
        "can't resolve",
        "does not exist",
        "doesn't exist",
    ]
    _issue_paper_conflict = [
        "conflicts with the paper",
        "conflicts with paper",
        "contradicts the paper",
        "contradicts paper",
    ]
    data_issue_keywords = _issue_base + [
        "missing dataset",
        "corrupted dataset",
        "dataset mismatch",
        "split mismatch",
        "label mismatch",
        "size mismatch",
        "source mismatch",
        "synthetic data",
        "synthetic proxy",
    ] + _issue_paper_conflict
    experiment_issue_keywords = _issue_base + [
        "wrong variant",
        "incompatible",
        "mismatch",
    ] + _issue_paper_conflict + [
        "self-contradictory",
        "contradictory",
        "broken control flow",
        "invalid evaluation",
        "hardcoded result",
        "hardcode",
        "fallback",
        "fallback constant",
        "hardcoded default",
        "failed computation",
        "np.random",
        "randomly generated",
        "wrong metric",
        "syntax error",
        "importerror",
        "modulenotfounderror",
        "module not found",
        "runtime error",
        "crash",
    ]

    data_subject_keywords = [
        "dataset",
        "datasets",
        "load_dataset",
        "datasets.load_dataset",
        "dataloader",
        "data loader",
        "train split",
        "test split",
        "validation split",
        "val split",
        "split",
        "label",
        "labels",
        "sample count",
        "dataset size",
        "data size",
        "huggingface dataset",
        "hf dataset",
    ]
    experiment_subject_keywords = [
        "model",
        "models",
        "automodel",
        "from_pretrained",
        "tokenizer",
        "architecture",
        "variant",
        "method",
        "implementation",
        "training",
        "inference",
        "evaluation",
        "metric",
        "protocol",
        "procedure",
        "control flow",
        "checkpoint",
    ]
    result_keywords = [
        "mismatch",
        "does not match",
        "do not match",
        "mathematically impossible",
        "impossible",
        "deviation",
        "value deviation",
        "different from the paper",
        "different from paper",
        "inconsistent with the reported result",
        "reported result is",
        "paper says",
        "paper reports",
        "reproduced result",
        "actual result",
        "observed result",
        "obtained result",
        "numerical mismatch",
        "numerical conflict",
        "fundamental conflict in values",
    ]
    successful_execution_keywords = [
        "runs successfully",
        "ran successfully",
        "executed successfully",
        "execution succeeded",
        "completed successfully",
        "successfully completed",
        "produced",
        "reproduced",
        "obtained",
        "observed",
        "computed",
        "generated",
    ]
    explicit_conflict_keywords = [
        "conflicts with the paper",
        "conflicts with paper",
        "contradicts the paper",
        "contradicts paper",
        "clearly conflicts with the paper",
        "clearly conflicts with paper",
        "mismatch",
        "does not match",
        "do not match",
        "different from the paper",
        "different from paper",
        "fabricated or unsupported",
        "wrong variant",
        "wrong metric",
        "self-contradictory",
        "contradictory",
        "broken control flow",
        "invalid evaluation",
        "synthetic data conflict",
        "synthetic data",
        "matches the fallback",
        "matches the hardcoded",
        "fallback constant",
        "hardcoded default",
        "failed computation",
    ]
    dataset_source_conflict_keywords = [
        "dataset identifier that does not exist",
        "hugging face dataset identifier that has never existed",
        "huggingface dataset identifier that has never existed",
        "dataset does not exist in the stated source",
        "dataset cannot be found from the repository's claimed source",
        "dataset cannot be found from the claimed source",
    ]
    model_source_conflict_keywords = [
        "model identifier that does not exist",
        "hugging face model identifier that has never existed",
        "huggingface model identifier that has never existed",
        "model does not exist in the stated source",
        "model cannot be found from the repository's claimed source",
        "model cannot be found from the claimed source",
    ]
    supporting_artifact_missing_keywords = [
        "missing checkpoint",
        "missing checkpoints",
        "missing log",
        "missing logs",
        "missing cache",
        "missing caches",
        "missing saved output",
        "missing saved outputs",
        "missing plotting artifact",
        "missing plotting artifacts",
        "missing plotting input",
        "missing plotting inputs",
        "missing intermediate artifact",
        "missing intermediate artifacts",
        "all_results.npy",
    ]

    def _has(haystack: str, keywords: List[str]) -> bool:
        return any(k in haystack for k in keywords)

    data_context = " ".join([text, evidence, code_ref, command])
    has_explicit_conflict = _has(data_context, explicit_conflict_keywords)
    has_dataset_source_conflict = _has(data_context, dataset_source_conflict_keywords)
    has_model_source_conflict = _has(data_context, model_source_conflict_keywords)
    has_only_supporting_artifact_gap = (
        _has(data_context, supporting_artifact_missing_keywords)
        and not has_explicit_conflict
        and not has_dataset_source_conflict
        and not has_model_source_conflict
    )

    if has_only_supporting_artifact_gap:
        return None

    # Data fabrication: dataset identity/source/size/label/split issues only.
    if _has(data_context, data_subject_keywords) and (
        has_dataset_source_conflict
        or (has_explicit_conflict and _has(data_context, data_issue_keywords))
    ):
        return "data_fabrication"

    # Experiment fabrication: model/method/implementation/procedure issues.
    if _has(data_context, experiment_subject_keywords) and (
        has_model_source_conflict
        or (has_explicit_conflict and _has(data_context, experiment_issue_keywords))
    ):
        return "experiment_fabrication"

    # Result fabrication: execution path succeeded enough to compare outputs, OR a math/value conflict was noted.
    if evidence and has_explicit_conflict and _has(text, result_keywords):
        # 1. Successful execution led to result mismatch
        if _has(text, successful_execution_keywords) or _has(evidence, successful_execution_keywords):
            return "result_fabrication"
        # 2. Static logic check (even without success) established mathematical impossibility in output values
        if any(phrase in text for phrase in ["paper says", "paper reports", "reported result is", "reproduced result", "actual result", "mathematically impossible", "value deviation"]):
            return "result_fabrication"

    return None


def _verdict_from_fabrication_type(fabrication_type: str) -> str:
    return _FAB_TYPE_TO_VERDICT_LABEL.get(fabrication_type, "Experiment Fabrication")


def _verdict_to_fabrication_type(verdict: str) -> Optional[str]:
    return _VERDICT_LABEL_TO_FAB_TYPE.get((verdict or "").strip())


def _format_execution_explanation(verdict_data: Dict, normalized_verdict: str) -> str:
    base_reason = (verdict_data.get("explanation") or "").strip()
    if not base_reason:
        base_reason = f"Execution stage classified this claim as {normalized_verdict}."

    if base_reason.lower().startswith("execution stage:"):
        return base_reason

    return f"Execution stage: {base_reason}"


def _execution_detection_mode(commands_executed: List[str]) -> str:
    if any(_is_code_execution_command(command) for command in commands_executed):
        return "code_execution"
    return "execution_artifact_analysis"


def _execution_verdict_options() -> str:
    return "Verified|Data Fabrication|Experiment Fabrication|Result Fabrication|No Code Files|Insufficient Evidence|Error"


def _normalize_verdict(verdict_data: Dict) -> str:
    raw_verdict = (verdict_data.get("verdict") or "Error").strip()
    fabrication_type = _infer_execution_classification(verdict_data)
    raw_fabrication_type = _verdict_to_fabrication_type(raw_verdict)

    # If the agent's verdict conflicts with a fabrication classification implied by
    # explicit execution_classification or the explanation/evidence text, prefer the
    # fabrication classification over a contradictory Verified or wrong fabrication label.
    if fabrication_type and raw_verdict == "Verified":
        return _verdict_from_fabrication_type(fabrication_type)

    if fabrication_type and raw_fabrication_type and raw_fabrication_type != fabrication_type:
        return _verdict_from_fabrication_type(fabrication_type)

    if raw_verdict in _VALID_EXECUTION_VERDICTS:
        return raw_verdict

    # If verdict is Mismatch or not valid, use fabrication_type if available
    if fabrication_type:
        return _verdict_from_fabrication_type(fabrication_type)

    return "Error"


def _has_verdict_fields(obj: Dict) -> bool:
    return "verdict" in obj and "claim_index" in obj


def _validate_verdict_list(data: Any) -> Optional[List[Dict]]:
    if not isinstance(data, list) or not data:
        return None

    cleaned: List[Dict] = []
    for item in data:
        if not isinstance(item, dict) or not _has_verdict_fields(item):
            return None
        cleaned.append(item)

    return cleaned


def _validate_single_verdict(data: Any) -> Optional[Dict]:
    if not isinstance(data, dict) or not _has_verdict_fields(data):
        return None
    return data


def _iter_strings(obj):
    """Recursively yield all string values from nested dicts/lists."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def _extract_json_blocks(text: str) -> List[str]:
    candidates: List[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    # 1. Markdown-wrapped JSON blocks (```json ... ```)
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        clean_block = block.strip()
        if clean_block and clean_block not in candidates:
            candidates.append(clean_block)

    # 2. Bare JSON blocks containing "verdict" (handles noisy stdout with huge data dumps)
    # Limit search to 100k chars to avoid catastrophic backtracking on massive logs
    for block in re.findall(r"(\{[\s\S]{0,100000}?\"verdict\"[\s\S]{0,100000}?\})", text):
        if block not in candidates:
            candidates.append(block)

    # 3. Codex NDJSON format: verdict is embedded in item["text"] of an agent_message line.
    # Fast-path: scan only agent_message lines (skip huge command_execution aggregated_output).
    # Collect all agent_message text candidates, then fall back to full _iter_strings if needed.
    ndjson_agent_texts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        # Direct verdict object (e.g. bare JSON output from non-NDJSON judges)
        if isinstance(obj, dict) and "verdict" in obj:
            cand = json.dumps(obj)
            if cand not in candidates:
                candidates.append(cand)
            continue

        # Codex NDJSON: look for agent_message items only
        if isinstance(obj, dict) and obj.get("type") == "item.completed":
            item = obj.get("item") or {}
            if item.get("type") == "agent_message":
                text_val = item.get("text") or ""
                if "verdict" in text_val:
                    ndjson_agent_texts.append(text_val)
                continue  # skip _iter_strings for this line

        # For non-NDJSON lines (or unrecognised NDJSON events): search all strings,
        # but cap string size to avoid catastrophic backtracking.
        for s in _iter_strings(obj):
            if "verdict" not in s or len(s) > 500_000:
                continue
            try:
                inner = json.loads(s)
                inner_str = json.dumps(inner)
                if inner_str not in candidates:
                    candidates.append(inner_str)
            except (json.JSONDecodeError, ValueError):
                if len(s) <= 100_000:
                    for block in re.findall(r"(\{[\s\S]{0,100000}?\"verdict\"[\s\S]{0,100000}?\})", s):
                        if block not in candidates:
                            candidates.append(block)

    # Prefer the last agent_message (closest to final verdict) over earlier ones
    for text_val in reversed(ndjson_agent_texts):
        try:
            inner = json.loads(text_val)
            inner_str = json.dumps(inner)
            if inner_str not in candidates:
                candidates.insert(1, inner_str)  # high priority, after full-text candidate
        except (json.JSONDecodeError, ValueError):
            for block in re.findall(r"(\{[\s\S]{0,100000}?\"verdict\"[\s\S]{0,100000}?\})", text_val):
                if block not in candidates:
                    candidates.insert(1, block)

    return candidates



def _search_verdict_list_in_text(text: str, seen_texts: Set[str]) -> Optional[List[Dict]]:
    decoder = json.JSONDecoder()

    for candidate in _extract_json_blocks(text):
        if candidate in seen_texts:
            continue
        seen_texts.add(candidate)

        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if parsed is not None:
            verdicts = _find_verdict_list(parsed, seen_texts)
            if verdicts:
                return verdicts

        for match in re.finditer(r"\[", candidate):
            try:
                parsed_fragment, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue

            verdicts = _find_verdict_list(parsed_fragment, seen_texts)
            if verdicts:
                return verdicts

    return None


def _find_verdict_list(payload: Any, seen_texts: Optional[Set[str]] = None) -> Optional[List[Dict]]:
    if seen_texts is None:
        seen_texts = set()

    single_verdict = _validate_single_verdict(payload)
    if single_verdict:
        return [single_verdict]

    verdicts = _validate_verdict_list(payload)
    if verdicts:
        return verdicts

    if isinstance(payload, dict):
        nested = payload.values()
    elif isinstance(payload, list):
        nested = payload
    else:
        nested = None

    if nested is not None:
        for item in nested:
            verdicts = _find_verdict_list(item, seen_texts)
            if verdicts:
                return verdicts
        return None

    if isinstance(payload, str) and payload.strip():
        return _search_verdict_list_in_text(payload, seen_texts)

    return None


def _is_timeout_error(verdict_data: Dict) -> bool:
    if not isinstance(verdict_data, dict):
        return False
    if verdict_data.get("verdict") != "Error":
        return False
    explanation = (verdict_data.get("explanation") or "").lower()
    return "timed out" in explanation


def _execution_agent_error_verdict(claim_index: int, explanation: str) -> Dict:
    return {
        "claim_index": claim_index,
        "verdict": "Error",
        "evidence_extracted": "",
        "explanation": explanation,
        "command_executed": "",
        "code_reference": "",
    }


def _pick_claim_verdict(verdicts: List[Dict], claim_index: int) -> Optional[Dict]:
    expected = str(claim_index)
    for verdict in verdicts:
        if str(verdict.get("claim_index")) == expected:
            return verdict

    if len(verdicts) == 1:
        fallback = dict(verdicts[0])
        fallback["claim_index"] = claim_index
        logging.warning(
            "Single verdict claim_index mismatch; forcing claim_index=%s (received=%s).",
            claim_index,
            verdicts[0].get("claim_index"),
        )
        return fallback

    return None


def _append_autonomous_execution_log(
    task_path: str,
    judge_type: str,
    claim: Dict,
    prompt: str,
    retry_attempt: int,
    retry_reason: str,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: Optional[int] = None,
    timed_out: bool = False,
) -> None:
    execute_dir = _fabscore_judge_dir(task_path, judge_type)
    os.makedirs(execute_dir, exist_ok=True)
    log_path = osp.join(execute_dir, "execution_log.json")
    entry: Dict[str, Any] = {
        "claim_index": claim["index"],
        "claim": claim,
        "prompt": prompt,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "timestamp": int(time.time()),
        "retry_attempt": retry_attempt,
        "retry_reason": retry_reason,
    }
    if timed_out:
        entry["timed_out"] = True
    _append_execution_log(log_path, entry)


def _autonomous_execute_claim(
    task_path: str,
    paper_file: str,
    claim: Dict,
    judge_type: str,
    model_name: Optional[str],
    workspace_dir: str,
    retry_attempt: int = 0,
    timeout_seconds: int = EXECUTION_AGENT_TIMEOUT_SECONDS,
    retry_reason: str = "",
) -> Dict:
    if not paper_file:
        raise ValueError("paper_file is required for execution.")

    fabscore_judge_dir = _fabscore_judge_dir(task_path, judge_type)
    progress_md_path = osp.join(fabscore_judge_dir, "progress.md")
    command_output_path = _claim_command_output_path(workspace_dir, claim["index"])
    if osp.exists(command_output_path):
        os.remove(command_output_path)
    retry_guidance = ""
    if retry_attempt > 0:
        retry_guidance = (
            f"\nRetry Context:\n"
            f"- This is retry attempt {retry_attempt} for this same claim after a previous timeout.\n"
            f"- Reuse any existing verification artifacts under {workspace_dir} when possible.\n"
            f"- Prefer the smallest decisive execution path for this claim only.\n"
            f"- Avoid unrelated exploratory analysis.\n"
        )
        if retry_reason:
            retry_guidance += f"- Previous attempt issue: {retry_reason}\n"

    prompt = f"""You are an autonomous research execution-verification agent.
Your current task is to verify the claim below using the repository at {task_path}. The overall pipeline will invoke this execution step separately for each claim that requires execution, so in this invocation you should focus only on the current claim below. There is a progress markdown file at {progress_md_path}, where you can see the actions taken in previous sessions. Please append your actions and findings in this file as well, so that future sessions can build on them.

Research Paper File:
{paper_file}

Claim To Verify:
{json.dumps(claim, indent=2, ensure_ascii=False)}

Execution Requirements:
- Verify this claim only.
- Please read the paper and understand the claim in its full context before you decide how to verify it. You may refer to the paper file as needed during verification.
- This claim was already classified as `execution_required`: when evidence is still missing, insufficient, or not specific enough, run the minimal useful command; when existing artifacts already suffice for this exact claim, reuse them without rerunning instead of repeating the same work.
- Analyze the repository at `{task_path}`, and run repository commands from the repository root `{task_path}`. Use `{workspace_dir}` only as the place to store fresh verification artifacts produced during this session.
- Before executing any new command, first inspect existing code and artifacts under `{task_path}`, under `{fabscore_judge_dir}` (for example `execution_log.json`, `fs_execution.json`, and `progress.md`), and under `{workspace_dir}` (for example plots, metrics, checkpoints, or other verification outputs from prior runs).
- If the existing artifacts already provide sufficient and claim-relevant evidence to verify the current claim, reuse them directly and do not rerun the same command to generate them again.
- Reuse an existing artifact only if you can justify that it matches the exact claim. Please do not reuse an existing artifact if it is not clearly relevant to the claim, or if it is unclear whether it corresponds to the claim, or if it is from a different execution path than the one relevant to the claim.
- If you decide to reuse an existing artifact instead of rerunning a command, explicitly state which artifact you reused and why it is sufficient for this claim.
- If you decide to run a command, find the relevant code path, run the minimal useful command, and compare the fresh result with the claim.
- If verification encounters a missing dataset, model, checkpoint, log, cache, trajectory, or other intermediate/generated artifact, do not classify it immediately. First identify the claim-relevant repository-native path that should load or generate the missing artifact, then attempt the minimal reproduction command for that exact blocker, such as `python experiment.py --out_dir ...`, `python train.py ...`, `python main.py ...`, a relevant `run_*.py`, or another project script.
- Distinguish carefully between claim-level objects and supporting artifacts:
  - A missing or unresolvable dataset can be a `data_fabrication` issue if the dataset object itself clearly conflicts with the paper.
  - A missing generated artifact such as `all_results.npy`, a plotting input, a cached metric file, a saved output, or another intermediate result is usually not by itself a fabrication verdict. If the code path is plausible but the supporting evidence remains incomplete after reasonable repo-native attempts, use `Insufficient Evidence` unless you can establish a concrete conflict with the paper.
- After reasonable inspection and execution attempts, classify the claim using the definitions below.
- If execution cannot proceed because of external environment issues, set `verdict` to `Error` in your JSON and give the concrete blocker in `explanation`.
- Save any fresh outputs under `{workspace_dir}` so the run is recorded.
- Do not modify existing source files under `{task_path}`.
- Before you print your final verdict JSON to stdout, you MUST update `{progress_md_path}`.
- Append a new session entry with session purpose `execution`, what files/context you inspected, what execution artifacts you created or updated in this session, a concise verdict summary for this claim, and what the next session should do.
- Use that exact file path. Do not write to a different relative path such as a nested `fabscore_*` directory inside the workspace.
- Do not overwrite previous entries; append only.
- Primary output contract: print the final verdict to stdout as parseable JSON for claim_index `{claim['index']}`—either one object or a one-element array; avoid extra prose (a single ```json``` fence or an outer CLI JSON wrapper is OK if the verdict object is still extractable).
- Create `{command_output_path}` ONLY if you actually run at least one repository command from `{task_path}` (for example `python`, `python3`, `bash`, `make`, etc.). Append the raw stdout and stderr for each such command, with clear separators (command line, then output).
- If you do not run any such command, you MUST NOT create `{command_output_path}` (do not create an empty file).
{retry_guidance}

Print to stdout a JSON object with this shape (or a one-item JSON array containing only that object). For `verdict`, use exactly one label from the list after the field—not the whole `A|B|C` literal:
{{
    "claim_index": {claim['index']},
    "verdict": "<one of: {_execution_verdict_options()}>",
    "evidence_extracted": "Fresh files, values, or concrete blocker used for the verdict",
    "explanation": "Short explanation of the verdict",
    "command_executed": "Exact command or commands run; if missing data or generated artifacts are part of your reasoning, include the attempted regeneration command here",
    "code_reference": "Relevant code path or exact blocker location, preferably file:line or function name",
    "execution_classification": {{
        "fabrication_type": "<one of: data_fabrication, experiment_fabrication, result_fabrication; include when verdict is a fabrication>",
        "reason": "Include when the verdict is Data Fabrication, Experiment Fabrication, or Result Fabrication"
    }}
}}

Use Title Case in `verdict` (e.g. `Data Fabrication`). Use snake_case only in `execution_classification.fabrication_type` (e.g. `data_fabrication`), matching the classification bullets below—not the display labels.

Classification rules for execution-stage outcomes:
- Use `Data Fabrication`, `Experiment Fabrication`, or `Result Fabrication` when execution reveals a concrete repository-side conflict with the paper.
- CRITICAL GUARDRAIL FOR MISSING CODE: If the specific model implementation, experimental setting, or results corresponding to a claim are completely absent from the provided repository, you MUST classify it as `No Code Files` (if completely absent with no trace) or `Insufficient Evidence` (if partial). Do NOT classify it as `Experiment Fabrication` or `Data Fabrication` solely based on the absence of code/artifacts. Fabrication verdicts require concrete, visible code or execution outputs that explicitly contradict the paper.
- Priority rule: `data_fabrication` > `experiment_fabrication` > `result_fabrication`. If multiple fabrication types appear to apply, you MUST assign only the highest-priority one. For example, if a claim has both a data conflict and an experiment conflict, classify it as `data_fabrication`.
- If exactly one fabrication type is supported by the evidence, assign that type directly without applying any extra priority logic.
- In normal reasoning, prefer checking fabrication in this order: data-level conflicts first, then experiment/procedure conflicts, then result-level conflicts.
- Use `data_fabrication` when the blocker is about the dataset itself rather than the experiment implementation. This includes cases where the claimed dataset cannot be imported or resolved, the dataset name cannot be found from the repository's claimed source (for example Hugging Face), the dataset file is corrupted, or the loaded dataset clearly conflicts with the paper in identity, source, size, labels, composition, or splits. A claimed dataset identifier that has never existed in the stated source (for example Hugging Face) should be treated as `data_fabrication`, because the data object itself is fabricated or unsupported. If the paper describes the dataset as a real-world dataset, but the dataset shown in the repository is synthetic data, it should be treated as `data_fabrication`. NOTE: Distinguish this accurately from fraudulent metric implementation (e.g., `np.random` outputs), which should be classified as `experiment_fabrication`. Note that if a dataset name ever exists in the stated source but now the dataset name cannot be found, this may be a transient issue rather than fabrication, and you may classify it as `Insufficient Evidence`; please classify as `data_fabrication` only if the dataset name has never existed in the stated source.
- Use `experiment_fabrication` when the blocker is about the experiment implementation or claimed experimental object rather than the dataset. This includes cases where the claimed model cannot be imported or resolved, the model name cannot be found from the repository's claimed source (for example Hugging Face), the loaded model clearly conflicts with the paper in identity, size, version, or architecture, or the implementation/method/training/inference/evaluation/metric pipeline clearly conflicts with the paper or is internally self-contradictory. A claimed model identifier that has never existed in the stated source (for example Hugging Face) should be treated as `experiment_fabrication`, because the claimed experimental object itself is fabricated or unsupported. Note that if a model name ever exists in the stated source but now the model name cannot be found, this may be a transient issue rather than fabrication, and you may classify it as `Insufficient Evidence`; please classify as `experiment_fabrication` only if the model name has never existed in the stated source.
- Use `result_fabrication` when the claim-relevant implementation matches the paper's description, but the reported values conflict with reproduced results or are internally consistent but externally/mathematically impossible. Use `result_fabrication` when the implementation and data have no conflict with the paper and there is no obvious logic error, but the final numbers fail to match.
  CRITICAL RULE ON SCALE: If the code implementation and logic match the paper, but the final numbers are different or mathematically impossible, you MUST classify this as `result_fabrication` regardless of the scale (e.g., even a 100x difference).
- Use `Insufficient Evidence` when there is a plausible claim-relevant implementation path and some relevant code, execution, or artifact evidence, but the currently available evidence is still not specific, complete, or reliable enough to justify `Verified`, `Data Fabrication`, `Experiment Fabrication`, or `Result Fabrication`.
  GENTLE RULE ON MISSING ARTIFACTS: If a required checkpoint or artifact is missing, but the repository provides a training or regeneration script, you MUST first attempt to run the script to reproduce it. Only if reproduction fails or it is too difficult to reproduce should you conclude `Insufficient Evidence`.
- Do not use `data_fabrication` merely because logs, checkpoints, caches, saved outputs, or plotting artifacts are missing. Those are not datasets. Missing supporting artifacts should not by themselves be treated as fabrication; use `Insufficient Evidence` unless execution establishes a concrete conflict with the paper. If the main blocker is an incompatible model/checkpoint/implementation path, use `experiment_fabrication`. If the blocker is external infrastructure rather than the repository itself, use `Error`.
- Decision rule:
  1. If the problem is which dataset is used or whether the dataset can be loaded and this establishes a concrete conflict with the paper, use `data_fabrication`.
  2. If the problem is which model/method is used or whether the experiment pipeline can be executed as described in the paper, and this establishes a concrete conflict with the paper, use `experiment_fabrication`.
  3. If the code implementation and calculation logic are consistent with the paper but the reported numbers are simply wrong, unsupported, or mathematically impossible, use `result_fabrication`.
  4. If more than one of the above applies, keep only the highest-priority label: `data_fabrication` > `experiment_fabrication` > `result_fabrication`.
  5. If there is some claim-relevant evidence but it remains inconclusive after sufficient analysis or execution and artifact inspection, use `Insufficient Evidence`.
- Examples:
    - Corrupted dataset file, `load_dataset(...)` points to a dataset identifier that has never existed in the stated source (e.g. Hugging Face), or the paper describes using a real-world dataset but the repository implementation uses synthetic data instead => `data_fabrication`.
    - A claimed model identifier has never existed in the stated source (e.g. Hugging Face), the wrong model implementation/architecture is used in evaluation, checkpoint/model pairing is incompatible, metric implementation is fraudulent (e.g., the code bypasses actual execution to **hardcode** a static result or uses synthetic generation (e.g., `np.random`) to fabricate output metrics), or the claimed script crashes due to fundamental implementation defects => `experiment_fabrication`.
    - The repository's code implementation and evaluation logic match the paper's description, but the paper says 0.742 and the actual limit or output of the code is 0.007 (mathematically impossible) => `result_fabrication`.
    - The paper reports a metric as belonging to Model A (Run 5), but it actually matches the saved output of Model B (Run 4). This "mis-mapping" or "cherry-picking" of results from other runs is `result_fabrication`.
- A plotting script is plausible, but the required generated input such as `all_results.npy` is missing and reasonable repo-native attempts still do not produce enough evidence to verify the plotted claim => `Insufficient Evidence`.
- Use `Verified` when claim-relevant existing artifacts or a successful fresh execution provide sufficiently specific evidence that supports the paper's claim.
    - IMPORTANT FOR FIGURES: Finding a pre-existing final image (e.g., .png, .pdf) in the repository is NOT sufficient for a `Verified` verdict. A pre-existing image could be a manually uploaded artifact from a previous experiment. To achieve `Verified`, you MUST either (a) find the underlying raw data files, logs, or metrics (e.g., .npy, .csv, .json, tensor logs) whose values correspond to the visual data plotted in the figure, OR (b) successfully execute a script that re-generates these data files or the image from scratch.
    - PERFORMANCE TIP: When verifying large data files (like `.npy` arrays), focus on confirming their shape, existence, and representative values. Do NOT print the entire contents of large arrays/logs to stdout, as this causes parser errors.
- Reserve `Error` for external execution failures, such as agent/CLI failure, infrastructure issues, missing GPU resources, timeout, or other environment problems that do not indicate fabrication in the repository itself.
"""

    model_cli = ["--model", model_name] if model_name else []
    if judge_type == "claude":
        cli_cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--verbose",
            *model_cli,
        ]
    elif judge_type == "gemini":
        cli_cmd = ["gemini", "--approval-mode", "yolo", "--output-format", "json", *model_cli, prompt]
    else:
        cli_cmd = ["codex", "exec", "--full-auto", "--sandbox", "workspace-write", "--json", *model_cli, prompt]

    try:
        process = subprocess.Popen(
            cli_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=task_path,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            result = subprocess.CompletedProcess(cli_cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            _append_autonomous_execution_log(
                task_path,
                judge_type,
                claim,
                prompt,
                retry_attempt,
                retry_reason,
                stderr=f"Execution timed out after {timeout_seconds}s.",
                returncode=None,
                timed_out=True,
            )
            return _execution_agent_error_verdict(
                claim["index"],
                f"Execution timed out after {timeout_seconds}s.",
            )

        try:
            os.sync()
        except AttributeError:
            pass
        time.sleep(1.0)

        _append_autonomous_execution_log(
            task_path,
            judge_type,
            claim,
            prompt,
            retry_attempt,
            retry_reason,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

        verdicts = _find_verdict_list((result.stdout or "").strip())
        if verdicts:
            picked_verdict = _pick_claim_verdict(verdicts, claim["index"])
            if picked_verdict:
                _apply_command_output_file_policy(command_output_path, picked_verdict)
                return picked_verdict

            received = [v.get("claim_index") for v in verdicts]
            return _execution_agent_error_verdict(
                claim["index"],
                "Agent returned a verdict list, but none of the items matched "
                f"claim_index={claim['index']} (received claim_index values: {received}).",
            )

        rc_note = ""
        if result.returncode not in (0, None):
            rc_note = f" (exit code {result.returncode})"
        err_body = (result.stderr[:1000] if result.stderr else "").strip()
        if not err_body:
            err_body = "No stderr; stdout did not contain a parseable verdict JSON"
            if result.stdout and result.stdout.strip():
                err_body += f" (stdout length {len(result.stdout)})"
        return _execution_agent_error_verdict(
            claim["index"],
            f"Agent failed to provide a valid verdict{rc_note}. {err_body}",
        )
    except Exception as exc:
        return _execution_agent_error_verdict(claim["index"], f"System error: {exc}")


def _execute_claim_with_timeout_retry(
    task_path: str,
    paper_file: str,
    claim: Dict,
    judge_type: str,
    model_name: Optional[str],
    workspace_dir: str,
) -> Dict:
    verdict_data = _autonomous_execute_claim(
        task_path=task_path,
        paper_file=paper_file,
        claim=claim,
        judge_type=judge_type,
        model_name=model_name,
        workspace_dir=workspace_dir,
        retry_attempt=0,
        timeout_seconds=EXECUTION_AGENT_TIMEOUT_SECONDS,
    )

    retry_attempt = 0
    while _is_timeout_error(verdict_data) and retry_attempt < MAX_TIMEOUT_RETRIES:
        retry_attempt += 1
        logging.warning(
            "Claim %s timed out during execution verification. Retrying attempt %s/%s.",
            claim["index"],
            retry_attempt,
            MAX_TIMEOUT_RETRIES,
        )
        verdict_data = _autonomous_execute_claim(
            task_path=task_path,
            paper_file=paper_file,
            claim=claim,
            judge_type=judge_type,
            model_name=model_name,
            workspace_dir=workspace_dir,
            retry_attempt=retry_attempt,
            timeout_seconds=EXECUTION_AGENT_TIMEOUT_RETRY_SECONDS,
            retry_reason="Previous attempt timed out before reaching a decisive claim-level execution verdict.",
        )

    return verdict_data


def _record_claim_result(results: Dict, claim: Dict, verdict_data: Dict) -> None:
    category = claim["category"]
    commands_executed = _extract_commands_executed(verdict_data)
    code_reference = _normalize_code_reference(verdict_data)
    code_references = _parse_code_references(code_reference)
    normalized_verdict = _normalize_verdict(verdict_data)
    entry = {
        "claim_index": claim["index"],
        "claim": claim["content"],
        "verdict": normalized_verdict,
        "explanation": _format_execution_explanation(verdict_data, normalized_verdict),
        "evidence": _build_evidence_text(
            verdict_data.get("evidence_extracted", verdict_data.get("evidence", "")),
            code_reference,
        ),
        "detection_mode": _execution_detection_mode(commands_executed),
        "decision_stage": "execution",
        "verification_process": {
            "files_analyzed": code_references,
            "commands_executed": commands_executed,
        },
    }

    results[category].append(entry)


def _record_execution_stage_error(results: Dict, item: Dict, message: str) -> None:
    category = _normalize_result_category(item.get("category", "results_section"))
    entry = {
        "claim_index": item.get("index"),
        "claim": item.get("content", ""),
        "verdict": "Error",
        "explanation": _format_execution_explanation({"explanation": message}, "Error"),
        "evidence": "N/A",
        "detection_mode": "execution_artifact_analysis",
        "decision_stage": "execution",
        "verification_process": {
            "files_analyzed": [],
            "commands_executed": [],
        },
    }
    results[category].append(entry)


def execute_analyzed_results(
    task_path: str,
    analysis: Dict,
    extracted_results_path: Optional[str] = None,
    paper_file: str = "",
    judge_type: str = "claude",
    model_name: Optional[str] = None,
) -> Dict:
    if not paper_file:
        raise ValueError("paper_file is required for execution.")

    extracted_data = _load_extracted_data(task_path, judge_type, extracted_results_path)
    claim_lookup = _claim_lookup_from_extracted(extracted_data)
    execute_dir = _fabscore_judge_dir(task_path, judge_type)
    workspace_dir = osp.join(execute_dir, "workspace")
    os.makedirs(execute_dir, exist_ok=True)
    os.makedirs(workspace_dir, exist_ok=True)

    progress_path = osp.join(execute_dir, "fs_progress.json")
    prev_progress = _load_progress(progress_path)
    if prev_progress:
        results = prev_progress.get("aggregated", _empty_execution_result())
        completed_claim_indices = set(prev_progress.get("completed_claim_indices", []))
        logging.info(
            f"Resuming execution: {len(completed_claim_indices)} claims already completed. "
            f"Skipping claim indices: {sorted(completed_claim_indices)}"
        )
    else:
        results = _empty_execution_result()
        completed_claim_indices = set()

    execution_claims: List[Dict] = []
    for item in _get_bucket_items(analysis, "execution_required"):
        try:
            index = int(item["index"])
        except (TypeError, ValueError):
            logging.warning("Skipping malformed execution_required item with invalid index: %s", item)
            _record_execution_stage_error(
                results,
                item,
                "Execution stage could not map this execution_required claim because its analysis index was missing or invalid.",
            )
            continue
        claim = claim_lookup.get(index)
        if not claim:
            logging.warning(
                "Execution-required claim index %s was not found in the current extracted claims. "
                "Recording it as an execution-stage error instead of dropping it.",
                index,
            )
            _record_execution_stage_error(
                results,
                item,
                "Execution stage could not map this execution_required claim to the current fs_extracted.json. "
                "This usually means fs_analysis.json and fs_extracted.json are out of sync.",
            )
            continue
        execution_claims.append({
            "index": index,
            "category": claim["category"],
            "content": claim["content"],
            "reason": item.get("reason", "Requires code execution to verify."),
            "candidate_files": item.get("candidate_files", []),
            "suggested_entrypoints": item.get("suggested_entrypoints", []),
        })

    logging.info("Execution stage will verify %s execution-required claims.", len(execution_claims))

    for claim in execution_claims:
        index = claim["index"]

        if index in completed_claim_indices:
            continue

        logging.info(f"Executing claim verification {index}: {claim['content'][:80]}")

        verdict_data = _execute_claim_with_timeout_retry(
            task_path=task_path,
            paper_file=paper_file,
            claim=claim,
            judge_type=judge_type,
            model_name=model_name,
            workspace_dir=workspace_dir,
        )
        _record_claim_result(results, claim, verdict_data)

        completed_claim_indices.add(index)
        _save_progress(progress_path, results, sorted(completed_claim_indices))

    _recompute_summary(results)

    execution_output_path = osp.join(execute_dir, "fs_execution.json")
    with open(execution_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if osp.exists(progress_path):
        os.remove(progress_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution-stage verification for FabScore claims.")
    parser.add_argument("--task_path", required=True, help="Path to the task directory.")
    parser.add_argument("--analysis_path", default=None, help="Path to fs_analysis.json.")
    parser.add_argument("--extracted_path", default=None, help="Path to fs_extracted.json.")
    parser.add_argument("--paper_file", required=True, help="Paper path relative to task_path.")
    parser.add_argument("--judge_type", default="claude", choices=SUPPORTED_JUDGES, help="Judge type.")
    parser.add_argument("--model_name", default=None, help="Specific model name override.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    analysis_path = args.analysis_path or osp.join(
        _fabscore_judge_dir(args.task_path, args.judge_type), "fs_analysis.json"
    )
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    results = execute_analyzed_results(
        task_path=osp.abspath(args.task_path),
        analysis=analysis,
        extracted_results_path=args.extracted_path,
        paper_file=args.paper_file,
        judge_type=args.judge_type,
        model_name=args.model_name,
    )
    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()