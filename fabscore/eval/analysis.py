import os.path as osp
import json
import logging
import argparse
from typing import Callable, Dict, List, Optional

from fabscore.judges.claude import run_claude_judge
from fabscore.judges.codex import run_codex_judge
from fabscore.judges.gemini import run_gemini_judge
from fabscore.utils.utils import SUPPORTED_JUDGES, _get_bucket_items


ANALYSIS_PROMPT_TEMPLATE = """You are an expert AI research code auditor.
Your task is to perform STATIC ANALYSIS only. Do not run experiments.

You are given:
1. The extracted numerical claims from the paper.
2. The paper file path.
3. The repository for the paper, which may contain code, scripts, data pipelines, logs, checkpoints, and other artifacts that could support or contradict the claims.
4. A progress markdown file at `{progress_md_path}`, where you can see the actions taken in previous sessions. Please append your actions and findings in this file as well, so that future sessions can build on them.

Research Paper File:
{paper_file}

Extracted Claims:
{results_json}

Repository Path:
{task_path}


Important: When analyzing code that implements probabilistic data augmentation or randomization, you MUST carefully distinguish between mutually exclusive (if/elif) and independent (if+if) probability implementations. For example, in an if/elif structure, each branch is exclusive and the probability for each augmentation is as specified. In an independent if+if structure, each augmentation is applied independently, so the probability of both being applied can be nonzero. Always base your probability judgment on the actual code logic, and explain your reasoning.

You must read the paper and analyze the repository, and then classify EVERY claim into exactly one of these buckets:

1. `no_code_files`
Meaning: there is no identifiable code, script, data pipeline, or claim-relevant execution record in the repository that could plausibly support this result. Use this only when the reported result appears unsupported because you cannot find any corresponding implementation path and you also cannot find any claim-relevant artifact or run output at all.
- Example situation:
    - There is neither a matching script or implementation path nor any claim-relevant artifact such as logs, saved outputs, checkpoints, or metric files. This should be `no_code_files`.

2. `obvious_hallucination`
Meaning: the repository does contain related code, but static inspection already reveals a clear and concrete conflict with the paper, so the reported result is unreliable or fabricated.
You MUST assign one `fabrication_type`:
- Priority rule: `data_fabrication` > `experiment_fabrication` > `result_fabrication`. If multiple fabrication types appear to apply, you MUST assign only the highest-priority one. For example, if a claim has both a data conflict and an experiment conflict, classify it as `data_fabrication`.
- If exactly one fabrication type is supported by the evidence, assign that type directly without applying any extra priority logic.
- In normal reasoning, prefer checking fabrication in this order: data-level conflicts first, then experiment/procedure conflicts, then result-level conflicts.
- `data_fabrication`: use this only when static evidence is already sufficient to establish a concrete conflict between the claim and the data object used by the code, including clear conflicts in dataset identity, composition, source, labels, or splits. For example:
    - Dataset names clearly conflict between the paper and the repository. For example, the dataset name used in the repository is `data1` while the paper describes it as `data2`.
    - Dataset sources clearly conflict between the paper and the repository. For example, the paper describes the dataset as a real-world dataset, but the dataset shown in the repository is synthetic data.
    - Dataset size or splits clearly conflict between the paper and the repository. For example, the dataset shown in the repository is a dataset with 10 samples, but the paper describes it as a dataset with 1000 samples.
    - A claimed dataset identifier has never existed in the stated source (for example Hugging Face), so the data object itself is fabricated or unsupported.
- `experiment_fabrication`: use this only when static evidence is already sufficient to establish a concrete conflict between the paper and the experimental object or procedure in repository's implementation. Identify a claim as `experiment_fabrication` if you find:
    - Experimental setup clearly conflicts between the paper and the repository. For example, the paper describes one training or inference setup, but the repository uses a different setup.
    - Model implementation clearly conflicts between the paper and the repository. For example, the paper claims model `A`, but the repository actually implements model `B`.
    - Evaluation protocol clearly conflicts between the paper and the repository. For example, the paper describes one evaluation procedure, benchmark setting, or comparison protocol, but the repository implements a different one.
    - The repository provides the claimed model but its performance or internal state clearly disagrees with the paper's description (e.g., paper says it's a 100M model, but code is 10M). (Note: Wrongly reporting a different run's result for a correctly implemented model is `result_fabrication`).
    - There are obvious logical inconsistencies in the experimental procedure.
    - Evaluation Metric Implementation Error: The code includes obvious and deliberate malfeasance or logical traps to hallucinate results, such as bypassing the actual model execution to **hardcode** a static number as the final evaluation result, or using synthetic generation (e.g., `np.random`) to fabricate output metrics.
    - A claimed model identifier has never existed in the stated source (for example Hugging Face), so the claimed experimental object itself is fabricated or unsupported.
- If the issue is about what data object is used or how the data is defined/composed/split and static evidence already establishes a concrete conflict, classify it as `data_fabrication`. If the data object is not the issue but the training, evaluation, model, or metric computation statically establishes a concrete conflict, classify it as `experiment_fabrication`.
- `result_fabrication`: use this when static analysis shows that the **output values** reported in the paper conflict with the actual results produced by the repository.
  CRITICAL RULE ON SCALE: If the repository's code implementation, evaluation logic, and experimental settings appear internally consistent and match the methodology described in the paper, but the final reported numbers are simply different (even if they are 100x larger or mathematically impossible given the formula), you MUST classify this as `result_fabrication`. Use `result_fabrication` ONLY when the implementation and data have no conflict with the paper and there is no obvious logic error, but the final reproduced numbers explicitly fail to match the paper, OR when the paper reports metrics from a DIFFERENT model/run than the one specified (a "mis-mapping" or "cherry-picking" of results). If the paper mentions which code file implements the exact claim, please analyze the specific code file. For example:
    - The paper claims the result is from Model A (Run 5), but you find the result actually matches the output of Model B (Run 4) in the repository. This is a deliberate mis-mapping of result metrics, so it is `result_fabrication`.
    - The paper mentions that `run5.py` implements the exact claim, but you find the execution result of `run4.py` is the same as the reported result, while the execution result of `run5.py` is different from the reported result. You shoud classify the claim as `result_fabrication`.

3. `static_verifiable`
Meaning: STATIC inspection alone is sufficient to conclude that the claim is reproducible or consistent with the implementation, without needing to run code. IMPORTANT: For figures, finding only a generated final image artifact (e.g. .png, .pdf) is NOT sufficient. You MUST also find the underlying data files or metrics (e.g. .csv, .npy, .json) whose values correspond to the data plotted in the figure to map it as `static_verifiable`.

4. `insufficient_evidence`
Meaning: there is some claim-relevant evidence in the repository, such as logs, saved outputs, checkpoints, cached metrics, or other artifacts, but static inspection still cannot establish a reliable support chain from the repository to the exact paper claim, and there is no suitable code path, entrypoint, or intact implementation available for the next verification step. Use this only when the currently available evidence is too incomplete, indirect, ambiguous, or provenance-unclear to justify `static_verifiable`, `obvious_hallucination`, or `no_code_files`, and when no execution can be run to verify the claim.
If a claim depends on a missing plotting input or intermediate artifact (such as all_results.npy), and the repository contains a plotting script, evaluation script, training script, or any plausible upstream code path that may regenerate it, do NOT classify the claim as insufficient_evidence at analysis stage. In that case, classify it as execution_required and let execution attempt the repo-native regeneration path first.
Use insufficient_evidence only after you have determined that no suitable repo-native code path exists for generating the missing claim-relevant artifact.
- Example situations:
    - For example, there is a `final_info.json`, log file, or saved metric table containing claim-relevant numbers, but you cannot find the corresponding script or implementation that generated those numbers. This should be `insufficient_evidence`, not `no_code_files`.
    - If a pre-generated figure artifact exists but there is also a plotting or regeneration script that can still be run to test the claim, do NOT use `insufficient_evidence`; use `execution_required`.

5. `execution_required`
Meaning: the repository appears to contain a plausible execution path, but static inspection alone is not enough; code execution is required to determine whether the claim is reproducible.
- Use this when the remaining uncertainty is genuinely execution-dependent and a repo-native run still has a realistic chance to resolve it.
- If the issue is missing data, missing checkpoints, missing logs, missing caches, missing `all_results.npy`, or any other missing intermediate/generated artifact, do NOT classify it as fabrication at analysis stage solely because it is absent. Instead, classify it as `execution_required` when the repository still appears able to regenerate the needed evidence through its own execution path. Otherwise, if no such path exists, classify it as `no_code_files`.

6. `error`
Meaning: use this only when the analysis output itself is incomplete or malformed for a claim, such as when the claim was not classified reliably due to judge/output issues. Do not use this as a scientific judgment about repository support.

Rules:
- CRITICAL GUARDRAIL FOR MISSING CODE: If the specific model implementation, experimental setting, or results corresponding to a claim are completely absent from the provided repository, you MUST classify it as `no_code_files` or `insufficient_evidence`. Do NOT classify it as `experiment_fabrication` or `data_fabrication` solely based on the absence of code/artifacts. Fabrication verdicts require concrete, visible code or artifacts that explicitly contradict the paper.
- Perform STATIC ANALYSIS only. Do not execute commands.
- Read the paper content using the paper file path directly.
- Every claim index must appear exactly once across the six buckets.
- Keep stage boundaries clear: analysis is allowed to inspect existing files and static repository structure, but it must not assume the outcome of a future regeneration attempt that has not been executed yet.
- If a result looks invented, first check whether there is a corresponding implementation path and whether there are any claim-relevant artifacts or run outputs. Use `no_code_files` only when both are absent. If some claim-relevant artifact or execution record exists but the code path is missing or unclear, prefer `insufficient_evidence` rather than `no_code_files`.
- If static inspection already reveals a clear experiment-side defect, such as metric misuse, a broken training or evaluation procedure, failed control flow, model or experimental-variant mismatch, a statistical formula error, or a wrong default output directory that mixes runs or variants, classify the claim immediately as `obvious_hallucination` with `experiment_fabrication`.
- Use `insufficient_evidence` only when there is some claim-relevant artifact or execution record, but static inspection cannot establish a trustworthy provenance chain for the exact claim and there is no suitable repo-native code path or entrypoint left for meaningful next-step verification. If no such artifact or code path exists at all, use `no_code_files`.
- Use `execution_required` whenever the remaining uncertainty is genuinely execution-dependent and there is still a plausible repo-native code path to run, such as when the relevant result has not been generated yet, a dataset or an intermediate artifact is missing but appears plausibly reproducible from the repository's own code path, or fresh execution is needed to determine the actual output value.
- If the issue is about what data is used or how the data is defined, composed, labeled, sourced, or split, use `data_fabrication` only when static evidence already establishes a concrete conflict. If the data is the same but the training setup, model selection, evaluation protocol, control flow, or metric computation statically establishes a concrete conflict, use `experiment_fabrication`.
- If both a higher-priority and lower-priority fabrication type are supported by the evidence, always keep only the higher-priority one according to `data_fabrication` > `experiment_fabrication` > `result_fabrication`.
- If the issue is missing data, missing checkpoints, missing logs, missing caches, missing `all_results.npy`, or any other missing intermediate/generated artifact, do NOT classify it as fabrication (including `data_fabrication` or `experiment_fabrication`) at analysis stage solely because it is absent. Classify it as `execution_required` if the repository still appears able to regenerate the needed evidence and there is a suitable code path for doing so, but use `insufficient_evidence` only if the repository already contains partial artifacts or outputs whose provenance remains too unclear for static resolution and there is no suitable code path for further claim-level verification. If no claim-relevant artifacts or code paths exist, use `no_code_files`.
- Use analysis-stage `result_fabrication` only when static evidence is already decisive and further execution is not needed or not meaningful for resolving the exact claim. If there is still a suitable repo-native code path that can directly test the claim, prefer `execution_required` rather than analysis-stage `result_fabrication`.
- If a figure, table, or saved artifact looks suspicious but there is still a repo-native plotting, evaluation, or regeneration path that can directly test the claim, do NOT stop at analysis; classify it as `execution_required`.
- Use `obvious_hallucination` only when the static evidence is strong.
- Use `error` only for analysis-stage output failures, not for repository-side scientific judgments.
- For each item, cite concrete repository evidence such as filenames, scripts, functions, or missing components.
- For `static_verifiable`, explain why static inspection is already sufficient.
- For `insufficient_evidence`, explain why the existing evidence is claim-relevant but still too incomplete or provenance-unclear for a reliable conclusion, and why additional execution is unlikely to resolve it, including whether the next-step code path is missing, incomplete, or not identifiable.
- For `execution_required`, provide likely entrypoints or candidate files when possible. If you cannot identify any suitable next-step code path, do not use `execution_required`.
- When you defer a claim because required supporting artifacts are absent, explicitly say that execution must first attempt to regenerate those artifacts with the repository's own code before any fabrication verdict is made.

Your analyzing process might be like this:
1. First, read the paper and understand the claims.
2. For each claim, check the paper if there are specific code files mentioned, and look for corresponding implementation paths in the repository.
3. If there is no corresponding implementation path and no claim-relevant artifact or execution record at all, classify the claim as `no_code_files`.
- For example, if you cannot find either the relevant code or any claim-relevant logs, outputs, checkpoints, or saved metrics, use `no_code_files`.
4. If there is a corresponding implementation path, analyze the code and static artifacts. If static inspection already establishes a concrete conflict between the paper and the code or artifacts, classify the claim as `obvious_hallucination` with a specific `fabrication_type`.
- For example, if the conflict is about what data object is used or how the data is defined/composed/split, classify it as `data_fabrication`.
- If the data object is not the issue but the training, evaluation, model selection, or metric computation statically establishes a concrete conflict, classify it as `experiment_fabrication`.
- If both apply, keep only the higher-priority label: `data_fabrication` > `experiment_fabrication` > `result_fabrication`.
- If static evidence is already decisive for the reported output itself and there is no suitable remaining repo-native code path to directly test the exact claim, classify it as `result_fabrication`. Otherwise prefer `execution_required`.
5. If there is some claim-relevant evidence but its provenance or mapping to the exact claim remains too unclear for a reliable conclusion, and additional execution is unlikely to resolve that ambiguity because the next-step code path is missing, incomplete, or not identifiable, classify it as `insufficient_evidence`.
- For example, if you can see claim-relevant run outputs or saved metrics, but you cannot find the script or implementation that produced them, use `insufficient_evidence` rather than `no_code_files`.
6. If there is any suitable repo-native code path that can still meaningfully test the claim, classify it as `execution_required` and cite the relevant repository paths and likely entrypoints for execution, even if the current static evidence suggests a possible mismatch.
- For example, if a figure PNG already exists but there is a plotting script or data-generation path that can still be run to verify whether that figure matches the paper, use `execution_required` rather than `insufficient_evidence`.
7. If static inspection is already sufficient to verify the claim, classify it as `static_verifiable` and explain why. For figures, explicitly state which underlying data file matches the plotted data.
8. If the analysis output is incomplete or malformed for a claim, classify it as `error` and explain that the analysis output was unreliable for this claim.

Requirements of progress.md:
- You are running in an independent coding-agent session.
- You MUST update `{progress_md_path}`.
- If the file does not exist, create it.
- Append a new session entry containing:
  - session purpose: `analysis`
  - what files/context you inspected
  - what JSON files you created or updated in this session
  - a concise summary of the classifications you made
  - the recommended next step for the next session
- Do not overwrite previous session entries; append a new dated section.

You MUST save a JSON object to `fabscore_{judge_type}/{output_analysis_path}` with this schema:
{{
    "no_code_files": [
    {{
      "index": 1,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "...",
      "code_evidence": ["path/or/symbol", "..."]
    }}
  ],
  "obvious_hallucination": [
    {{
      "index": 2,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "...",
      "fabrication_type": "data_fabrication|experiment_fabrication|result_fabrication",
      "code_evidence": ["path/or/symbol", "..."]
    }}
  ],
  "static_verifiable": [
    {{
      "index": 3,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "...",
      "code_evidence": ["path/or/symbol", "..."]
    }}
  ],
  "insufficient_evidence": [
    {{
      "index": 4,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "...",
      "code_evidence": ["path/or/symbol", "..."]
    }}
  ],
  "execution_required": [
    {{
      "index": 5,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "...",
      "candidate_files": ["path/to/file.py"],
      "suggested_entrypoints": ["python eval.py --config ..."]
    }}
  ],
  "error": [
    {{
      "index": 6,
      "category": "table|figure|results_section",
      "content": "...",
      "reason": "Analysis output was incomplete or malformed for this claim.",
      "code_evidence": []
    }}
  ],
  "summary": {{
    "total_claims": 0,
    "no_code_files": 0,
    "obvious_hallucination": 0,
    "data_fabrication": 0,
    "experiment_fabrication": 0,
    "result_fabrication": 0,
    "static_verifiable": 0,
    "insufficient_evidence": 0,
    "execution_required": 0,
    "error": 0
  }}
}}

Return valid JSON only through the file write. Do not rely on stdout as the final artifact.
"""

ANALYSIS_BUCKETS = (
    "no_code_files",
    "obvious_hallucination",
    "static_verifiable",
    "insufficient_evidence",
    "execution_required",
    "error",
)

JUDGE_RUNNERS: Dict[str, Callable[..., object]] = {
    "claude": run_claude_judge,
    "codex": run_codex_judge,
    "gemini": run_gemini_judge,
}


def _flatten_extracted_results(results: Dict) -> List[Dict]:
    flat_claims: List[Dict] = []
    idx_counter = 1
    for category_key, category_name in (
        ("tables", "table"),
        ("figures", "figure"),
        ("results_section", "results_section"),
    ):
        for item in results.get(category_key, []):
            flat_claims.append({
                "index": idx_counter,
                "category": category_name,
                "content": item,
            })
            idx_counter += 1
    return flat_claims


def _normalize_analysis(data: Dict, claim_lookup: Dict[int, Dict]) -> Dict:
    normalized = {
        "no_code_files": [],
        "obvious_hallucination": [],
        "static_verifiable": [],
        "insufficient_evidence": [],
        "execution_required": [],
        "error": [],
        "summary": {
            "total_claims": len(claim_lookup),
            "no_code_files": 0,
            "obvious_hallucination": 0,
            "data_fabrication": 0,
            "experiment_fabrication": 0,
            "result_fabrication": 0,
            "static_verifiable": 0,
            "insufficient_evidence": 0,
            "execution_required": 0,
            "error": 0,
        },
    }
    bucket_assignments: Dict[int, List[str]] = {}
    chosen_items: Dict[int, Dict] = {}
    summary = normalized["summary"]

    for bucket in ANALYSIS_BUCKETS:
        bucket_items = _get_bucket_items(data, bucket)

        for raw_item in bucket_items:
            try:
                index = int(raw_item.get("index"))
            except (TypeError, ValueError):
                continue

            claim = claim_lookup.get(index)
            if not claim:
                continue

            item = dict(raw_item)
            item["index"] = index
            item.setdefault("category", claim["category"])
            item.setdefault("content", claim["content"])
            bucket_assignments.setdefault(index, []).append(bucket)
            chosen_items.setdefault(index, item)

    for index, claim in claim_lookup.items():
        assigned_buckets = list(dict.fromkeys(bucket_assignments.get(index, [])))
        if len(assigned_buckets) > 1:
            normalized["error"].append({
                "index": index,
                "category": claim["category"],
                "content": claim["content"],
                "reason": (
                    "The analysis session assigned this claim to multiple buckets: "
                    + ", ".join(assigned_buckets)
                    + ". It was converted to an analysis-stage error."
                ),
                "code_evidence": [],
            })
            continue

        if len(assigned_buckets) == 1:
            bucket = assigned_buckets[0]
            normalized[bucket].append(chosen_items[index])
            continue

        normalized["error"].append({
            "index": index,
            "category": claim["category"],
            "content": claim["content"],
            "reason": "The analysis session did not classify this claim explicitly, so it was backfilled as an analysis-stage error rather than no_code_files.",
            "code_evidence": [],
        })

    for bucket in ANALYSIS_BUCKETS:
        summary[bucket] = len(normalized[bucket])

    for item in normalized["obvious_hallucination"]:
        fabrication_type = item.get("fabrication_type")
        if fabrication_type in summary:
            summary[fabrication_type] += 1

    return normalized


def analyze_extracted_results(
    task_path: str,
    paper_file: str,
    extracted_results_path: Optional[str] = None,
    output_analysis_path: str = "fs_analysis.json",
    judge_type: str = "claude",
    model_name: Optional[str] = None,
    wait_seconds: Optional[int] = None,
) -> Dict:
    if not paper_file:
        raise ValueError("paper_file is required for analysis.")

    if extracted_results_path is None:
        extracted_results_path = osp.join(task_path, f"fabscore_{judge_type}", "fs_extracted.json")

    with open(extracted_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_runner = JUDGE_RUNNERS.get(judge_type)
    if judge_runner is None:
        raise ValueError(f"Unknown judge: {judge_type}")

    flat_claims = _flatten_extracted_results(results)
    claim_lookup = {item["index"]: item for item in flat_claims}
    results_json = json.dumps(flat_claims, indent=2, ensure_ascii=False)
    progress_md_path = osp.join(task_path, f"fabscore_{judge_type}", "progress.md")

    prompt_params = {
        "paper_file": paper_file,
        "results_json": results_json,
        "task_path": task_path,
        "progress_md_path": progress_md_path,
        "judge_type": judge_type,
        "output_analysis_path": output_analysis_path,
    }

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(**prompt_params)

    output_relpath = osp.join(f"fabscore_{judge_type}", output_analysis_path)
    log_filename = "analysis_log.json"
    log_path = osp.join(task_path, osp.dirname(output_relpath) or ".", log_filename)

    judge_runner(
        task_path=task_path,
        paper_file=paper_file,
        output_path=output_relpath,
        prompt=prompt,
        model_name=model_name,
        wait_seconds=wait_seconds,
        output_dir=None,
        skip_if_dir_exists=False,
        log_filename=log_filename,
    )

    if osp.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            raw_log = f.read()
        try:
            agent_output = json.loads(raw_log)
        except json.JSONDecodeError:
            agent_output = raw_log
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "agent_output": agent_output}, f, indent=4, ensure_ascii=False)

    with open(osp.join(task_path, output_relpath), "r", encoding="utf-8") as f:
        analysis = json.load(f)

    normalized = _normalize_analysis(analysis, claim_lookup)
    with open(osp.join(task_path, output_relpath), "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static analysis over extracted FabScore claims.")
    parser.add_argument("--task_path", required=True, help="Path to the task directory.")
    parser.add_argument("--paper_file", required=True, help="Paper path relative to task_path.")
    parser.add_argument("--extracted_results_path", default=None, help="Path to fs_extracted.json.")
    parser.add_argument("--judge_type", default="claude", choices=SUPPORTED_JUDGES, help="Judge type.")
    parser.add_argument("--model_name", default=None, help="Specific model name override.")
    parser.add_argument("--wait_seconds", type=int, default=None, help="Optional override for analysis output wait time.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    analysis = analyze_extracted_results(
        task_path=osp.abspath(args.task_path),
        paper_file=args.paper_file,
        extracted_results_path=args.extracted_results_path,
        judge_type=args.judge_type,
        model_name=args.model_name,
        wait_seconds=args.wait_seconds,
    )
    print(json.dumps(analysis, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()