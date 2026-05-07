import os
import os.path as osp
import sys
import json
import logging
import argparse
from typing import Optional
from dotenv import load_dotenv

from fabscore.eval.extraction import extract_results
from fabscore.eval.analysis import (
    analyze_extracted_results,
    _flatten_extracted_results,
    _normalize_analysis,
)
from fabscore.eval.execution import execute_analyzed_results
from fabscore.eval.summarization import generate_final_summary
from fabscore.utils.utils import SUPPORTED_JUDGES


load_dotenv()


def fabscore(
    task_path: str,
    judge_type: str = "claude",
    model_name: Optional[str] = None,
    paper_filename: str = "",
    extraction_only: bool = False,
    analysis_only: bool = False,
    execution_only: bool = False,
):
    """
    Run the full FabScore evaluation pipeline on a task folder.

        Pipeline Steps:
            0. **Discovery**       (resolve paper file)
            1. **Extraction**      (extraction.py)
            2. **Analysis**        (analysis.py)
            3. **Execution**       (execution.py)
            4. **Summarization**   (summarization.py)
    """

    logging.info("=" * 60)
    logging.info("FabScore Pipeline Started")
    logging.info("=" * 60)
    logging.info(f"  task_path             : {task_path}")
    logging.info(f"  judge_type            : {judge_type}")
    logging.info(f"  model_name            : {model_name or '(default)'}")
    logging.info(f"  paper_filename        : {paper_filename}")
    logging.info(f"  extraction_only       : {extraction_only}")
    logging.info(f"  analysis_only         : {analysis_only}")
    logging.info(f"  execution_only        : {execution_only}")
    logging.info("=" * 60)

    if judge_type not in SUPPORTED_JUDGES:
        raise ValueError(f"Unknown judge_type: {judge_type}. Must be one of {SUPPORTED_JUDGES}")

    task_path = osp.abspath(task_path)
    if not osp.isdir(task_path):
        raise FileNotFoundError(f"Task path not found: {task_path}")
    if not paper_filename:
        raise ValueError("paper_filename is required. Please provide the paper file name or relative path inside the task directory.")

    # ── Step 0: Discovery ────────────────────────────────────────────────
    output_dir = osp.join(task_path, f"fabscore_{judge_type}")
    extracted_results_file = osp.join(output_dir, "fs_extracted.json")
    
    # Unified paper discovery for all steps
    resolved_paper_path = paper_filename if osp.isabs(paper_filename) else osp.join(task_path, paper_filename)
    resolved_paper_path = osp.abspath(resolved_paper_path)
    if not osp.exists(resolved_paper_path):
        raise FileNotFoundError(
            f"Could not resolve paper_filename '{paper_filename}' under task path {task_path}."
        )
    if not osp.isfile(resolved_paper_path):
        raise FileNotFoundError(f"Paper path is not a file: {resolved_paper_path}")

    paper_file_rel = osp.relpath(resolved_paper_path, task_path)

    # ── Step 1: Extraction ──────────────────────────────────────────────
    if osp.exists(extracted_results_file):
        logging.info(f"Step 1/4: Using existing extraction results: {extracted_results_file}")
    else:
        logging.info(f"Step 1/4: Extracting experimental results using {judge_type}...")
        logging.info(f"  Using Paper: {resolved_paper_path}")
        extract_results(task_path=task_path, paper_filename=paper_filename, judge_type=judge_type, model_name=model_name)
        
        if not osp.exists(extracted_results_file):
            raise RuntimeError(f"Extraction failed to create {extracted_results_file}")
        logging.info(f"  ✓ Extraction complete.")

    with open(extracted_results_file, 'r', encoding='utf-8') as f:
        extracted_results = json.load(f)

    if extraction_only:
        logging.info("Extraction-only mode complete.")
        return extracted_results

    # ── Step 2: Analysis ────────────────────────────────────────────────
    logging.info(f"Step 2/4: Running static analysis using {judge_type}...")
    analysis_path = osp.join(output_dir, "fs_analysis.json")

    if osp.exists(analysis_path):
        logging.info(f"  Using existing analysis: {analysis_path}")
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        claim_lookup = {
            item["index"]: item
            for item in _flatten_extracted_results(extracted_results)
        }
        normalized_analysis = _normalize_analysis(analysis, claim_lookup)
        if normalized_analysis != analysis:
            logging.warning(
                "Existing analysis file did not match the current normalized schema or extracted claims. "
                "Rewriting %s with a normalized version.",
                analysis_path,
            )
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(normalized_analysis, f, indent=4, ensure_ascii=False)
        analysis = normalized_analysis
    else:
        analysis = analyze_extracted_results(
            task_path=task_path,
            paper_file=paper_file_rel,
            extracted_results_path=extracted_results_file,
            output_analysis_path="fs_analysis.json",
            judge_type=judge_type,
            model_name=model_name,
        )
        logging.info("  ✓ Analysis complete.")

    if analysis_only:
        logging.info("Analysis-only mode complete.")
        return analysis

    # ── Step 3: Execution ───────────────────────────────────────────────
    logging.info(f"Step 3/4: Executing runtime verification for execution-required claims using {judge_type}...")
    result = execute_analyzed_results(
        task_path=task_path,
        analysis=analysis,
        extracted_results_path=extracted_results_file,
        paper_file=paper_file_rel,
        judge_type=judge_type,
        model_name=model_name,
    )
    logging.info("  ✓ Execution stage complete.")

    if execution_only:
        logging.info("Execution-only mode complete.")
        return result

    # ── Step 4: Summarization (final metrics and breakdowns) ────────────
    logging.info("Step 4/4: Generating final assessment summary...")
    result = generate_final_summary(analysis, result)

    # Persist the final merged summary artifact.
    summary_path = osp.join(output_dir, "fs_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    assessment = result.get("final_assessment", {})
    logging.info(f"  ✓ Final results saved to {summary_path}")
    logging.info(f"  FabScore: {assessment.get('fab_score', 0):.2%} ({assessment.get('verified', 0)}/{assessment.get('total_claims', 0)} claims verified)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="FabScore — Automated verification of experimental results in research papers."
    )
    parser.add_argument("--task_path", required=True, help="Path to the task root directory.")
    parser.add_argument("--paper_filename", required=True,
                        help="Paper filename or relative path inside the task directory.")
    parser.add_argument("--judge_type", default="claude", choices=SUPPORTED_JUDGES,
                        help="Judge to use for extraction and verification (default: claude).")
    parser.add_argument("--model_name", default=None, help="Model name override for the judge CLI (e.g. claude-sonnet-4-20250514).")
    parser.add_argument("--extraction_only", action="store_true",
                        help="Run extraction only, skip analysis, verification, and summarization.")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Run extraction and analysis only, skip verification and summarization.")
    parser.add_argument("--execution_only", action="store_true",
                        help="Run through the execution stage and skip final summarization writeout.")

    args = parser.parse_args()

    only_modes = [args.extraction_only, args.analysis_only, args.execution_only]
    if sum(bool(mode) for mode in only_modes) > 1:
        parser.error("--extraction_only, --analysis_only, and --execution_only are mutually exclusive.")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"fabscore_{args.judge_type}.log"),
            logging.StreamHandler(),
        ],
    )

    try:
        result = fabscore(
            task_path=args.task_path,
            judge_type=args.judge_type,  
            model_name=args.model_name,
            paper_filename=args.paper_filename,
            extraction_only=args.extraction_only,
            analysis_only=args.analysis_only,
            execution_only=args.execution_only,
        )

        should_print_summary = not any([
            args.extraction_only,
            args.analysis_only,
            args.execution_only,
        ])
        if result and should_print_summary:
            assessment = result.get("final_assessment", {})
            print("\n" + "=" * 60)
            print("  FabScore Verification Summary")
            print("=" * 60)
            print(f"  Score:       {assessment.get('fab_score', 0):.2%}")
            print(f"  Verified:    {assessment.get('verified', 0)}")
            print(f"  Data Fabrication:       {assessment.get('data_fabrication', 0)}")
            print(f"  Experiment Fabrication: {assessment.get('experiment_fabrication', 0)}")
            print(f"  Result Fabrication:     {assessment.get('result_fabrication', 0)}")
            print(f"  No Code Files:           {assessment.get('no_code_files', 0)}")
            print(f"  Insufficient Evidence:  {assessment.get('insufficient_evidence', 0)}")
            print(f"  Error:       {assessment.get('error', 0)}")
            print(f"  Total:       {assessment.get('total_claims', 0)}")
            print("=" * 60)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
