import os.path as osp
import logging
import argparse
from typing import Callable, Dict, Optional

from fabscore.utils.utils import SUPPORTED_JUDGES

from fabscore.judges.claude import run_claude_judge
from fabscore.judges.codex import run_codex_judge
from fabscore.judges.gemini import run_gemini_judge

extract_results_prompt = """You are an expert AI research paper reviewer.
Your task is to extract the experimental results reported in a paper.
Please extract the results point by point from three sources: tables, figures, and claims in the results section.
The paper is provided as a file in the workspace. It may be a PDF or a Markdown document. Read the specified paper file directly instead of expecting its contents to be pasted into the prompt. If the paper is Markdown, image files referenced from that Markdown are available in the workspace. Mathematical formulas may appear as Unicode symbols (e.g., θ for theta, ∑ for summation, ∝ for proportional to, ∥ for norm) rather than LaTeX notation. Please interpret them accordingly.
INSTRUCTIONS:
- For tables, extract each numerical value along with its corresponding row and column headers. The recording format should be like “1. Table X, row name, column name: <value>”. Please only record the table number. No need to record row number and column number.
- For figures:
     Inspect the paper file directly.
     1. **Extract Model Output**:
         - If the paper file is Markdown, for each figure mentioned in the text or embedded as an image in the Markdown, extract its image file path (e.g., "images/figure1.png").
         - If the paper file is PDF, record the figure by its figure label and associated caption from the paper itself (e.g., "Figure 1", "Fig. 2", "Figure 3(a)"). Do NOT invent workspace image paths when the source is only a PDF.
     2. **Recording Format**:
         - If the paper file is Markdown: "1. images/figure1.png: [Caption]"
         - If the paper file is PDF: "1. Figure 1: [Caption]"
     3. **Deduplication Rule**:
        - Please read the paper carefully when you extract figures. When the same figure is reused in multiple places, keep a single entry using the earliest explicit figure reference or the canonical image path/figure label.
- For claims in the results section, please extract numerical results from the **text body** of the Results/Experiments section (and its subsections) following these guidelines:
1. What to extract:
    - Extract quantitative results that are model outputs or performance metrics (e.g., accuracy, loss, F1 score, precision, recall, speed improvements, error rates).
    - Include exact numerical values: "The accuracy is 85.3%".
    - Include comparative numerical values: "We improved F1 score by 2 points".
    - Include numerical values written as words: "eighty-five percent" or "five percentage points".
    - Split multiple results in one sentence into separate bullet points.
    - Preserve the original wording and meaning as much as possible.
    - Prefer copying the original sentence span or the minimal exact clause from the paper instead of paraphrasing it.
    - Do NOT rewrite, normalize, summarize, or restate a result in your own words unless a tiny trim is needed to isolate one numerical claim from a longer sentence.
    - If you split multiple numerical claims from one sentence, each extracted item should still remain as close as possible to the exact original wording of the relevant clause.
2. What NOT to extract:
    - Do NOT extract qualitative statements without valid performance numerical values: "Our method performs better", "significant improvement", "dropped rapidly", "became negligible", "near-perfect". 
    - **CRITICAL**: If a sentence describes a result qualitatively (e.g., "error became negligible") and the ONLY numbers in that sentence are setup parameters or trial indices (e.g., "by epoch 5", "in trial 3", "for batch size 64"), you MUST skip it. A setup number does NOT count as a performance metric.
    - Do NOT extract figure or table captions. A CAPTION is the official label identifying a figure/table (e.g., the line "Figure 1: Training loss..."). You MUST ignore these lines entirely. Even if a caption contains a revolutionary result not found anywhere else, YOU MUST DISCARD IT. Do not hallucinate that a caption's result exists in the body text if it's not actually there.
    - Do NOT extract experimental setup, implementation details, or hyperparameters, such as:
      * Training configuration: "epoch 5", "trained for 100 epochs", "3 random seeds", "50 training steps"
      * Hyperparameters: "batch size 32", "learning rate 0.001", "lr = 10^-3", "dropout 0.1"
      * Data setup: "N = 1000 samples", "We sample 50 examples from each dataset", "1k train / 200 test"
      * Hardware: "H100 x 8 GPUs", "trained on 4 A100s"
      * Architecture details: "two hidden layers of size 128", "embedding dimension 768"
      * Meta-update/schedule parameters: "meta-update every T = 10 steps", "K = 20 meta-batches"
    - Only extract metrics/outputs they GET from running experiments, not values they SET.
    - Do NOT extract the same numerical result mentioned in multiple sentences repetitively; only extract it once.
3. Source sections:
    - Please only focus on the Results/Experiments section and its subsections.

IMPORTANT — Extraction order and deduplication across tables, figures, and results_section:
You MUST follow this strict three-step procedure:
  Step 1: First, extract all "tables" entries.
  Step 2: Then, extract all "figures" entries.
  Step 3: Finally, extract "results_section" claims from the text body. In this step, you MUST cross-check every candidate claim against the tables you already extracted:
    - If a numerical value in the text is **already captured** in a table entry (even if the wording differs), do NOT include it in "results_section".
    - If Table 2 already contains "Model X, Accuracy: 90%", and the text says "Model X achieves 90% accuracy", do NOT add this to "results_section" because it duplicates the table entry.
    - **Note on Figures**: You do NOT need to cross-check against figures. If a claim restates or summarizes data shown in a figure, you SHOULD still include it in "results_section" if it is a valid numerical result mentioned in the text.

Global deduplication rules:
  - If two candidate entries are semantically the same result, keep only one.
  - Treat entries as duplicates when they report the same underlying experiment outcome, metric/value pair, or the same figure content, even if the wording, sentence location, or caption text differs slightly.
  - Prefer the earliest occurrence in the paper, and prefer the more specific/canonical wording when choosing which duplicate to keep.
  - Never output the same figure twice just because it is referenced in two places.
  - Never output the same textual numerical result twice just because it appears once in a paragraph and again in a summary sentence.

Your extracted results should be saved in a JSON file named "fs_extracted.json" with the following format:
```json
{
  "tables": [
    "1. Table 2, Model X, Accuracy: 90%",
    ...
  ],
  "figures": [
        "1. images/figure1.png: [Caption]",
        "2. Figure 2: [Caption]",
        "3. Figure 3(a): [Caption]",
    ...
  ],
  "results_section": [
    "1. We improved the F1 score by 2 points",
    ...
  ]
}
```
- Hard constraints for `results_section`:
1. Every extracted item in "results_section" MUST contain at least one explicit numeric token, such as a digit (0-9), a percentage sign (%), a decimal number (e.g., 0.85), or words like "percent", "percentage points", "points", "times", "x faster", etc.
2. Explicit numeric tokens include both numerical digits and written numbers (e.g., "five", "ten percent").
3. If a sentence does NOT contain any explicit numeric token, you MUST NOT include it in "results_section", even if it describes a trend like "dropped rapidly", "became negligible", or "near-perfect".
4. If a numerical result is already present in "tables", it MUST NOT appear in "results_section" — no duplicates across these two categories. However, duplication between "figures" and "results_section" is explicitly ALLOWED.
5. Each extracted item in "results_section" should preserve the paper's original wording as closely as possible. Prefer verbatim extraction of the relevant clause over paraphrasing.
6. Do NOT include any experimental setup or hyperparameter values (e.g., "epoch 5", "batch size 32", "K = 20", "N = 1000") in "results_section". These are not experimental results.

## NEGATIVE EXAMPLES - DO NOT EXTRACT:
- "Figure 3: Confidence-filtered ... test accuracy remains >98%." -> REASON: This is a figure caption. Discard.
- "Table 1: Performance metrics ... Model A achieves 95%." -> REASON: This is a table caption. Discard.
- "Trust calibration error became negligible by epoch 5." -> REASON: The result "negligible" is qualitative. "5" is just an epoch index (setup). Discard.
- "We use a batch size of 64 and train for 10 epochs." -> REASON: This is experimental setup/hyperparameters. Discard.
- "The framework shows significant improvement." -> REASON: No numerical value. Discard.
"""

JUDGE_RUNNERS: Dict[str, Callable[..., object]] = {
    "claude": run_claude_judge,
    "codex": run_codex_judge,
    "gemini": run_gemini_judge,
}


def _resolve_paths(task_path: str, paper_filename: str):
    """Resolve a required task_path and paper_filename to absolute/relative paths."""
    if not task_path:
        raise ValueError("task_path is required for extraction.")
    if not paper_filename:
        raise ValueError("paper_filename is required for extraction.")

    task_path = osp.abspath(task_path)
    abs_paper = paper_filename if osp.isabs(paper_filename) else osp.join(task_path, paper_filename)
    abs_paper = osp.abspath(abs_paper)
    if not osp.exists(abs_paper):
        raise FileNotFoundError(
            f"Could not resolve paper_filename '{paper_filename}' under task path {task_path}."
        )
    if not osp.isfile(abs_paper):
        raise FileNotFoundError(f"Paper path is not a file: {abs_paper}")

    paper_file_relpath = osp.relpath(abs_paper, task_path)
    return task_path, paper_file_relpath


def _run_extraction_agent(
    judge_type: str,
    *,
    task_path: str,
    paper_file_relpath: str,
    output_path: str,
    prompt: str,
    model_name: Optional[str],
    wait_seconds: Optional[int],
) -> None:
    judge_runner = JUDGE_RUNNERS.get(judge_type)
    if judge_runner is None:
        raise ValueError(f"Unknown judge: {judge_type}")

    judge_runner(
        task_path=task_path,
        paper_file=paper_file_relpath,
        output_path=output_path,
        prompt=prompt,
        model_name=model_name,
        wait_seconds=wait_seconds,
        skip_if_dir_exists=False,
        log_filename="extraction_log.json",
    )


def extract_results(
    task_path: str,
    paper_filename: str,
    output_path: str = "fs_extracted.json",
    judge_type: str = "claude",
    prompt: str = extract_results_prompt,
    model_name: Optional[str] = None,
    wait_seconds: Optional[int] = None,
):
    """
    Extract experimental results from a research paper.

    Args:
        task_path (str): Path to the task root directory.
        paper_filename (str): Paper filename or relative path inside the task directory.
        output_path (str): Name of the output JSON file. Default is "fs_extracted.json".
        judge_type (str): Type of agent to use for extraction. Default is "claude".
        prompt (str): The prompt to guide the extraction process.
        model_name (str, optional): Model name override for the extraction agent.
                     For claude, e.g. "claude-sonnet-4-20250514", "claude-opus-4-20250514".
                     For codex, e.g. "o4-mini", "gpt-5.2-codex".
                     If None, uses the agent's default model.
        wait_seconds (int, optional): Override for how long to wait for the
                     extraction output file after the judge process exits.
    """
    task_path, paper_file_relpath = _resolve_paths(task_path, paper_filename)
    logging.info(f"Resolved task_path: {task_path}")
    if paper_file_relpath:
        logging.info(f"Paper file (relative to task): {paper_file_relpath}")

    paper_is_pdf = bool(paper_file_relpath and paper_file_relpath.lower().endswith(".pdf"))
    
    refinement_suffix = (
        f"\nYour output file should be saved in the directory 'fabscore_{judge_type}/' inside the task folder."
        f"\nThe paper file you must inspect is: {paper_file_relpath}"
        + (
            "\nThis paper is a PDF. In the `figures` array, record figures by figure label and its caption such as `Figure 1: xxx` or `Figure 3(a): xxx`."
            if paper_is_pdf
            else ""
        )
        +
        "\n\nAfter drafting your extraction, do a final pass: scan every sentence in the Results/Experiments section that has numerical values. "
        "Only extract from lines that represent the actual body text of the paper. Make sure no valid numerical results from the ACTUAL body text are missed. "
        f"\n\nprogress.md REQUIREMENT: You MUST create or update 'fabscore_{judge_type}/progress.md'. "
        "Append a new session entry with session purpose `extraction`, what paper/context you inspected, what JSON files you created or updated in this session, and what the next session should do. "
        "Do not overwrite previous entries."
    )

    _run_extraction_agent(
        judge_type,
        task_path=task_path,
        paper_file_relpath=paper_file_relpath,
        output_path=output_path,
        prompt=prompt + refinement_suffix,
        model_name=model_name,
        wait_seconds=wait_seconds,
    )
   

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Extract experimental results from a research paper.")
    parser.add_argument("--task_path", required=True, help="Path to the task root directory.")
    parser.add_argument("--paper_filename", required=True, help="Paper filename or relative path inside the task directory.")
    parser.add_argument("--judge_type", default="claude", choices=SUPPORTED_JUDGES, help="Type of judge to use for extraction.")
    parser.add_argument("--model_name", default=None, help="Specific model name to use for the judge.")
    parser.add_argument("--wait_seconds", type=int, default=None, help="Optional override for extraction output wait time.")
    
    args = parser.parse_args()

    extract_results(
        task_path=args.task_path,
        paper_filename=args.paper_filename,
        judge_type=args.judge_type,
        model_name=args.model_name,
        wait_seconds=args.wait_seconds,
    )