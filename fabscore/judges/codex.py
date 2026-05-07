import os
import os.path as osp
import logging
from typing import Optional

from fabscore.judges.base import run_judge_cli

codex_prompt = """You are an expert AI research paper reviewer.
Your task is to validate the experimental results reported in the paper against the provided code.   
"""

def run_codex_judge(
    task_path: str,
    paper_file: str,
    output_path: str = "verification.json",
    prompt = codex_prompt,
    wait_seconds: int = 10800,
    check_interval: int = 10,
    model_name: Optional[str] = None,
    sandbox: str = "workspace-write",
    output_dir: str = "fabscore_codex",
    skip_if_dir_exists: bool = True,
    log_filename: Optional[str] = None,
):
    """
    Runs the Codex judge on the given task path.

    Uses the OpenAI Codex CLI (``codex exec``) in non-interactive mode
    to evaluate a research paper's experimental results against its code.

    Args:
        task_path (str): The task root directory to run in.
        paper_file (str): Relative path to the paper file within task_path.
        output_path (str): Name of the output file.
        prompt (str): The prompt to send to Codex.
        wait_seconds (int): Max seconds to wait for the output file.
        check_interval (int): Seconds between checks for the output file.
        model_name (str, optional): Model to use. If None, uses the default model.
        sandbox (str): Sandbox policy for Codex CLI.
        output_dir (str): Directory to store output. If None, output_path is treated as relative to task_path.
        skip_if_dir_exists (bool): If True and output_dir exists, skip execution.

    Returns:
       dict or str or None: JSON content or raw text.
    """
    full_prompt = prompt
    if paper_file:
        full_prompt += f"\n\nThe paper file to analyze is located at: {paper_file}"

    cmd = [
        "codex", "exec",
        "--full-auto",
        "--sandbox", sandbox,
        "--json",
    ]
    if model_name:
        cmd.extend(["--model", model_name])
    # Prompt is a positional argument for codex exec
    cmd.append(full_prompt)

    return run_judge_cli(
        cli_name="codex",
        cli_command=cmd,
        task_path=task_path,
        output_path=output_path,
        wait_seconds=wait_seconds,
        check_interval=check_interval,
        output_dir=output_dir,
        skip_if_dir_exists=skip_if_dir_exists,
        log_filename=log_filename,
    )
