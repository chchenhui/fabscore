import os
import os.path as osp
import logging
from typing import Optional

from fabscore.judges.base import run_judge_cli

claude_prompt = """You are an expert AI research paper reviewer.
Your task is to validate the experimental results reported in the paper against the provided code.   
"""

def run_claude_judge(
    task_path: str,
    paper_file: str,
    output_path: str = "verification.json",
    prompt = claude_prompt,
    wait_seconds: int = 10800,
    check_interval: int = 10,
    model_name: Optional[str] = None,
    output_dir: str = "fabscore_claude",
    skip_if_dir_exists: bool = True,
    log_filename: Optional[str] = None,
):
    """
    Runs the Claude judge on the given task path.

    Args:
        task_path (str): The task root directory to run in.
        paper_file (str): Relative path to the paper file within task_path.
        output_path (str): Name of the output file.
        prompt (str): The prompt to send to Claude.
        wait_seconds (int): Max seconds to wait for the output file.
        check_interval (int): Seconds between checks for the output file.
        model_name (str, optional): Model to use. If None, uses the default model.
        output_dir (str): Directory to store output. If None, output_path is treated as relative to task_path.
        skip_if_dir_exists (bool): If True and output_dir exists, skip execution.

    Returns:
       dict or str or None: JSON content or raw text.
    """
    full_prompt = prompt
    if paper_file:
        full_prompt += f"\n\nThe paper file to analyze is located at: {paper_file}"

    cmd = [
        "claude", "-p", full_prompt,
        "--output-format", "json",
        "--verbose", "--dangerously-skip-permissions"
    ]
    if model_name:
        cmd.extend(["--model", model_name])

    return run_judge_cli(
        cli_name="claude",
        cli_command=cmd,
        task_path=task_path,
        output_path=output_path,
        wait_seconds=wait_seconds,
        check_interval=check_interval,
        output_dir=output_dir,
        skip_if_dir_exists=skip_if_dir_exists,
        log_filename=log_filename,
    )
