import os
import os.path as osp
import logging
from typing import Optional

from fabscore.judges.base import run_judge_cli

gemini_prompt = """You are an expert AI research paper reviewer.
Your task is to validate the experimental results reported in the paper against the provided code.   
"""

def run_gemini_judge(
    task_path: str,
    paper_file: str,
    output_path: str = "verification.json",
    prompt = gemini_prompt,
    wait_seconds: int = 10800,
    check_interval: int = 10,
    model_name: Optional[str] = None,
    output_dir: str = "fabscore_gemini",
    skip_if_dir_exists: bool = True,
    log_filename: Optional[str] = None,
):
    """
    Runs the Gemini judge on the given task path using the Gemini CLI.

    Args:
        task_path (str): The task root directory to run in.
        paper_file (str): Relative path to the paper file within task_path.
        output_path (str): Name of the output file.
        prompt (str): The prompt to send to Gemini.
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
        "gemini",
        "--approval-mode", "yolo",
        "--output-format", "json",
    ]
    if model_name:
        cmd.extend(["--model", model_name])
    # Prompt is the positional query
    cmd.append(full_prompt)

    return run_judge_cli(
        cli_name="gemini",
        cli_command=cmd,
        task_path=task_path,
        output_path=output_path,
        wait_seconds=wait_seconds,
        check_interval=check_interval,
        output_dir=output_dir,
        skip_if_dir_exists=skip_if_dir_exists,
        log_filename=log_filename,
    )
