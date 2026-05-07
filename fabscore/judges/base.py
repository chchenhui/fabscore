"""
Common base function for running LLM CLI judges.

All three judge implementations (Claude, Codex, Gemini) share ~95% identical code for:
  - directory/output path management
  - subprocess execution
  - output prettification
  - file polling and result reading

This module extracts that common logic. Each judge module (claude.py, codex.py, gemini.py)
only needs to build its CLI-specific command and call ``run_judge_cli``.
"""

import os
import os.path as osp
import re
import subprocess
import time
import logging
import shutil
import json
from typing import Optional, List

DEFAULT_OUTPUT_WAIT_SECONDS = 10800


def run_judge_cli(
    cli_name: str,
    cli_command: List[str],
    task_path: str,
    output_path: str = "verification.json",
    wait_seconds: Optional[int] = DEFAULT_OUTPUT_WAIT_SECONDS,
    check_interval: int = 10,
    output_dir: Optional[str] = None,
    skip_if_dir_exists: bool = True,
    log_filename: Optional[str] = None,
):
    """
    Execute an LLM CLI judge and return its output.

    This is the shared implementation used by all judge wrappers (Claude, Codex, Gemini).
    It handles:
      - CLI availability check
      - Output directory creation and skip-if-exists logic
      - Subprocess execution with stdout capture
      - JSON prettification of log output
      - Polling for the expected output file
      - Reading and returning the result (JSON or raw text)

    Args:
        cli_name: Name of the CLI binary (e.g. "claude", "codex", "gemini").
        cli_command: The fully constructed command list to execute.
        task_path: The task root directory to run in (passed as ``cwd``).
        output_path: Name of the output file the agent is expected to write.
        wait_seconds: Max seconds to wait for the output file after execution.
            If None, uses ``DEFAULT_OUTPUT_WAIT_SECONDS``.
        check_interval: Seconds between polls for the output file.
        output_dir: Directory to store output. If None, output_path is relative to task_path.
        skip_if_dir_exists: If True and output_dir already exists, skip execution.
        log_filename: Basename for the raw CLI log file. Defaults to ``{cli_name}_log.json``.

    Returns:
        dict, str, or None: Parsed JSON, raw text content, or None if skipped.
    """
    # Check CLI availability
    if not shutil.which(cli_name):
        raise RuntimeError(
            f"Command '{cli_name}' not found in PATH. "
            f"Please ensure the {cli_name} CLI is installed and configured."
        )

    if log_filename is None:
        log_filename = f"{cli_name}_log.json"
    if wait_seconds is None:
        wait_seconds = DEFAULT_OUTPUT_WAIT_SECONDS

    # Resolve output paths
    final_output_path = output_path
    output_log_file = log_filename

    if output_dir:
        abs_output_dir = osp.join(task_path, output_dir)
        if skip_if_dir_exists and osp.isdir(abs_output_dir):
            logging.info(f"Directory {output_dir} already exists, skip this command.")
            return None
        os.makedirs(abs_output_dir, exist_ok=True)
        final_output_path = osp.join(output_dir, output_path)
        output_log_file = osp.join(output_dir, log_filename)
    else:
        if osp.dirname(output_path):
            os.makedirs(osp.join(task_path, osp.dirname(output_path)), exist_ok=True)
        output_log_file = osp.join(osp.dirname(output_path) or ".", log_filename)

    # All paths are relative to task_path
    abs_log = osp.join(task_path, output_log_file)
    abs_output = osp.join(task_path, final_output_path)

    logging.info(f"Running {cli_name} judge for {task_path}...")

    import signal
    with open(abs_log, "w") as fout:
        # Use Popen with start_new_session to ensure we can kill the process group on interrupt
        process = subprocess.Popen(
            cli_command,
            stdout=fout,
            stderr=subprocess.PIPE,
            text=True,
            cwd=task_path,
            start_new_session=True  # Create new process group
        )
        
        try:
            _, stderr_output = process.communicate(timeout=wait_seconds)
            
            # Reconstruct CompletedProcess for compatibility
            result = subprocess.CompletedProcess(
                args=cli_command,
                returncode=process.returncode,
                stdout=None, # stdout went to file
                stderr=stderr_output
            )
        except subprocess.TimeoutExpired:
            logging.error(
                f"{cli_name} judge process timed out after {wait_seconds}s. "
                "Terminating process group..."
            )
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise RuntimeError(f"{cli_name} judge process timed out after {wait_seconds}s.")
            
        except KeyboardInterrupt:
            logging.warning("\n[Interrupted] Terminating judge process group...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise
        except Exception as e:
            # Ensure process is killed on other errors too
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except: 
                pass
            raise

    # Try to prettify the log JSON
    try:
        with open(abs_log, "r") as f:
            content = json.load(f)
        with open(abs_log, "w") as f:
            json.dump(content, f, indent=4, ensure_ascii=False)
    except (json.JSONDecodeError, FileNotFoundError):
        pass

    if result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        logging.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"{cli_name} command failed with return code {result.returncode}")

    logging.info("Command executed successfully.")

    # Wait for the output file to appear
    for waited in range(0, wait_seconds, check_interval):
        if osp.exists(abs_output):
            logging.info(f"Generated {final_output_path} after {waited}s.")
            break
        time.sleep(check_interval)
    else:
        raise ValueError(f"Failed to generate {final_output_path} after {wait_seconds}s.")

    # Read and return the result
    try:
        with open(abs_output, "r", encoding="utf-8") as f:
            content = f.read()
            if output_path.endswith(".json"):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Robust extraction: look for the LAST json-like block containing "verdict"
                    import re
                    matches = list(re.finditer(r"(\{[\s\S]*?\"verdict\"[\s\S]*?\})", content))
                    if not matches:
                        logging.error(f"No verdict block found in {final_output_path} (length={len(content)})")
                        raise RuntimeError(f"{output_path} is invalid JSON and no verdict block found.")
                    try:
                        data = json.loads(matches[-1].group(1))
                    except json.JSONDecodeError:
                        raise RuntimeError(f"Failed to parse extracted verdict block from {output_path}")

                # Persist cleaned/prettified JSON on disk
                with open(abs_output, "w", encoding="utf-8") as f_out:
                    json.dump(data, f_out, indent=4, ensure_ascii=False)
                return data
            else:
                return content
    except Exception as e:
        logging.error(f"Failed to read/parse {final_output_path}: {e}")
        raise
