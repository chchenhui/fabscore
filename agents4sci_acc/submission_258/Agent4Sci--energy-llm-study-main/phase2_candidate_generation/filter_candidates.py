import os
import sys
import shutil
import unittest
import importlib.util
import re
import ast
import multiprocessing
import csv
from datetime import datetime

# --- Configuration ---
TIMEOUT_SECONDS = 2.0

# --- Helper Functions ---

def extract_clean_code(raw_text: str) -> str:
    """Extracts and normalizes Python code from various LLM output formats."""
    # Extract from <python_code> tags
    match = re.search(r"<python_code>(.*?)</python_code>", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(1)

    code = raw_text.strip()
    # Remove triple-backtick markdown fences
    code = re.sub(r"^```python\n", "", code)
    code = re.sub(r"^```\n", "", code)
    code = re.sub(r"\n```$", "", code)
    # Remove any __main__ guard clauses
    code = re.sub(r"if\s+__name__\s*==\s*['\"]__main__['\"]:(.|\n)*", "", code, flags=re.DOTALL)
    return code.strip()


def pick_first_function(tree):
    """Find the first top-level function in the AST."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def log_message(log_file, msg):
    """Append a timestamped message to the log file and print it."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Core Testing Logic ---

def test_runner_target(candidate_path, test_path, test_module_name, func_name, result_queue):
    """Run a candidate solution inside a separate process."""
    try:
        sys.path.insert(0, test_path)
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=test_path, pattern=f"{test_module_name}.py")

        spec = importlib.util.spec_from_file_location("candidate_module", candidate_path)
        candidate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate_module)

        solution_func = getattr(candidate_module, func_name, None)
        if solution_func is None:
            result_queue.put(False)
            return

        # Always bind as `solution` for tests
        for test_case in suite:
            for test in test_case:
                if isinstance(test, unittest.TestCase):
                    setattr(test, "solution", staticmethod(solution_func))

        runner = unittest.TextTestRunner(stream=open(os.devnull, "w"))
        result = runner.run(suite)

        result_queue.put(result.wasSuccessful())
    except Exception:
        result_queue.put(False)


def validate_and_run_candidate(candidate_path, test_path, test_module_name, processed_dir, log_file):
    """
    Parse candidate, pick correct function (solve or first top-level),
    and run tests in a sandbox process.
    Returns (passed: bool, processed: bool).
    """
    try:
        with open(candidate_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        code = extract_clean_code(raw_text)
        if not code:
            log_message(log_file, f"{candidate_path}  [FAIL] - No code found.")
            return False, False

        try:
            tree = ast.parse(code)
        except SyntaxError:
            log_message(log_file, f"{candidate_path}  [FAIL] - Syntax error.")
            return False, False

        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        processed = False

        if "solve" in func_names:
            target_func = "solve"
        else:
            fallback = pick_first_function(tree)
            if fallback is None:
                log_message(log_file, f"{candidate_path}  [FAIL] - No function found.")
                return False, False
            target_func = fallback
            # Rename it to `solve` in the code for consistency
            code = re.sub(rf"def\s+{fallback}\s*\(", "def solve(", code, count=1)
            processed = True

        # Write processed version if needed
        if processed:
            processed_path = os.path.join(processed_dir, os.path.basename(candidate_path))
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(code)
            run_path = processed_path
        else:
            run_path = candidate_path

        # Run tests
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=test_runner_target,
            args=(run_path, test_path, test_module_name, "solve", result_queue),
        )
        process.start()
        process.join(timeout=TIMEOUT_SECONDS)

        if process.is_alive():
            process.terminate()
            process.join()
            log_message(log_file, f"{candidate_path}  [FAIL] - Timeout.")
            return False, processed

        if process.exitcode != 0:
            log_message(log_file, f"{candidate_path}  [FAIL] - Crashed.")
            return False, processed

        if not result_queue.empty():
            is_correct = result_queue.get()
        else:
            is_correct = False

        log_message(
            log_file,
            f"{candidate_path}  [{'PASS' if is_correct else 'FAIL'}]"
            + (" (processed)" if processed else " (as-is)")
        )
        return is_correct, processed

    except Exception as e:
        log_message(log_file, f"{candidate_path}  [FAIL] - Internal error ({type(e).__name__}).")
        return False, False


# --- Main Execution Logic ---

def main():
    multiprocessing.set_start_method("spawn", force=True)

    MODEL_ID = "codellama:70b-instruct"
    generated_dir = f"generated_candidates-{MODEL_ID}-prompt_v5"
    filtered_dir = f"filtered_candidates-{MODEL_ID}-prompt_v5"
    processed_dir = f"processed_candidates-{MODEL_ID}-prompt_v5"
    benchmarks_dir = "benchmarks"
    logs_dir = "logs_filtering"
    summary_csv = "filter_summary.csv"

    if not os.path.exists(generated_dir):
        print(f"Error: Directory '{generated_dir}' not found. Run the generation script first.")
        return

    for d in [filtered_dir, processed_dir, logs_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # CSV summary header
    with open(summary_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Task", "Total", "Passed", "Processed", "Failed"])

    print("--- 🚀 Starting Candidate Filtering ---")
    task_dirs = sorted([d for d in os.listdir(generated_dir) if os.path.isdir(os.path.join(generated_dir, d))])

    for task_id in task_dirs:
        log_file = os.path.join(logs_dir, f"{task_id}.log")
        log_message(log_file, f"\n=== Filtering candidates for task: {task_id} ===")

        test_path = os.path.join(benchmarks_dir, task_id)
        if not os.path.exists(test_path):
            log_message(log_file, f"⚠️ Warning: No benchmark directory found for {task_id}. Skipping.")
            continue

        test_module_name = f"test_{task_id[4:]}"  # assumes '000_taskname' → 'test_taskname'
        task_generated_dir = os.path.join(generated_dir, task_id)
        task_filtered_dir = os.path.join(filtered_dir, task_id)
        task_processed_dir = os.path.join(processed_dir, task_id)
        os.makedirs(task_filtered_dir)
        os.makedirs(task_processed_dir)

        candidates = sorted([c for c in os.listdir(task_generated_dir) if c.endswith(".py")])
        total, passed, processed_count = 0, 0, 0

        for candidate_file in candidates:
            candidate_path = os.path.join(task_generated_dir, candidate_file)
            total += 1

            try:
                is_correct, was_processed = validate_and_run_candidate(
                    candidate_path, test_path, test_module_name, task_processed_dir, log_file
                )
                if is_correct:
                    shutil.copy(candidate_path, os.path.join(task_filtered_dir, candidate_file))
                    passed += 1
                    if was_processed:
                        processed_count += 1
            except Exception as e:
                log_message(log_file, f"{candidate_path}  [FAIL] - Fatal error {type(e).__name__}, skipping candidate.")
                continue

        failed = total - passed
        log_message(
            log_file,
            f"Summary for {task_id}: total={total}, passed={passed}, processed={processed_count}, failed={failed}"
        )

        # Append row to CSV
        with open(summary_csv, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([task_id, total, passed, processed_count, failed])

    print(f"\n✅ Candidate filtering complete. Valid candidates are in '{filtered_dir}'.")
    print(f"Processed variants (renamed functions) are in '{processed_dir}'.")
    print(f"Logs saved in '{logs_dir}'.")
    print(f"CSV summary written to '{summary_csv}'.\n")


if __name__ == "__main__":
    main()
