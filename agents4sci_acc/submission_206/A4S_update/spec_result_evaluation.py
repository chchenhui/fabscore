import json, re, traceback
from typing import Any, Dict, List, Tuple

# claude-3-5-sonnet-20241022
# MODEL = "claude-3-7-sonnet-20250219"
#"gpt-4o"
MODEL = "claude-3-5-sonnet-20241022"


INPUT_PATH = f"all_spec_{MODEL}.jsonl"
SUMMARY_TXT = f"results_summary_spec_{MODEL}.txt"
ERROR_PATH = f"failed_spec_{MODEL}.jsonl"

PRELUDE = (
    "from typing import *\n"
    "import math, functools, itertools, collections\n"
)

def extract_python(code: str) -> str:
    s = code.strip()
    m = re.match(r"(?s)^\s*```(?:python)?\s*(.*?)\s*```\s*$", s, flags=re.IGNORECASE)
    return m.group(1) if m else s

def alias_entry_point_if_needed(code: str, entry_point: str) -> str:

    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, flags=re.M) is not None:
        return code

    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code, flags=re.M)
    if m:
        first_func = m.group(1)
        if first_func != entry_point:
            code += f"\n\n# auto alias for evaluation\n{entry_point} = {first_func}\n"
    return code

def run_test(task: Dict[str, Any], raw_code: str) -> Tuple[bool, str, str]:
    ns: Dict[str, Any] = {}
    try:
        code = extract_python(raw_code)
        code = alias_entry_point_if_needed(code, task["entry_point"])
        code_wrapped = f"{PRELUDE}\n{code}\n\n{task['test']}\n\ncheck({task['entry_point']})"
        exec(compile(code_wrapped, "<eval>", "exec"), ns)
        return True, "", ""
    except AssertionError as e:
        return False, f"AssertionError: {e}", traceback.format_exc()
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}", traceback.format_exc()

def main() -> None:
    data: List[Dict[str, Any]] = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    results: Dict[str, Dict[str, Any]] = {}
    passed_cnt = 0
    failed_records: List[Dict[str, Any]] = []

    for item in data:
        tid = item.get("task_id")
        entry_point = item.get("entry_point")
        test = item.get("test")
        gen = item.get("generated_solution")

        if not gen:
            results[tid] = {"passed": False, "error": "No generated_solution in record"}
            failed_records.append({
                "task_id": tid,
                "entry_point": entry_point,
                "error": "No generated_solution in record",
                "traceback": "",
                "test": test,
                "generated_solution": None,
                "prompt": item.get("prompt"),
            })
            continue

        passed, err, tb = run_test(
            {"entry_point": entry_point, "test": test},
            gen
        )
        results[tid] = {"passed": passed, "error": err}
        if passed:
            passed_cnt += 1
        else:
            failed_records.append({
                "task_id": tid,
                "entry_point": entry_point,
                "error": err,                
                "traceback": tb,              
                "test": test,
                "generated_solution": gen,
                "prompt": item.get("prompt"), 
            })

    total = len(results)
    print(f"Total: {total}, Passed: {passed_cnt}, Failed: {total - passed_cnt}")
    for tid, res in results.items():
        mark = "✅" if res["passed"] else "❌"
        detail = "" if res["passed"] else f" {res['error']}"
        print(f"{tid} {mark}{detail}")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"Total: {total}, Passed: {passed_cnt}, Failed: {total - passed_cnt}\n")

    with open(ERROR_PATH, "w", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nerror saved to: {ERROR_PATH}（Total: {len(failed_records)} ）")

if __name__ == "__main__":
    main()
