import os
import json
import asyncio
from typing import Tuple, Dict, Any, List

from openai import AsyncOpenAI

API_KEY = ""
client = AsyncOpenAI(api_key=API_KEY)
# client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.anthropic.com/v1/")

# claude-3-5-sonnet-20241022
# MODEL = "claude-3-7-sonnet-20250219"
MODEL = "gpt-4o"
TEMPERATURE = 0

MAX_CONCURRENCY = 10
sem = asyncio.Semaphore(MAX_CONCURRENCY)

SUMMARY_TXT = f"results_summary_baseline_{MODEL}.txt" 
PRELUDE = (
    "from typing import *\n"
    "import math, functools, itertools, collections\n"
)


async def fetch_code(prompt: str, task_id: str) -> Tuple[str, str]:
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful coding assistant. "
                            "Return ONLY valid Python code that defines the requested function "
                            "with the EXACT SAME name/signature. "
                            "You may add standard-library imports if needed. "
                            "No explanations, no markdown fences, no comments, no tests."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        code = resp.choices[0].message.content or ""
        return task_id, code
    except Exception as e:
        return task_id, f"ERROR: {e}"


def run_test(task: Dict[str, Any], code: str) -> Tuple[bool, str]:
    if code.startswith("ERROR:"):
        return False, code

    ns: Dict[str, Any] = {}
    try:
    
        code_wrapped = f"{PRELUDE}\n{code}\n\n{task['test']}\n\ncheck({task['entry_point']})"
        exec(code_wrapped, ns)
        return True, ""
    except Exception as e:
        return False, str(e)


async def main() -> None:
    with open("HumanEval.jsonl", "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = [json.loads(line) for line in f]

    coros = [fetch_code(item["prompt"], item["task_id"]) for item in data]
    codes = await asyncio.gather(*coros, return_exceptions=False)

    results: Dict[str, Dict[str, Any]] = {}
    index = {d["task_id"]: d for d in data}

    for task_id, code in codes:
        task = index[task_id]
        passed, err = run_test(task, code)
        results[task_id] = {"passed": passed, "error": err}

    total = len(results)
    passed_cnt = sum(1 for v in results.values() if v["passed"])
    print(f"Total: {total}, Passed: {passed_cnt}, Failed: {total - passed_cnt}")
    for tid, res in results.items():
        if res["passed"]:
            print(f"{tid} ✅")
        else:
            print(f"{tid} ❌ {res['error']}")
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"Total: {total}, Passed: {passed_cnt}, Failed: {total - passed_cnt}\n")


    with open(f"all_baseline_{MODEL}.jsonl", "w", encoding="utf-8") as f:
        for tid, res in results.items():
            item = index[tid]
            item["generated_solution"] = code
            item["error"] = res["error"]
            item["passed"] = res["passed"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"failed_baseline_{MODEL}.jsonl", "w", encoding="utf-8") as f:
        for tid, res in results.items():
            if not res["passed"]:
                item = index[tid]
                item["generated_solution"] = code
                item["error"] = res["error"]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ NEED OPENAI_API_KEY 。")
    asyncio.run(main())
