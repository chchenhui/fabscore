"""Output parser for CHECK_i / FINAL_ANSWER fields from model outputs.

Extracts structured fields using regex with fallback heuristics.
Normalizes numeric (int/float equivalence) and boolean (yes/no) answers.
"""

import re
from typing import Any


def _normalize_answer(raw: str) -> str:
    raw = raw.strip()
    lower = raw.lower()

    bool_map = {
        "yes": "Yes", "true": "Yes", "correct": "Yes",
        "complete": "Yes", "strongly complete": "Yes",
        "a is complete": "Yes", "a is strongly complete": "Yes",
        "no": "No", "false": "No", "incorrect": "No",
        "not complete": "No", "not strongly complete": "No", "incomplete": "No",
        "a is not complete": "No", "a is not strongly complete": "No",
    }
    if lower in bool_map:
        return bool_map[lower]

    asymp_yes = re.match(
        r'^f\s*=\s*[Oo]\s*\(\s*g\s*\)$', raw, re.IGNORECASE
    )
    if asymp_yes:
        return "Yes"

    asymp_no = re.match(
        r'^f\s*\\?\\?neq\s*[Oo]\s*\(\s*g\s*\)$', raw, re.IGNORECASE
    )
    if asymp_no:
        return "No"

    if re.match(r'^a\s*\\?\\?text\{\s*is complete', lower):
        return "Yes"
    if re.match(r'^a\s*\\?\\?text\{\s*is not complete', lower):
        return "No"

    try:
        val = float(raw)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    return raw


def parse_checks(text: str, k: int = 3) -> list[str | None]:
    results: list[str | None] = [None] * k
    for i in range(1, k + 1):
        pattern = rf"CHECK_{i}\s*:\s*(.+?)(?=CHECK_{i+1}|FINAL_ANSWER|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            raw = raw.split("\n")[0].strip()
            results[i - 1] = _normalize_answer(raw)
    return results


def parse_final_answer(text: str) -> str | None:
    pattern = r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        return _normalize_answer(raw)

    lines = text.strip().split("\n")
    if lines:
        last_line = lines[-1].strip()
        if last_line:
            return _normalize_answer(last_line)

    return None


def extract_boxed_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        ans = boxed[-1].strip()
        ans = re.sub(r"^\\text\{", "", ans)
        ans = ans.rstrip("}")
        return _normalize_answer(ans)
    return None


def extract_answer_robust(text: str, family: str | None = None) -> str | None:
    if not text or not text.strip():
        return None

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    ans_match = re.findall(
        r"(?:the answer is|the final answer is|therefore,?\s*(?:the answer is)?)"
        r"\s*\\?(?:boxed\{)?([^},.\n]+)",
        text, re.IGNORECASE,
    )
    if ans_match:
        ans = ans_match[-1].strip().rstrip("}")
        if ans:
            return _normalize_answer(ans)

    if family in ("asymptotics", "completeness"):
        yes_no = re.findall(r"\b(Yes|No)\b", text, re.IGNORECASE)
        if yes_no:
            return _normalize_answer(yes_no[-1])

    if family == "convolution":
        nums = re.findall(r"(?:=\s*|is\s+)(-?\d+)", text)
        if nums:
            return _normalize_answer(nums[-1])

    return None


def parse_output(text: str, k: int = 3) -> dict[str, Any]:
    checks = parse_checks(text, k)
    final = parse_final_answer(text)
    return {
        "checks": checks,
        "final_answer": final,
        "raw_text": text,
    }
