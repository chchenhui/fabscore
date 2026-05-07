# Answer extraction from generated text.
# extract_mc_answer: for MMStar multiple-choice (A/B/C/D)
# extract_yesno_answer: for HallusionBench Yes/No questions -> "1"/"0"
# Both first try to extract from <answer> tags (VLAA-Thinker thinking format).
import re


def _get_answer_block(text):
    match = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_mc_answer(text):
    answer_block = _get_answer_block(text)
    search_text = answer_block if answer_block else text

    patterns = [
        r'(?:^|\n)\s*([A-D])\s*[:.]',
        r'(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-D])\)?',
        r'\(([A-D])\)',
        r'\b([A-D])\s*[:.](?:\s)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, search_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    matches = re.findall(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', search_text)
    if matches:
        return matches[-1]

    if answer_block is None:
        matches = re.findall(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', text)
        if matches:
            return matches[-1]

    return ""


def extract_yesno_answer(text):
    answer_block = _get_answer_block(text)
    search_text = answer_block if answer_block else text

    text_lower = search_text.strip().lower()
    if text_lower.startswith("yes"):
        return "1"
    if text_lower.startswith("no"):
        return "0"

    last_yes = text_lower.rfind("yes")
    last_no = text_lower.rfind("no")

    if last_yes == -1 and last_no == -1:
        if answer_block is not None:
            return extract_yesno_answer_raw(text)
        return "0"
    if last_yes == -1:
        return "0"
    if last_no == -1:
        return "1"

    return "1" if last_yes > last_no else "0"


def extract_yesno_answer_raw(text):
    text_lower = text.strip().lower()
    last_yes = text_lower.rfind("yes")
    last_no = text_lower.rfind("no")

    if last_yes == -1 and last_no == -1:
        return "0"
    if last_yes == -1:
        return "0"
    if last_no == -1:
        return "1"
    return "1" if last_yes > last_no else "0"
