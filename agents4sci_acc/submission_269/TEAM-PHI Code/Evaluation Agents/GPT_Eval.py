import os
import json
import re
from openai import AzureOpenAI


endpoint = "your_Azure_endpoint"

deployment = "gpt-35-turbo"
# deployment = "gpt-4o-mini"

api_version = "api_version"

subscription_key = "your_subscription_key"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


input_json_path = "De-id Agents' Final Output/gemma2_deid_outputs.json"


notes_dir = "folder_contain_clinical_notes"


guideline_path = "guideline_eval.txt"


output_dir = "gpt35_evaluate_gemma2"
os.makedirs(output_dir, exist_ok=True)


with open(guideline_path, "r", encoding="utf-8") as f:
    guideline = f.read()

with open(input_json_path, "r", encoding="utf-8") as f:
    items = json.load(f)
if not isinstance(items, list):
    raise ValueError("Input JSON must be a list of items.")


TITLE_RE = re.compile(
    r"^\s*(mr|mrs|ms|miss|dr|prof|sir|madam|madame|mister)\.?\s+",
    re.IGNORECASE,
)

def strip_person_title(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    s = TITLE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    return s.strip()

def load_note_text(base_dir: str, file_name: str) -> str:
    path = os.path.join(base_dir, file_name)
    if not os.path.exists(path):
        print(f"!!! Warning: note file not found: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        if isinstance(js, dict) and "original_text" in js:
            return js["original_text"]
        return json.dumps(js, ensure_ascii=False)
    except json.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def build_pairs_from_deid_output(deid_output: dict,
                                 strip_titles: bool = True,
                                 dedup: bool = True):
    pairs = []
    if not isinstance(deid_output, dict):
        return pairs
    for key, values in deid_output.items():
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            continue
        for v in values:
            if v is None:
                continue
            val = str(v).strip()
            if not val:
                continue
            if key.upper() == "PERSON" and strip_titles:
                val = strip_person_title(val)
                if not val:
                    continue
            pairs.append((key.lower(), val))
    if dedup:
        seen, uniq = set(), []
        for c, v in pairs:
            if (c, v) in seen:
                continue
            seen.add((c, v))
            uniq.append((c, v))
        return uniq
    return pairs


total = len(items)
for idx, item in enumerate(items, start=1):
    file_name = item.get("file", f"item_{idx}.json")
    deid_output = item.get("deid_output", {})

    note_text = load_note_text(notes_dir, file_name)
    pairs = build_pairs_from_deid_output(deid_output)

    user_content = (
        guideline.strip()
        + "\n\nClinical note:\n"
        + note_text
        + "\n\nHere are the pairs:\n"
        + json.dumps(pairs, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": "You are an experienced doctor who helps with PHI annotation."},
        {"role": "user", "content": user_content},
    ]

    print(f">>> Evaluating {file_name} ({idx}/{total}) ...")
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )
    eval_text = response.choices[0].message.content

    out_data = {
        "file": file_name,
        "pairs": pairs,
        "eval_raw": eval_text,
    }
    out_path = os.path.join(output_dir, os.path.basename(file_name))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")

print("All done.")
