import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


local_model_dir = "Llama-3-70B-Instruct"

input_json_path = "gemma2_deid_outputs.json"

notes_dir = "folder_contain_clinical_notes"

guideline_path = "guideline_eval.txt"

output_dir = "llama_70b_evaluate_gemma2"
os.makedirs(output_dir, exist_ok=True)


STRIP_TITLES_FOR_PERSON = True
DEDUP_PAIRS = True
MAX_NEW_TOKENS = 512
DO_SAMPLE = False 
TEMPERATURE = 0.0   
TOP_P = 1.0  


print(">>> Loading local model...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

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
                                 strip_titles: bool = False,
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
        seen = set()
        uniq = []
        for c, v in pairs:
            if (c, v) in seen:
                continue
            seen.add((c, v))
            uniq.append((c, v))
        return uniq
    return pairs

def parse_number_from_json_text(s: str):
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(s[start:end+1])
            n = obj.get("Number of Correct Pairs")
            if isinstance(n, int):
                return n, obj
    except Exception:
        pass
    return None, None

def build_terminators(tok: AutoTokenizer):
    terms = []
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            terms.append(tok.eos_token_id)
        else:
            terms.extend([t for t in tok.eos_token_id if isinstance(t, int)])
    try:
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            terms.append(eot_id)
    except Exception:
        pass
    terms = sorted(set([t for t in terms if isinstance(t, int)]))
    return terms


terminators = build_terminators(tokenizer)
gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
)
if terminators:
    gen_kwargs["eos_token_id"] = terminators if len(terminators) > 1 else terminators[0]

total = len(items)
for idx, item in enumerate(items, start=1):
    file_name = item.get("file", f"item_{idx}.json")
    deid_output = item.get("deid_output", {})

    note_text = load_note_text(notes_dir, file_name)
    pairs = build_pairs_from_deid_output(
        deid_output,
        strip_titles=STRIP_TITLES_FOR_PERSON,
        dedup=DEDUP_PAIRS,
    )

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

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    print(f">>> Evaluating {file_name} ({idx}/{total}) ...")
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    response_tokens = outputs[0][input_ids.shape[-1]:]
    eval_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    parsed_n, parsed_obj = parse_number_from_json_text(eval_text)

    out_data = {
        "file": file_name,
        "pairs": pairs,
        "eval_raw": eval_text,
        "parsed": parsed_obj if parsed_obj is not None else None,
        "Number of Correct Pairs": parsed_n if parsed_n is not None else None,
    }
    out_path = os.path.join(output_dir, os.path.basename(file_name))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")

print("All done.")
