import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mlx_lm import load, generate

MODEL_DIR = "gemma-2-9b-it-mlx-q4"
GUIDELINE_PATH = "guideline.txt"
INPUT_DIR = Path("folder_contain_clinical_notes")
OUTPUT_DIR = Path("gemma2_out") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 384
TEMPERATURE = 0.2
TOP_P = 0.9
SLEEP_BETWEEN = 0.03
OVERWRITE = True

PHI_KEYS = {
    "PERSON", "NAME", "AGE", "DOB", "DATE", "ADDRESS", "LOCATION", "HOSPITAL",
    "PHONE", "EMAIL", "MRN", "SSN", "ID", "ZIP", "CITY", "STATE",
    "DOCTOR", "PROVIDER", "INSURANCE", "ACCOUNT", "PATIENT"
}

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def build_prompt(guideline: str, note_text: str) -> str:
    return f"{guideline.rstrip()}\n\n{note_text.strip()}"

def _strip_fences(s: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.I)
    return m.group(1).strip() if m else s

def _first_braced(s: str) -> Optional[str]:
    start, depth = -1, 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start != -1:
                    return s[start:i+1]
    return None

def _normalize_jsonish(s: str) -> str:
    t = s.strip()
    t = re.sub(r"//.*?$", "", t, flags=re.M)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r'(?P<prefix>[{\s,])\s*([A-Za-z][\w\s\-]*?)\s*:',
               lambda m: f'{m.group("prefix")}"{m.group(2).strip()}":', t)

    t = t.replace("\\'", "__S__")
    t = t.replace("'", '"').replace("__S__", "\\'")
    return t

def parse_to_dict(text: str) -> Dict[str, Any]:

    raw = (text or "").strip()
    if not raw:
        return {}
    seg = _strip_fences(raw)


    if seg.startswith("{") and seg.endswith("}"):
        for cand in (seg, _normalize_jsonish(seg)):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj.get("extracted_PHI", obj) \
                        if isinstance(obj.get("extracted_PHI", obj), dict) else obj
            except Exception:
                pass


    br = _first_braced(seg)
    if br:
        for cand in (br, _normalize_jsonish(br)):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj.get("extracted_PHI", obj) \
                        if isinstance(obj.get("extracted_PHI", obj), dict) else obj
            except Exception:
                pass


    fallback: Dict[str, Any] = {}
    for line in seg.splitlines():
        m = re.match(r"\s*(?:[-*]\s*)?([A-Za-z _]+)\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip().strip(",;")
        ku = k.upper().replace(" ", "_")
        if ku in PHI_KEYS and v:
            fallback[ku] = v
    return fallback

def generate_once(model, tokenizer, prompt: str) -> str:
    kwargs = dict(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=MAX_TOKENS)
    try:
        return generate(**kwargs, temperature=TEMPERATURE, top_p=TOP_P)
    except TypeError:
        return generate(**kwargs)

# --------------------
def main():
    print(f"Loading local model from: {MODEL_DIR}")
    model, tokenizer = load(MODEL_DIR)
    guideline = load_text(Path(GUIDELINE_PATH))

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        print(f"[WARN] No JSON files in {INPUT_DIR.resolve()}")
        return

    for i, fp in enumerate(files, 1):
        out_fp = OUTPUT_DIR / (fp.stem + ".out.json")
        if (not OVERWRITE) and out_fp.exists():
            print(f"[{i}/{len(files)}] Skip existing -> {out_fp.name}")
            continue

        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            note = str(data.get("text", "")).strip()
            if not note:
                print(f"[{i}/{len(files)}] {fp.name}: empty 'text', skip")
                continue

            prompt = build_prompt(guideline, note)
            out_text = generate_once(model, tokenizer, prompt)
            phi_dict = parse_to_dict(out_text) or {}


            result = {
                "filename": fp.name,
                "extracted_PHI": phi_dict
            }
            out_fp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(files)}] OK -> {out_fp.name}")

            time.sleep(SLEEP_BETWEEN)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {fp.name}: {e}")

if __name__ == "__main__":
    main()
