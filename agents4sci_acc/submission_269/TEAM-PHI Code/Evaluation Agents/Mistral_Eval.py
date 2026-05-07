from __future__ import annotations
import argparse, ast, json, re, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from mlx_lm import load, generate  # pip install mlx mlx-lm


DEFAULT_MODEL_DIR = "mistral-7b-instruct-v0.3-mlx-q4"
DEFAULT_DEID_DIR  = "deid_outputs"
DEFAULT_NOTES_DIR = "folder_contain_cinical_notes"
DEFAULT_GUIDE     = "guideline_eval.txt"
MAX_TOKENS        = 512
TEMPERATURE       = 0.0
TOP_P             = 1.0
SLEEP             = 0.02

FORMAT_SPEC = """
Return one valid JSON object with exactly these keys:
- "Number of Correct Pairs": <int>
- "CorrectByCategory": {"NAME": <int>, "DATE_TIME": <int>}
- "Correct Pairs": [[<CATEGORY>, <VALUE>], ...]
Normalization rules:
- Treat PERSON as NAME.
- Treat DATE or DATETIME as DATE_TIME.
Only output the JSON, no extra text.
""".strip()

# ============= I/O =============
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

# ============= deid_output -> pairs =============
SEP_RE_GENERIC = re.compile(r"\s*(?:,|，|;| and | AND | & )\s*")
REDACT_RE = re.compile(r"^(?:x+|\*+|<\s*redacted\s*>|\[\s*redacted\s*\]|redacted|n/?a|na)$", re.I)
TITLE_RE = re.compile(r"^\s*(mr|mrs|ms|miss|dr|prof|sir|madam|madame|mister)\.?\s+", re.I)
STRIP_PUNCT_RE = re.compile(r"^[\s\.,;:!?\(\)\[\]\{\}\"']+|[\s\.,;:!?\(\)\[\]\{\}\"']+$")

DATE_KEYS = {
    "DATE", "DOB", "BIRTHDATE", "DATE_OF_BIRTH",
    "DOS", "DOA", "DOD", "ADMIT_DATE", "DISCHARGE_DATE",
    "DATE_TIME", "DATETIME"
}
MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December"
DATE_PAT = re.compile(
    rf"""(?:
        \b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b
      | \b\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}\b
      | \b(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{2,4}}\b
    )""", re.I | re.VERBOSE
)

def normalize_value(v: str) -> str:
    v = STRIP_PUNCT_RE.sub("", v.strip())
    v = re.sub(r"\s+", " ", v)
    return v

def _split_generic(val: str) -> List[str]:
    parts = [p for p in SEP_RE_GENERIC.split(val) if p != ""]
    return parts if parts else ([val] if val else [])

def _extract_dates(val: str) -> List[str]:
    hits = [m.group(0).strip(" ,;") for m in DATE_PAT.finditer(val)]
    return hits if hits else _split_generic(val)

def strip_person_title(name: str) -> str:
    s = TITLE_RE.sub("", name.strip())
    s = re.sub(r"\s+", " ", s).replace(".", "")
    return s.strip()

def parse_phi_maybe_string(phi: Any) -> Dict[str, Any]:
    if isinstance(phi, dict):
        return phi
    if isinstance(phi, str):
        s = phi.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict): return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict): return obj
        except Exception:
            pass
    return {}

def flatten_extracted_phi(phi: Dict[str, Any],
                          strip_titles_for_person: bool = True,
                          dedup: bool = True) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    d = parse_phi_maybe_string(phi)

    for key, values in d.items():
        cat = str(key).upper().strip()
        vals = values if isinstance(values, (list, tuple)) else [values]
        for v in vals:
            if v is None: continue
            raw = str(v)
            chunks = _extract_dates(raw) if cat in DATE_KEYS else _split_generic(raw)
            for piece in chunks:
                s = normalize_value(piece)
                if not s or REDACT_RE.match(s): continue
                if cat == "PERSON" and strip_titles_for_person:
                    s = strip_person_title(s)
                    if not s: continue
                pairs.append((cat, s))

    if dedup:
        seen, out = set(), []
        for c, v in pairs:
            if (c, v) in seen: continue
            seen.add((c, v)); out.append((c, v))
        pairs = out
    return pairs


def map_cat(c: str) -> str:
    cu = c.upper().strip().replace(" ", "_")
    if cu in ("PERSON", "NAME"): return "NAME"
    if cu in ("DATE_TIME", "DATETIME", "DATE"): return "DATE_TIME"
    return cu

def percat_pred_counts(pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    out = {"NAME": 0, "DATE_TIME": 0}
    for it in pairs:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            mc = map_cat(str(it[0]))
            if mc in out:
                out[mc] += 1
    return out

def build_pred_set(pairs: List[Tuple[str, str]]) -> set[Tuple[str, str]]:
    s = set()
    for it in pairs:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            mc = map_cat(str(it[0]))
            val = normalize_value(str(it[1]))
            s.add((mc, val))
    return s

# ============= notes & tokens =============
def _find_note_path(notes_dir: Path, file_name: str) -> Optional[Path]:
    p = notes_dir / file_name
    if p.exists(): return p
    stem = Path(file_name).stem
    for cand in notes_dir.glob("*.json"):
        if cand.stem == stem:
            return cand
    return None

def load_note_text(notes_dir: Path, file_name: str) -> str:
    p = _find_note_path(notes_dir, file_name)
    if p is None:
        print(f"[WARN] note not found: {notes_dir}/{file_name}")
        return ""
    try:
        js = read_json(p)
        if isinstance(js, dict):
            for k in ("text", "original_text", "note", "content", "note_text"):
                if k in js and isinstance(js[k], str):
                    return js[k]
        return json.dumps(js, ensure_ascii=False)
    except Exception:
        return read_text(p)

TOKEN_RE = re.compile(r"\w+|[^\s]")
def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def _strip_fences(s: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
    return m.group(1).strip() if m else s

def _first_braced(s: str) -> Optional[str]:
    start, depth = -1, 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
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
    t = t.replace("\\'", "__S__").replace("'", '"').replace("__S__", "\\'")
    return t

def parse_eval_json(s: str) -> Optional[Dict[str, Any]]:
    raw = (s or "").strip()
    if not raw: return None
    seg = _strip_fences(raw)
    blk = _first_braced(seg)
    if not blk: return None
    for cand in (blk, _normalize_jsonish(blk)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def generate_once(model, tokenizer, prompt: str) -> str:
    kwargs = dict(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=MAX_TOKENS)
    try:
        return generate(**kwargs, temperature=TEMPERATURE, top_p=TOP_P)
    except TypeError:
        return generate(**kwargs)

# ============= Recall-Proxy  =============
def average_entity_total_across_models(deid_dir: Path, target_files_set: set[str]) -> float:
    totals = []
    for jf in sorted(deid_dir.glob("*.json")):
        try:
            items = read_json(jf)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        this_total = 0
        for it in items:
            fname = it.get("file") or it.get("filename")
            if not fname or fname not in target_files_set:
                continue
            deid_output = it.get("deid_output") or it.get("extracted_PHI") or {}
            pairs = flatten_extracted_phi(deid_output, strip_titles_for_person=True, dedup=True)
            this_total += len(pairs)
        if this_total > 0:
            totals.append(this_total)
    if not totals:
        return 0.0
    return sum(totals) / len(totals)


def main():
    ap = argparse.ArgumentParser(
        description="Mistral-7B(MLX) judge ALL deid models; category precision uses intersection with predictions."
    )
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--deid_dir",  default=DEFAULT_DEID_DIR)
    ap.add_argument("--notes_dir", default=DEFAULT_NOTES_DIR)
    ap.add_argument("--guideline", default=DEFAULT_GUIDE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_csv", default="mistral7b_judge_all_models.csv")
    args = ap.parse_args()

    model, tokenizer = load(args.model_dir)
    guideline = read_text(Path(args.guideline)).strip()

    deid_dir = Path(args.deid_dir)
    notes_dir = Path(args.notes_dir)
    model_files = sorted(deid_dir.glob("*.json"))
    if not model_files:
        raise SystemExit(f"[ERROR] No model jsons in {deid_dir.resolve()}")

    rows: List[Dict[str, Any]] = []

    for mf in model_files:
        try:
            items = read_json(mf)
        except Exception as e:
            print(f"[WARN] skip bad json: {mf.name} ({e})")
            continue
        if not isinstance(items, list):
            print(f"[WARN] not a list -> {mf.name}, skip")
            continue

        total_pairs = 0
        total_correct = 0
        total_tokens = 0
        used_files: List[str] = []

        # per-category
        percat_pred_tot = {"NAME": 0, "DATE_TIME": 0}
        percat_corr_tot = {"NAME": 0, "DATE_TIME": 0}

        print(f"\n=== Evaluating model: {mf.name} ===")
        for idx, it in enumerate(items, 1):
            if args.limit and len(used_files) >= args.limit:
                break

            file_name = it.get("file") or it.get("filename")
            if not file_name:
                continue
            deid_output = it.get("deid_output") or it.get("extracted_PHI") or {}


            pairs = flatten_extracted_phi(deid_output, strip_titles_for_person=True, dedup=True)
            total_pairs += len(pairs)
            pred_set = build_pred_set(pairs)
            pred_cnt = percat_pred_counts(pairs)
            for k in percat_pred_tot:
                percat_pred_tot[k] += pred_cnt[k]

            # note / tokens
            note_text = load_note_text(notes_dir, file_name)
            total_tokens += count_tokens(note_text)


            prompt = (
                guideline + "\n\n" + FORMAT_SPEC + "\n\n"
                "NOTE:\n" + note_text + "\n\n"
                "PAIRS:\n" + json.dumps(pairs, ensure_ascii=False)
            )
            out_text = generate_once(model, tokenizer, prompt)
            obj = parse_eval_json(out_text)


            if isinstance(obj, dict) and isinstance(obj.get("Number of Correct Pairs"), int):
                total_correct += min(obj["Number of Correct Pairs"], len(pairs))


            if isinstance(obj, dict) and isinstance(obj.get("Correct Pairs"), list):
                corr_local = {"NAME": 0, "DATE_TIME": 0}
                for it2 in obj["Correct Pairs"]:
                    if isinstance(it2, (list, tuple)) and len(it2) >= 2:
                        mc = map_cat(str(it2[0])); val = normalize_value(str(it2[1]))
                        if mc in corr_local and (mc, val) in pred_set:
                            corr_local[mc] += 1

                for k in corr_local:
                    percat_corr_tot[k] += min(corr_local[k], pred_cnt[k])
            elif isinstance(obj, dict) and isinstance(obj.get("CorrectByCategory"), dict):
                for k, v in obj["CorrectByCategory"].items():
                    mc = map_cat(str(k))
                    if mc in percat_corr_tot and isinstance(v, int):
                        percat_corr_tot[mc] += min(v, pred_cnt[mc])

            used_files.append(file_name)
            if idx % 10 == 0:
                print(f"[{idx}/{len(items)}] total_correct={total_correct} | "
                      f"DATE_TIME:{percat_corr_tot['DATE_TIME']}/{percat_pred_tot['DATE_TIME']}  "
                      f"NAME:{percat_corr_tot['NAME']}/{percat_pred_tot['NAME']}")
            time.sleep(SLEEP)

        # overall
        precision = (total_correct / total_pairs) if total_pairs > 0 else 0.0
        coverage  = (total_pairs / total_tokens) if total_tokens > 0 else 0.0
        avg_entities_all = average_entity_total_across_models(deid_dir, set(used_files))
        recall_proxy = (total_correct / avg_entities_all) if avg_entities_all > 0 else 0.0

        # per-cat precision
        name_prec = (percat_corr_tot["NAME"] / percat_pred_tot["NAME"]) if percat_pred_tot["NAME"] > 0 else 0.0
        dt_prec   = (percat_corr_tot["DATE_TIME"] / percat_pred_tot["DATE_TIME"]) if percat_pred_tot["DATE_TIME"] > 0 else 0.0

        print(f"---> {mf.name}: used={len(used_files)}, pairs={total_pairs}, correct={total_correct}, "
              f"precision={precision:.4f}, coverage={coverage*100:.2f}%, recall-proxy={recall_proxy:.4f}")
        print(f"     NAME precision={name_prec:.4f} ({percat_corr_tot['NAME']}/{percat_pred_tot['NAME']}), "
              f"DATE_TIME precision={dt_prec:.4f} ({percat_corr_tot['DATE_TIME']}/{percat_pred_tot['DATE_TIME']})")

        rows.append({
            "model_file": mf.name,
            "files_used": len(used_files),
            "total_pairs": total_pairs,
            "num_correct": total_correct,
            "precision": precision,
            "coverage": coverage,
            "recall_proxy": recall_proxy,
            "name_pred": percat_pred_tot["NAME"],
            "name_correct": percat_corr_tot["NAME"],
            "name_precision": name_prec,
            "datetime_pred": percat_pred_tot["DATE_TIME"],
            "datetime_correct": percat_corr_tot["DATE_TIME"],
            "datetime_precision": dt_prec,
        })

    print("\n===== Mistral-7B judge: metrics per model =====")
    print(f"{'Model':<30} {'Used':>6} {'Pairs':>8} {'Correct':>8} {'Prec':>8} {'Cov%':>8} {'RP':>8} "
          f"{'NAME_P':>8} {'NAME_C':>8} {'NAME_π':>8} {'DT_P':>8} {'DT_C':>8} {'DT_π':>8}")
    for r in rows:
        print(f"{r['model_file']:<30} {r['files_used']:>6} {r['total_pairs']:>8} {r['num_correct']:>8} "
              f"{r['precision']:>8.4f} {r['coverage']*100:>8.2f} {r['recall_proxy']:>8.4f} "
              f"{r['name_pred']:>8} {r['name_correct']:>8} {r['name_precision']:>8.4f} "
              f"{r['datetime_pred']:>8} {r['datetime_correct']:>8} {r['datetime_precision']:>8.4f}")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model_file","files_used","total_pairs","num_correct",
                    "precision","coverage","recall_proxy",
                    "name_pred","name_correct","name_precision",
                    "datetime_pred","datetime_correct","datetime_precision",
                ]
            )
            w.writeheader(); w.writerows(rows)
        print(f"\n[OK] Saved CSV -> {args.save_csv}")

if __name__ == "__main__":
    main()
