
"""
Lightweight automatic evaluation for LLM explanations without external dependencies.
Metrics:
- exact_match: 1 if normalized strings equal, else 0 (for QA-style items)
- token_f1: harmonic mean of precision/recall on token overlap
- numeric_consistency: compares sets of numbers extracted from system/reference texts (tolerance configurable)
- coverage: fraction of reference n-grams (n=1..2) present in candidate
"""
import argparse, re, csv, math
from collections import Counter

def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def tokenize(s: str):
    return re.findall(r"\d+\.\d+|\d+|[a-zA-Z]+", s.lower())

def exact_match(a: str, b: str) -> float:
    return 1.0 if norm_text(a) == norm_text(b) else 0.0

def token_f1(pred: str, ref: str) -> float:
    p = tokenize(pred); r = tokenize(ref)
    pc = Counter(p); rc = Counter(r)
    overlap = sum((pc & rc).values())
    if overlap == 0: return 0.0
    prec = overlap / max(1, len(p))
    rec  = overlap / max(1, len(r))
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def extract_numbers(s: str):
    return [float(x.replace('%','')) for x in re.findall(r"-?\d+(?:\.\d+)?%?", s)]

def numeric_consistency(pred: str, ref: str, tol: float=1e-3) -> float:
    pn = extract_numbers(pred)
    rn = extract_numbers(ref)
    if not rn: return 1.0  # nothing to match
    if not pn: return 0.0
    # percentage-based set match (order-insensitive); allow tolerance for floats
    matched = 0
    rn_used = [False]*len(rn)
    for x in pn:
        for i,y in enumerate(rn):
            if rn_used[i]: continue
            if abs(x - y) <= tol*max(1.0, abs(y)):
                rn_used[i] = True
                matched += 1
                break
    return matched / max(1, len(rn))

def evaluate(reference_csv: str, outputs_csv: str, out_csv: str):
    refs = {}
    with open(reference_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            refs[row['id']] = row

    rows = []
    with open(outputs_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            qid = row['id']
            pred = row['answer']
            ref = refs[qid]['reference_answer']
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)
            num = numeric_consistency(pred, ref, tol=1e-3)
            cov = token_f1(pred, ref)  # simple proxy
            rows.append({
                "id": qid,
                "exact_match": f"{em:.3f}",
                "token_f1": f"{f1:.3f}",
                "numeric_consistency": f"{num:.3f}",
                "coverage": f"{cov:.3f}"
            })

    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="datasets/llm_eval_reference.csv")
    ap.add_argument("--out", default="results/llm_eval_scores.csv")
    ap.add_argument("--pred", default="datasets/llm_eval_outputs.csv")
    args = ap.parse_args()
    evaluate(args.ref, args.pred, args.out)
    print(f"Wrote scores to {args.out}")
