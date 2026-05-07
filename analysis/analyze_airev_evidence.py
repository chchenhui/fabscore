import json
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
from fabscore.utils.llm import create_client

COVERAGE_PATH = "airev_coverage.json"
CACHE_PATH = "airev_evidence_cache.json"
CLASSIFIED_PATH = "airev_evidence_classified.json"
MODEL = "claude-sonnet-4-6" 

PROMPT_TEMPLATE = """
You are an expert meta-reviewer analyzing AI-generated peer reviews.
Your task is to classify the following 'evidence' provided by an AI reviewer into exactly ONE of the following four categories:

1. Intra-paper Contradictions: The evidence points out numbers or statements that contradict each other within the text, tables, or figures of the paper (e.g., abstract vs. table, impossible sum).
2. Mathematical Implausibility: The evidence points out a mathematical error, incorrect calculation, or violation of basic formulas (e.g., NPV calculation is wrong).
3. Lack of Quantitative Evidence: The evidence criticizes the paper for missing details, absent tables, or lacking statistical variability to support a claim.
4. Code-to-Text Mismatch: The evidence explicitly compares the paper's text to a provided code snippet and notes a discrepancy.

Evidence to classify:
"{evidence}"

Reply in JSON format exactly like this:
{{
  "category": "1. Intra-paper Contradictions",
  "reason": "A brief 1-sentence explanation of why it fits this category."
}}
"""

def llm_classify(client, evidence_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(evidence=evidence_text)
    try:
        response_text = client.generate(prompt=prompt)
        
        # Clean up potential markdown blocks
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if "```" in clean_text:
            clean_text = clean_text[:clean_text.find("```")]
            
        result = json.loads(clean_text.strip())
        return result
    except Exception as e:
        print(f"  [Error classifying] {e}")
        return {"category": "1. Intra-paper Contradictions", "reason": f"Error: {str(e)}"}

def print_unique_stats(evidence_to_class):
    """Calculate and print stats based on unique evidence strings."""
    unique_stats = Counter()
    for res in evidence_to_class.values():
        cat = res.get("category", "Unknown")
        # Handle label migration if needed
        if "In-Paper" in cat or "In-Document" in cat or "Internal" in cat:
            cat = "1. Intra-paper Inconsistency"
        unique_stats[cat] += 1
    
    n = len(evidence_to_class)
    print("\n" + "=" * 70)
    print(f" AI Review Evidence Classification (Unique Basis, N={n})")
    print("=" * 70)
    if n > 0:
        for cat, count in sorted(unique_stats.items()):
            pct = (count / n) * 100
            print(f"  {cat:40s}: {count:>3d} ({pct:.1f}%)")
    else:
        print("  No evidence found to classify.")
    print("-" * 70)

def main():
    # --- Fast Track: If classified file already exists, use it ---
    if os.path.exists(CLASSIFIED_PATH):
        print(f"Found existing results in {CLASSIFIED_PATH}. Generating report...")
        with open(CLASSIFIED_PATH, encoding='utf-8') as f:
            classified_data = json.load(f)
        
        evidence_to_class = {}
        for item in classified_data:
            ev = item.get("evidence", "").strip()
            cat = item.get("ai_reason_category") or item.get("category", "Unknown")
            if ev and ev not in evidence_to_class:
                evidence_to_class[ev] = {"category": cat}
        
        print_unique_stats(evidence_to_class)
        return

    # --- Standard Track: Perform classification ---
    if not os.path.exists(COVERAGE_PATH):
        print(f"Error: {COVERAGE_PATH} not found.")
        return

    with open(COVERAGE_PATH, encoding='utf-8') as f:
        data = json.load(f)

    all_claims = []
    unique_evidences = set()

    for rec in data:
        sub_id = rec.get("submission_id")
        for fab in rec.get("fabrications", []):
            ev = fab.get("airev_evidence", "").strip()
            if ev:
                all_claims.append({
                    "sub_id": sub_id,
                    "claim_index": fab.get("claim_index"),
                    "claim": fab.get("claim"),
                    "verdict": fab.get("fabscore_verdict"),
                    "evidence": ev
                })
                unique_evidences.add(ev)

    unique_list = sorted(list(unique_evidences))
    print(f"Total claims: {len(all_claims)}, Unique evidence strings: {len(unique_list)}")
    
    evidence_to_class = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            evidence_to_class = json.load(f)
        print(f"Loaded {len(evidence_to_class)} classifications from cache.")

    print(f"Initializing LLM client ({MODEL})...")
    client = create_client(model_name=MODEL)
    
    for i, ev_text in enumerate(unique_list, 1):
        if ev_text in evidence_to_class: continue
        print(f"[{i}/{len(unique_list)}] Classifying: {ev_text[:50]}...")
        res = llm_classify(client, ev_text)
        evidence_to_class[ev_text] = res
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(evidence_to_class, f, indent=2, ensure_ascii=False)

    print_unique_stats(evidence_to_class)

    final_results = []
    for claim in all_claims:
        res = evidence_to_class.get(claim["evidence"], {"category": "Unknown"})
        claim["ai_reason_category"] = res.get("category", "Unknown")
        claim["ai_reason_explanation"] = res.get("reason", "")
        final_results.append(claim)

    with open(CLASSIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed classification saved to {CLASSIFIED_PATH}")

if __name__ == "__main__":
    main()
