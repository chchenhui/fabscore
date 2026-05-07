"""
extract_airev_atomic_llm.py

Core question: "Did the AI review also flag the fabrications that FabScore found?"

For each submission that has both an AI review (agents4sci_aireviews) and
FabScore results (fabscore_claude/fs_summary.json), this script:

  1. Loads only the claims FabScore judged as fabrications
     (Data/Experiment/Result Fabrication, No Code Files) from
     fabscore_claude/fs_summary.json as the baseline.
  2. Loads the full AI review text.
  3. Asks an LLM whether each fabrication claim was BOTH mentioned AND
     flagged as problematic by the AI review.

Per-claim output fields (only AI-review-confirmed entries are recorded):
  claim_index      – FabScore claim index
  claim            – fabrication claim text
  fabscore_verdict – e.g. "Experiment Fabrication"
  section          – tables / figures / results_section
  airev_evidence   – the relevant sentence(s) from the review

Per-submission stats:
  fabscore_fabrications – total FabScore fabrication claims (denominator)
  caught                – how many the AI review also flagged
  missed                – fabscore_fabrications - caught
  recall                – caught / fabscore_fabrications

Output: airev_coverage.json
"""

import json, os
from fabscore.utils.llm import create_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AIREV_DIR    = "../agents4sci_aireviews"
BASE_DIR     = ".."
MODEL        = "claude-sonnet-4-6"
MAX_TOKENS   = 16384
OUTPUT_PATH  = "airev_coverage.json"
SECTIONS     = ("tables", "figures", "results_section")
FAB_VERDICTS = {"Data Fabrication", "Experiment Fabrication", "Result Fabrication"}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
COVERAGE_PROMPT = """\
You are auditing an AI-generated peer review of a scientific paper.
We have extracted the following atomic claims from the paper — each
one is a specific data point, table value, figure observation, or
experimental result stated in the paper.

Your task: for each atomic claim, determine whether the AI review
BOTH (a) refers to this specific claim or data point AND (b) flags it as
problematic using language such as: inconsistent, incorrect, mismatch,
hallucinated, fabricated, unsupported, not reproducible, does not match,
not found, wrong value, has problems, or similar criticism.

A claim is only considered caught (airev_caught=true) if the review
explicitly mentions it AND criticises it. If the review merely mentions
the value without criticism, set airev_caught=false.

AI Review text:
\"\"\"
{review_text}
\"\"\"

Atomic claims:
{claims_list}

Reply ONLY with valid JSON — one entry per claim in the same order:
{{
  "judgments": [
    {{
      "idx": <0-based index>,
      "airev_caught": true/false,
      "airev_evidence": "<the relevant sentence(s) from the review that flag this claim, or empty string>"
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_llm = create_client(MODEL, max_tokens=MAX_TOKENS)


def llm_call(prompt: str) -> dict:
    for attempt in range(2):
        try:
            text = _llm.generate(prompt)
            # Find the first '{' to skip any reasoning preamble
            brace = text.find("{")
            if brace >= 0:
                text = text[brace:]
            # Strip trailing markdown fences if present
            if "```" in text:
                text = text[:text.rfind("```")]
            return json.loads(text)
        except Exception as e:
            print(f"  [LLM ERROR attempt {attempt+1}] {e}")
    return {}


def load_fab_claims(sub_id: int) -> list:
    """Load fabrication claims from fabscore_claude/fs_summary.json."""
    for split in ("agents4sci_acc", "agents4sci_rej"):
        path = os.path.join(BASE_DIR, split, f"submission_{sub_id}",
                            "fabscore_claude", "fs_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        claims = []
        for sec in SECTIONS:
            for item in data.get(sec, []):
                if item.get("verdict", "") in FAB_VERDICTS:
                    claims.append({
                        "claim_index":      item.get("claim_index"),
                        "claim":            item.get("claim", ""),
                        "fabscore_verdict": item.get("verdict", ""),
                        "section":          sec,
                    })
        return claims
    return []


def get_review_text(notes: list) -> str:
    return "\n\n".join(
        note.get("content", {}).get("comment", "")
        for note in notes
        if note.get("content", {}).get("comment", "")
    )


def judge_coverage(review_text: str, fab_claims: list) -> list:
    """Single LLM call: did the review mention AND criticise each fabrication claim?"""
    if not fab_claims:
        return []
    claims_list = "\n".join(
        f"[{i}] ({c['fabscore_verdict']}) {c['claim']}"
        for i, c in enumerate(fab_claims)
    )
    result = llm_call(COVERAGE_PROMPT.format(
        review_text=review_text,
        claims_list=claims_list,
    ))
    return result.get("judgments", [])


def compute_stats(all_claims: list, fabrications: list) -> dict:
    caught = len(fabrications)
    total  = len(all_claims)
    return {
        "fabscore_fabrications": total,
        "caught":                caught,
        "missed":                total - caught,
        "recall":                round(caught / total, 4) if total else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    return {r["submission_id"]: r for r in existing}


def main():
    files = sorted(f for f in os.listdir(AIREV_DIR) if f.endswith(".json"))

    done   = load_existing(OUTPUT_PATH)
    output = list(done.values())
    if done:
        print(f"Resuming: {len(done)} submissions already processed.")

    for fname in files:
        with open(os.path.join(AIREV_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        sub_id    = data.get("submission_number")
        task_name = f"submission_{sub_id}"
        title     = data.get("title", "")

        if sub_id in done:
            print(f"  [SKIP] {sub_id}")
            continue

        print(f"\n=== Submission {sub_id}: {title[:60]} ===")

        fab_claims  = load_fab_claims(sub_id)
        review_text = get_review_text(data.get("airev_notes", []))

        if not fab_claims:
            print(f"  No fabrication claims in fs_summary.json — skipping.")
            record = {
                "submission_id": sub_id,
                "task_name":     task_name,
                "title":         title,
                "fabrications":  [],
                "stats":         {"fabscore_fabrications": 0, "caught": 0, "missed": 0, "recall": None},
            }
        else:
            print(f"  FabScore fabrications: {len(fab_claims)}")
            judgments    = judge_coverage(review_text, fab_claims)
            judgment_map = {j["idx"]: j for j in judgments}

            merged = []
            for i, c in enumerate(fab_claims):
                j = judgment_map.get(i, {})
                if j.get("airev_caught", False):
                    merged.append({
                        "claim_index":      c["claim_index"],
                        "claim":            c["claim"],
                        "fabscore_verdict": c["fabscore_verdict"],
                        "section":          c["section"],
                        "airev_evidence":   j.get("airev_evidence", ""),
                    })

            stats = compute_stats(fab_claims, merged)
            print(f"  caught={stats['caught']}  missed={stats['missed']}  recall={stats['recall']}  (of {stats['fabscore_fabrications']} FabScore fabrications)")

            record = {
                "submission_id": sub_id,
                "task_name":     task_name,
                "title":         title,
                "fabrications":  merged,
                "stats":         stats,
            }

        output.append(record)
        done[sub_id] = record

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done. {len(output)} submissions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
