"""
classify_fab_patterns.py

For each fabricated claim in fabrication_explanations.json, use an LLM judge to
classify the explanation into the per-verdict pattern categories from
analyze_fabrication_types.py.

  Data Fabrication (D1–D3 + Others):
    D1. Synthetic/mock data used instead of real data described in the paper
    D2. Non-existent or wrong dataset referenced in the paper
    D3. Data values in the paper conflict with actual data files
    D4. Others

  Experiment Fabrication (E1–E5 + Others):
    E1. Simulation or hardcoded values replace an experimental component
    E2. Formula or metric implementation produces incorrect values
    E3. Execution logic or order inconsistent with paper description
    E4. Code implementation bug causes incorrect execution
    E5. Unavailable referenced model or dataset
    E6. Others

  Result Fabrication (R1–R3 + Others):
    R1. Reported value conflicts with stored execution logs or artifacts
    R2. Reported value conflicts with re-execution results
    R3. Mathematically impossible value
    R4. Others

Results are cached incrementally to <CACHE_PATH> and final output saved to <OUTPUT_PATH>.

Usage: uv run python classify_fab_patterns.py
"""

import json
import os
from collections import Counter

from fabscore.utils.llm import create_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH  = "fabrication_explanations.json"
CACHE_PATH  = "fab_patterns_cache.json"
OUTPUT_PATH = "fab_patterns_classified.json"
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 8192

# ---------------------------------------------------------------------------
# Per-verdict categories and prompts
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Data Fabrication": [
        "D1. Synthetic/mock data used instead of real data described in the paper",
        "D2. Non-existent or wrong dataset referenced in the paper",
        "D3. Data values in the paper conflict with actual data files",
        "D4. Others",
    ],
    "Experiment Fabrication": [
        "E1. Simulation or hardcoded values replace an experimental component",
        "E2. Formula or metric implementation produces incorrect values",
        "E3. Execution logic or order inconsistent with paper description",
        "E4. Code implementation bug causes incorrect execution",
        "E5. Unavailable referenced model or dataset",
        "E6. Others",
    ],
    "Result Fabrication": [
        "R1. Reported value conflicts with stored execution logs or artifacts",
        "R2. Reported value conflicts with re-execution results",
        "R3. Mathematically impossible value",
        "R4. Others",
    ],
}

PROMPTS = {
    "Data Fabrication": """\
You are an expert in AI research integrity and reproducibility auditing.

We have examined an ML paper and flagged the following claim as a \
**Data Fabrication** — meaning the data used in the experiment does not match what the \
paper claims. Your task is to read the claim and our explanation carefully, \
then determine which of the following specific reasons best describes WHY this is a \
data fabrication:

D1. Synthetic/mock data used instead of real data described in the paper
    — The code generates data programmatically (e.g. np.random, generate_mock, \
synthetic_cohort) or uses a mock formula to produce numbers, rather than loading the \
real-world dataset the paper describes.

D2. Non-existent or wrong dataset referenced in the paper
    — The dataset file, Hugging Face repository, or external resource the paper claims \
to use does not exist, cannot be found, or is an entirely different dataset from what \
was actually used in the code.

D3. Data values in the paper conflict with actual data files
    — The paper states specific statistics, counts, or numeric values (e.g. class sizes, \
prevalence rates, revenue figures, dataset sizes, split ratios), but the actual \
CSV/database/config/log files on disk show different values, or the code hardcodes \
numbers that do not match the real data.

D4. Others
    — The fabrication involves a data-related issue that does not clearly fit D1–D3.

Now read the following:

Claim (what the paper reports):
{claim}

Our explanation (why we flagged this as fabricated):
{explanation}

Based on the claim and explanation above, select the single most fitting category D1–D4 \
that explains the root cause of this data fabrication.

Reply in JSON exactly like this (no other text):
{{"category": "D1. Synthetic/mock data used instead of real data described in the paper", "reason": "One sentence citing specific evidence from the explanation."}}
""",

    "Experiment Fabrication": """\
You are an expert in AI research integrity and reproducibility auditing.

We have examined an ML paper and flagged the following claim as an \
**Experiment Fabrication** — meaning the experimental result reported in the paper \
cannot be reproduced because the experiment itself was not properly implemented or run. \
Your task is to read the claim and our explanation carefully, then determine \
which of the following specific reasons best describes WHY this is an experiment fabrication:

E1. Simulation or hardcoded values replace an experimental component
    — A key part of the experiment (e.g. training loop, human-AI interaction, policy \
evaluation) is replaced by a programmatic simulation, stub, or hardcoded output, so no \
real experiment was ever conducted. The code produces the numbers directly rather than \
by running the described method.

E2. Formula or metric implementation produces incorrect values
    — The experiment is run, but the formula or metric used to compute the reported \
number is mathematically wrong (e.g. wrong F1 formula, incorrect CI calculation, \
misapplied Jaccard similarity), so the output is systematically incorrect regardless \
of the inputs.

E3. Execution logic or order inconsistent with paper description
    — The code runs, but a critical step described in the paper (e.g. ablation variant, \
fairness penalty, evaluation phase) is missing, ignored, or always bypassed due to a \
logic flaw (e.g. condition always true, variable computed but never used).

E4. Code implementation bug causes incorrect execution
    — The code crashes or silently fails due to a software bug (e.g. AttributeError, \
dimension mismatch, device mismatch, wrong tensor shape) before producing valid results, \
so the reported numbers could not have come from running this code.

E5. Unavailable referenced model or dataset
    — The code references an external resource (e.g. a Hugging Face model ID, checkpoint \
file, pretrained weight) that does not exist or is incorrect, so the experiment cannot \
have been run as described.

E6. Others
    — The fabrication involves an experimental issue that does not clearly fit E1–E5.

Now read the following:

Claim (what the paper reports):
{claim}

Our explanation (why we flagged this as fabricated):
{explanation}

Based on the claim and explanation above, select the single most fitting category E1–E6 \
that explains the root cause of this experiment fabrication.

Reply in JSON exactly like this (no other text):
{{"category": "E1. Simulation or hardcoded values replace an experimental component", "reason": "One sentence citing specific evidence from the explanation."}}
""",

    "Result Fabrication": """\
You are an expert in AI research integrity and reproducibility auditing.

We have examined an ML paper and flagged the following claim as a \
**Result Fabrication** — meaning the numeric result reported in the paper does not \
match what the code actually produces. Your task is to read the claim and our \
explanation carefully, then determine which of the following specific reasons best \
describes WHY this is a result fabrication:

R1. Reported value conflicts with stored execution logs or artifacts
    — Without re-running anything, we found a stored file (e.g. final_info.json, \
results CSV, experiment log, embedded figure in the PDF) that already records a value \
contradicting the paper, or a run is mislabeled / results from different runs are mixed.

R2. Reported value conflicts with re-execution results
    — We actually ran the code and obtained a different number from what the \
paper reports. The discrepancy is discovered at execution time and would not be visible \
without running the code (e.g. "running the script produces X, not the claimed Y").

R3. Mathematically impossible value
    — The reported number is impossible given the paper's own methodology: it exceeds a \
theoretical bound (e.g. probability > 1, F1 > 1), contradicts the paper's own formula \
when applied to the stated inputs, or violates a mathematical constraint such as a BH \
correction ceiling.

R4. Others
    — The fabrication involves a result discrepancy that does not clearly fit R1–R3.

Now read the following:

Claim (what the paper reports):
{claim}

Our explanation (why we flagged this as fabricated):
{explanation}

Based on the claim and explanation above, select the single most fitting category R1–R4 \
that explains the root cause of this result fabrication.

Reply in JSON exactly like this (no other text):
{{"category": "R1. Reported value conflicts with stored execution logs or artifacts", "reason": "One sentence citing specific evidence from the explanation."}}
""",
}

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def llm_classify(client, verdict: str, claim: str, explanation: str) -> dict:
    prompt = PROMPTS[verdict].format(claim=claim, explanation=explanation)
    for attempt in range(2):
        try:
            text = client.generate(prompt=prompt)
            brace = text.find("{")
            if brace == -1:
                raise ValueError("No JSON object in response")
            text = text[brace:]
            if "```" in text:
                text = text[:text.rfind("```")]
            return json.loads(text.strip())
        except Exception as e:
            if attempt == 1:
                suffix = {"Data Fabrication": "D4", "Experiment Fabrication": "E6",
                          "Result Fabrication": "R4"}.get(verdict, "Unknown")
                print(f"    [LLM error after 2 attempts] {e}")
                return {"category": f"{suffix}. Others",
                        "reason": f"Error: {e}"}

# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------
def print_stats(classified: list[dict]) -> None:
    by_verdict: dict[str, list] = {}
    for item in classified:
        v = item.get("verdict", "Unknown")
        by_verdict.setdefault(v, []).append(item)

    print("\n" + "=" * 80)
    print(" Fabrication Pattern Classification")
    print("=" * 80)

    for verdict in ["Data Fabrication", "Experiment Fabrication", "Result Fabrication"]:
        items = by_verdict.get(verdict, [])
        if not items:
            continue
        total  = len(items)
        cats   = CATEGORIES.get(verdict, [])
        counts = Counter(item.get("pattern_category", "Unknown") for item in items)

        print(f"\n  [{verdict}]  n={total}")
        print(f"  {'Category':<72} {'Count':>6} {'%':>6}")
        print("  " + "-" * 86)
        for cat in cats:
            n = counts.get(cat, 0)
            print(f"    {cat:<70} {n:>6}  {n/total*100:>5.1f}%")
        # Surface any unexpected labels
        for cat, n in counts.items():
            if cat not in cats:
                print(f"    [unexpected] {cat:<56} {n:>6}  {n/total*100:>5.1f}%")

    print("\n" + "=" * 80)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if os.path.exists(OUTPUT_PATH):
        print(f"Found existing results at {OUTPUT_PATH}. Printing stats...")
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            classified = json.load(f)
        print_stats(classified)
        return

    with open(INPUT_PATH, encoding="utf-8") as f:
        source: dict[str, list[dict]] = json.load(f)

    all_entries: list[dict] = []
    for verdict, entries in source.items():
        if verdict not in PROMPTS:
            print(f"Warning: no prompt defined for verdict '{verdict}', skipping.")
            continue
        for e in entries:
            all_entries.append({
                "file":        e.get("file", ""),
                "claim":       e.get("claim", ""),
                "explanation": e.get("explanation", ""),
                "verdict":     verdict,
            })

    total = len(all_entries)
    print(f"Total entries to classify: {total}")

    cache: dict[str, dict] = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached classifications.")

    def cache_key(entry: dict) -> str:
        return f"{entry['verdict']}|||{entry['file']}|||{entry['claim']}"

    client = create_client(model_name=MODEL, max_tokens=MAX_TOKENS)

    for i, entry in enumerate(all_entries, 1):
        key = cache_key(entry)
        if key in cache:
            continue
        print(f"[{i}/{total}] ({entry['verdict']}) {entry['claim'][:60]}...")
        result = llm_classify(client,
                              verdict=entry["verdict"],
                              claim=entry["claim"],
                              explanation=entry["explanation"])
        cache[key] = result
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    classified: list[dict] = []
    for entry in all_entries:
        key    = cache_key(entry)
        result = cache.get(key, {"category": "Unknown", "reason": ""})
        classified.append({
            **entry,
            "pattern_category": result.get("category", "Unknown"),
            "pattern_reason":   result.get("reason", ""),
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=2, ensure_ascii=False)
    print(f"\nSaved classified results to {OUTPUT_PATH}")

    print_stats(classified)


if __name__ == "__main__":
    main()
