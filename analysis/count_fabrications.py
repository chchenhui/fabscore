import json
import glob
from collections import defaultdict

# ============================================================
# Step 1: Collect all fabrication explanations
# ============================================================

FABRICATION_TYPES = {
    "Data Fabrication",
    "Experiment Fabrication",
    "Result Fabrication",
}

TARGET_DIRS = ["agents4sci_acc", "agents4sci_rej", "mlragent_papers", "aiscientist_papers", "fars_papers"]
CLAIM_KEYS = ["tables", "figures", "results_section"]

fabrications = defaultdict(list)

for source_dir in TARGET_DIRS:
    for path in glob.glob(f"{source_dir}/**/fabscore_claude/fs_summary.json", recursive=True):
        with open(path) as f:
            data = json.load(f)

        all_claims = []
        for key in CLAIM_KEYS:
            all_claims += data.get(key, [])

        for item in all_claims:
            if "claim_index" not in item:
                continue
            verdict = item.get("verdict", "")
            if verdict in FABRICATION_TYPES:
                fabrications[verdict].append({
                    "file": path,
                    "claim": item.get("claim", ""),
                    "explanation": item.get("explanation", ""),
                })

print("=== Fabrication Count ===")
for verdict, items in fabrications.items():
    print(f"  {verdict}: {len(items)} entries")
print(f"  Total: {sum(len(items) for items in fabrications.values())} entries")

with open("fabrication_explanations.json", "w") as f:
    json.dump(fabrications, f, indent=2, ensure_ascii=False)
print("\nSaved to fabrication_explanations.json\n")