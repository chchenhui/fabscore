"""
analyze_airev_coverage.py

For the agents4science submissions that have both FabScore results and AI reviews,
compare:
  - fabscore_count : fabrications detected by FabScore (Data/Experiment/Result Fabrication)
  - airev_count    : fabrications detected by AI reviews (from airev_coverage.json)

Usage: uv run python analyze_airev_coverage.py
"""

import json, os
import numpy as np

COVERAGE_PATH = "airev_coverage.json"
FAB_VERDICTS  = {"Data Fabrication", "Experiment Fabrication", "Result Fabrication"}
SECTIONS      = ("tables", "figures", "results_section")
ACC_DIR       = "../agents4sci_acc"
REJ_DIR       = "../agents4sci_rej"

# ---------------------------------------------------------------------------
# Load airev_coverage
# ---------------------------------------------------------------------------
with open(COVERAGE_PATH, encoding="utf-8") as f:
    coverage = json.load(f)

airev_map = {r["submission_id"]: r for r in coverage}

# ---------------------------------------------------------------------------
# Load FabScore fabrication counts from fs_summary.json
# ---------------------------------------------------------------------------
def load_fabscore_count(sub_id: int) -> int | None:
    for split in (ACC_DIR, REJ_DIR):
        path = os.path.join(split, f"submission_{sub_id}",
                            "fabscore_claude", "fs_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for sec in SECTIONS:
            for item in data.get(sec, []):
                if item.get("verdict", "") in FAB_VERDICTS:
                    count += 1
        return count
    return None

acc_ids = {
    int(d.replace("submission_", ""))
    for d in os.listdir(ACC_DIR)
    if d.startswith("submission_")
}

# ---------------------------------------------------------------------------
# Build paired dataset
# ---------------------------------------------------------------------------
rows = []
for sub_id, rec in airev_map.items():
    s = rec["stats"]
    # Prefer the count already stored in airev_coverage.json;
    # fall back to counting from fs_summary.json for older entries.
    fs_count = s.get("fabscore_fabrications")
    if fs_count is None:
        fs_count = load_fabscore_count(sub_id)
    if fs_count is None:
        continue   # no fs_summary.json found
    airev_count = s.get("caught", s.get("fabrications_detected", 0))
    rows.append({
        "sub_id":   sub_id,
        "title":    rec["title"][:50],
        "group":    "acc" if sub_id in acc_ids else "rej",
        "fabscore": fs_count,
        "airev":    airev_count,
        "diff":     fs_count - airev_count,   # positive = FabScore found more
    })

rows.sort(key=lambda r: r["sub_id"])
print(f"Paired submissions: {len(rows)}")
print(f"  acc: {sum(1 for r in rows if r['group']=='acc')}")
print(f"  rej: {sum(1 for r in rows if r['group']=='rej')}")
print()

# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------
fs_arr   = np.array([r["fabscore"] for r in rows], dtype=float)
ai_arr   = np.array([r["airev"]    for r in rows], dtype=float)
diff_arr = np.array([r["diff"]     for r in rows], dtype=float)

print("=== Descriptive Stats ===")
print(f"  FabScore fabrications : mean={fs_arr.mean():.2f}  median={np.median(fs_arr):.1f}  total={fs_arr.sum():.0f}")
print(f"  AI review fabrications: mean={ai_arr.mean():.2f}  median={np.median(ai_arr):.1f}  total={ai_arr.sum():.0f}")
print(f"  Difference (FS - AI)  : mean={diff_arr.mean():.2f}  median={np.median(diff_arr):.1f}")
print()

for grp in ("acc", "rej"):
    g = [r for r in rows if r["group"] == grp]
    if not g:
        continue
    gfs = np.array([r["fabscore"] for r in g])
    gai = np.array([r["airev"]    for r in g])
    print(f"  [{grp}] n={len(g)}  FabScore mean={gfs.mean():.2f}  AI mean={gai.mean():.2f}")
print()

# ---------------------------------------------------------------------------
# Overall recall: of all FabScore fabrications, how many did AI reviews catch?
# ---------------------------------------------------------------------------
tasks_with_fab = [r for r in rows if r["fabscore"] > 0]
total_fab      = sum(r["fabscore"] for r in tasks_with_fab)
total_caught   = sum(r["airev"]    for r in tasks_with_fab)
overall_recall = total_caught / total_fab if total_fab else None

print("=== AI Review Recall over FabScore Fabrications ===")
print(f"  Tasks with FabScore fabrications : {len(tasks_with_fab)}")
print(f"  Total FabScore fabrications      : {total_fab}")
print(f"  Total caught by AI reviews       : {total_caught}")
print(f"  Overall recall (caught / total)  : {overall_recall:.4f}" if overall_recall is not None else "  Overall recall: N/A")

caught_by_ai = [r for r in tasks_with_fab if r["airev"] > 0]
missed_entirely = [r for r in tasks_with_fab if r["airev"] == 0]
print(f"  Tasks where AI Review also detected >=1 fab : {len(caught_by_ai)} (out of {len(tasks_with_fab)})")
print(f"  Tasks completely missed by AI Review        : {len(missed_entirely)} and the rate is {len(missed_entirely)/len(tasks_with_fab)*100:.2f}%")
print()


for grp in ("acc", "rej"):
    g = [r for r in tasks_with_fab if r["group"] == grp]
    if not g:
        continue
    gfab     = sum(r["fabscore"] for r in g)
    gcaught  = sum(r["airev"]    for r in g)
    grecall  = gcaught / gfab if gfab else None
    print(f"  [{grp}] tasks={len(g)}  fab={gfab}  caught={gcaught}  "
          f"recall={grecall:.4f}" if grecall is not None else f"  [{grp}] recall: N/A")
print()

# ---------------------------------------------------------------------------
# Per-submission table
# ---------------------------------------------------------------------------
# print("=== Per-submission breakdown ===")
# print(f"{'sub_id':>8}  {'grp':>3}  {'FabScore':>8}  {'AIRev':>5}  {'diff':>5}  title")
# print("-" * 80)
# for r in sorted(rows, key=lambda x: -x["diff"]):
#     print(f"{r['sub_id']:>8}  {r['group']:>3}  "
#           f"{r['fabscore']:>8}  {r['airev']:>5}  {r['diff']:>+5}  {r['title']}")
# print()

# ---------------------------------------------------------------------------
# Per-category recall
# ---------------------------------------------------------------------------
cat_total = {"Data Fabrication": 0, "Experiment Fabrication": 0, "Result Fabrication": 0}
cat_caught = {"Data Fabrication": 0, "Experiment Fabrication": 0, "Result Fabrication": 0}

for sub_id, rec in airev_map.items():
    # Only count if the submission actually has an fs_summary (i.e. is in our paired rows)
    if not any(r["sub_id"] == sub_id for r in rows):
        continue

    # 1. Add caught from airev_coverage.json's fabrications array
    for fab in rec.get("fabrications", []):
        v = fab.get("fabscore_verdict")
        if v in cat_caught:
            cat_caught[v] += 1
            
    # 2. Add total from fs_summary.json
    for split in (ACC_DIR, REJ_DIR):
        path = os.path.join(split, f"submission_{sub_id}", "fabscore_claude", "fs_summary.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for sec in SECTIONS:
                for item in data.get(sec, []):
                    v = item.get("verdict", "")
                    if v in cat_total:
                        cat_total[v] += 1
            break

print("=== Per-Category Recall ===")
for cat in ["Data Fabrication", "Experiment Fabrication", "Result Fabrication"]:
    t = cat_total[cat]
    c = cat_caught[cat]
    r = c / t if t > 0 else 0
    print(f"  {cat:22s} : caught {c:4d} / total {t:4d}  (Recall: {r*100:.1f}%)")
print()

