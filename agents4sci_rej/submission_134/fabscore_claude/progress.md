## Session: extraction — 2026-04-24

**Purpose:** Extract experimental results from paper.

**Paper inspected:** `134_Research_on_CAPTCHAs_Targe.pdf`
- Title: "Research on CAPTCHAs Targeted at AI: Human–Easy, Model–Hard Visual Tasks for the AI Era"
- 9 pages; evaluates 5 multimodal models vs. 5 human participants on 5 visual task families

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — contains tables (20 entries from Table 1 & Table 2), figures (1 entry: Figure 1), and results_section (empty — all body-text numerical results in Section 5 are duplicates of Table 1 entries)

**Extraction notes:**
- Table 1: Human vs. AI accuracy by task family (size, color, combined, bird-leg, hand-finger); includes Human %, AI %, and Gap columns.
- Table 2: Overall accuracy per model across all 50 items.
- Figure 1: Bar chart of human-minus-AI correct-ratio differences per item.
- Results section body text (Section 5) contains approximate values (≈85.6%, ≈5.6%, ≈15.6%, ≈20.8%) that correspond to rounded Table 1 values — excluded as duplicates per deduplication rules.

**Next session:** Review/scoring or analysis of extracted results.

---

## Session: analysis — 2026-04-24

**Purpose:** Static analysis of all 21 extracted claims against repository artifacts.

**Files inspected:**
- `materials/original data/AI-outcome/result.csv` — per-picture per-model accuracy + summary row (row 52) with overall model averages
- `materials/original data/difference_calculate/top_diff.csv` — per-picture human_acc, ai_acc, diff (50 rows)
- `materials/original data/difference_calculate/kafang_test_outcome/human_ai_diff_analysis.csv` — per-picture diff + Fisher/chi2 p-values + significance flags
- `materials/original data/difference_calculate/kafang_test_outcome/diff_plot.png` — generated Figure 1

**Verification summary:**

Table 1 (claims 1–15): All human and AI accuracy values were re-computed from top_diff.csv per-task-family:
- A_size_only: human=78%, AI=5.6%≈6%, gap=+72 ✓
- B_color_only: human=100%, AI=85.6%≈86%, gap=+14 ✓
- C_size_and_color: human=86%, AI=0%, gap=+86 ✓
- Birds: human=94%, AI=15.6%≈16%, gap=+78 ✓
- Hands: human=98%, AI=20.8%≈21%, gap=+77 ✓

Table 2 (claims 16–20): Exact matches in result.csv summary row:
- GPT-5=0.334, Claude=0.248, Gemini=0.314, Doubao=0.204, Qwen=0.176 ✓

Figure 1 (claim 21): Underlying data fully present in human_ai_diff_analysis.csv (diff values + significant column for green bar coloring) and diff_plot.png exists.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — all 21 claims classified as `static_verifiable`

**Classifications:** 21/21 → `static_verifiable`

**Next step:** No further action needed; all claims are statically verified.
