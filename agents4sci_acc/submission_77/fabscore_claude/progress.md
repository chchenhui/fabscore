# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction

**Paper inspected:** `77_Visible_Yet_Unreadable_A_Sy.pdf` — "Visible Yet Unreadable: A Systematic Blind Spot of Vision–Language Models Across Writing Systems"

**Summary:** The paper benchmarks VLMs on two psychophysics-inspired tasks: Chinese idiom character-fusion and English word color-overlay fusion. It reports recognition accuracy across 9 VLMs under multiple prompting strategies, plus a 100% human baseline.

**JSON files created:**
- `fabscore_claude/fs_extracted.json` — contains tables (81 entries from Table 3), figures (6 figures), and results_section (4 text-body numerical claims not already in tables).

**Key extraction notes:**
- Table 3 covers Chinese idiom fusion task with strict match, average match, and human evaluation (100%) for all models/prompts.
- Figure 4 provides English word fusion accuracy (up to 20% for GPT-5 detailed); exact per-model values readable from the bar chart but only the 20% cap and qualitative ordering are stated in the text body.
- Figures 5 and 6 show per-word/per-idiom difficulty (0% hardest; 38.9%–55.6% easiest for English; 2.5%–8% easiest for Chinese idioms).
- No additional numerical tables exist in the paper.

**Next session should:** verify or extend extraction if supplementary materials are added; no further action needed for base extraction.

## Session 2 — 2026-04-24
**Purpose:** analysis

**Files inspected:**
- `77_Visible_Yet_Unreadable_A_Sy.pdf` — full paper re-read (all 16 pages)
- `character-fusion-generator-main/cin-gen.py` — Chinese character fusion generator
- `character-fusion-generator-main/en-color-gradient-fusion.py` — English word fusion generator
- `character-fusion-generator-main/cin_100.txt` — 100 Chinese idioms
- `character-fusion-generator-main/en_100.txt` — 100 English words
- `character-fusion-generator-main/README.md`
- Repository structure via Explore agent

**JSON files created:**
- `fabscore_claude/fs_analysis.json` — full claim classifications for all 91 claims

**Key findings:**
- The repository contains ONLY stimulus generation code and word/idiom lists. There are NO:
  - VLM evaluation/inference scripts (no code to call GPT-4o, GPT-5, Claude, Gemini, LLaVA, Qwen APIs)
  - Result files (no JSON/CSV/log with recognition accuracy numbers)
  - Per-model or per-stimulus output data
  - Figure generation scripts with underlying data
  - Human evaluation data
- Table 3 (81 claims, indices 1-81): ALL classified as `no_code_files` — no evaluation pipeline or results exist
- Figure 1 (claim 82): `no_code_files` — conceptual diagram made with Gemini per AI checklist, no code
- Figure 2 (claim 83): `execution_required` — `cin-gen.py` implements the described lr/tb/diag fusion modes with idiom "滥竽充数" in cin_100.txt
- Figure 3 (claim 84): `execution_required` — `en-color-gradient-fusion.py` implements the red/green overlay with "hardware" in en_100.txt
- Figures 4-6 (claims 85-87): `no_code_files` — require VLM evaluation data not present
- Results section (claims 88-91): `no_code_files` — all derived from VLM runs, no code or data present

**Classification summary:**
- no_code_files: 89 claims (indices 1-82, 85-91)
- execution_required: 2 claims (indices 83, 84)
- static_verifiable: 0
- insufficient_evidence: 0
- obvious_hallucination: 0
- error: 0

**Next session:** No further analysis needed. If execution capability is added, run cin-gen.py and en-color-gradient-fusion.py to verify claims 83 and 84. The VLM evaluation results (Table 3) cannot be verified without running API-dependent inference code that is absent from the repository.

## Session 3 — 2026-04-24
**Purpose:** execution — verify claim 83 (Figure 2: character fusion for "滥竽充数")

**Files inspected:**
- `fabscore_claude/progress.md` — prior session context
- `character-fusion-generator-main/cin-gen.py` — fusion code (lr/tb/diag modes)
- `character-fusion-generator-main/cin_100.txt` — confirmed "滥竽充数" at line 7

**Commands run:**
```
python3 character-fusion-generator-main/cin-gen.py --font /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc --sentence "滥竽充数" --all_modes --out_dir fabscore_claude/workspace/fig2_output
```

**Artifacts created:**
- `fabscore_claude/workspace/fig2_output/滥竽充数_660138d6_all_modes.png` (14404 bytes) — 2×3 grid image
- `fabscore_claude/workspace/fig2_output/labels.jsonl` — metadata confirming pairs ["滥+竽", "充+数"] with modes ["lr", "tb", "diag", "lr", "tb", "diag"]
- `fabscore_claude/workspace/claim_83_command_output.txt` — raw command output

**Verdict summary:** VERIFIED. The script successfully generated a 2-row × 3-column grid showing each character pair (滥+竽 and 充+数) fused using all three modes (lr=left-right, tb=top-bottom, diag=diagonal). This matches exactly what Figure 2 claims to illustrate: systematic fusion of "滥竽充数" pairs along different axes.

**Next session:** Claim 83 is resolved. Claim 84 (Figure 3, English word fusion) still needs execution verification via en-color-gradient-fusion.py.

## Session 4 — 2026-04-24
**Purpose:** execution — verify claim 84 (Figure 3: English word fusion for "hardware")

**Files inspected:**
- `fabscore_claude/progress.md` — prior session context
- `character-fusion-generator-main/en-color-gradient-fusion.py` — full code review

**Commands run:**
```
python3 character-fusion-generator-main/en-color-gradient-fusion.py --words hardware --color_scheme1 pure_red --color_scheme2 pure_green --out_dir ./fabscore_claude/workspace/fig3_output --save_originals
```

**Artifacts created:**
- `fabscore_claude/workspace/fig3_output/1.png` (14586 bytes) — fused overlay of "hard"(red) + "ware"(green)
- `fabscore_claude/workspace/fig3_output/1_orig2parts.png` (13349 bytes) — side-by-side originals
- `fabscore_claude/workspace/fig3_output/labels.jsonl` — confirms word="hardware", groups="hard+ware", colors="pure_red+pure_green"
- `fabscore_claude/workspace/claim_84_command_output.txt` — raw command output

**Verdict summary:** VERIFIED. The script successfully generated the Figure 3 illustration: "hardware" is split into "hard" (first 4 letters, red) and "ware" (last 4 letters, green), then overlaid into a single fused form. This exactly matches the claim description.

**Next session:** No further execution needed for claims 83 and 84. All other claims (1-82, 85-91) are no_code_files and do not require execution.
