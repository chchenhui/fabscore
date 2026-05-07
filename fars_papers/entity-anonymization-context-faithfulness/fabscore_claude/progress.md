# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf (Entity-Anonymized Context Prompts for Improving Context Faithfulness in Knowledge-Conflict QA, FARS / Analemma)
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- **Tables:** 42 entries extracted from 4 tables (Table 1: main results on ConFiQA-MC 6k with Llama-3.1-8B; Table 2: EACP + ContextFocus composition on 1.5k subset; Table 3: cross-model generalization Llama vs Qwen with delta columns; Table 4: no-harm control on 500-example non-counterfactual subset). Deduplicated Llama raw values from Table 3 that already appear in Table 1; kept unique delta columns and all Qwen entries.
- **Figures:** 3 figures (Figure 1: EACP method overview; Figure 2: C vs B bar chart; Figure 3: cross-model generalization bar chart).
- **Results section:** 2 text-body claims whose delta values are not captured in any table (Condition B vs A: −19.96 Pc; combined method E vs C/D deltas: +3.60 and +21.54 points).

**Next session should:**
- Run analysis/scoring against reference extractions if available.
- Verify no valid numerical results were missed by re-scanning the Experiments section.

## Session 2 — 2026-04-24
**Purpose:** analysis (static)
**Files inspected:**
- `exp/EXPERIMENT_RESULTS/condition_a_oi_baseline/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_b_inventory_ids/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_c_eacp/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_d_contextfocus/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/condition_e_eacp_contextfocus/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/qwen_replication/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/orig_context_noharm/RESULTS.json`
- `exp/eacp/results/orig_context_noharm.csv`
- `exp/eacp/results/confiqa_qa_sanity.csv`
- Full repository directory tree via Explore agent

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Summary of classifications (47 total claims):**
- `static_verifiable`: 44 claims (indices 1-12, 16-47 excluding 13-15)
  - All Table 1 conditions A/B/C metrics: exact matches in per-condition RESULTS.json files
  - All Table 2 conditions A/C/D/E on 1500-subset: exact matches in condition_d and condition_e RESULTS.json
  - All Table 3 Llama and Qwen cross-model deltas: exact matches in qwen_replication/RESULTS.json
  - All Table 4 no-harm control EM values: exact matches in orig_context_noharm RESULTS.json + CSV
  - Figures 1-3: underlying metric data present in RESULTS.json with exact value matches
  - Results-section claims (B vs A: -19.96 Pc; E vs C/D: +3.60/+21.54): computable from RESULTS.json
- `no_code_files`: 3 claims (indices 13-15, Context-DPO reference row in Table 1)
  - Context-DPO is an external reference with no implementation or artifacts in this repo

**Key observation:** The EXPERIMENT_RESULTS/ folder was populated by an agent-based evaluation system. All primary metric files contain exact matches to paper claims; however, raw inference output .jsonl files (referenced in RESULTS.json) are absent from the repository.

**Next session should:**
- If execution is possible: run `run_inference.py` with `run_condition_a.sh` / `run_condition_b.sh` / `run_condition_c.sh` to regenerate raw outputs and verify the RESULTS.json values originate from real model inference.
