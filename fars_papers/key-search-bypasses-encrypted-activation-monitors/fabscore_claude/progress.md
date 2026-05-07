# Progress Log

## Session 1 — 2026-04-24
**Purpose:** extraction
**Paper inspected:** paper.pdf — "Key-Search Attacks Bypass Encrypted Activation Monitors" (FARS / Analemma)
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (40 entries from Table 1 and Table 2), figures (4 entries: Figure 1–4), and results_section (3 entries from Sections 4.3 and 4.4 body text not already covered by tables)

**Summary of extraction:**
- Table 1: TPR@FPR=1e-3 and AUROC for Unencrypted Monitor, Encrypted (K=1), Key-Search Attack across K=1..512, plus TPR Drop rows.
- Table 2: KL Div., ASR@10, L2 Dist., ΔTPR K=32, ΔTPR K=64, Constraints for Standard/Balanced/High-Diversity encryptors.
- Figures: overview diagram (Fig 1), TPR vs K curve (Fig 2), diversity ablation (Fig 3), score variance vs bypass probability (Fig 4).
- Results section body text: diversity ratio claim (3.7×/3.5–5.0×), correlation r=0.458, and 5× variance difference between bypassed/non-bypassed prompts.

**Next session should:** perform analysis or scoring (fs_analysis.json) based on the extracted results.

## Session 2 — 2026-04-24
**Purpose:** analysis (static analysis, claim classification)

**Files inspected:**
- `paper.pdf` (all 8 pages)
- `exp/EXPERIMENT_RESULTS/baseline_unencrypted_monitor/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/encrypted_monitor_key_search/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/diversity_ablation/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/score_distribution_analysis/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/encryptor_training/RESULTS.json`
- `exp/EXPERIMENT_RESULTS/optimize_trace/iteration_0/RESULTS.json`
- `exp/results/effectiveness_evaluation.json`
- Full repository structure via Explore agent

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created; classifies all 47 claims

**Summary of classifications:**
- **static_verifiable (45 claims):** Claims 1–33, 36–47. All Table 1 values (AUROC, TPR@FPR=1e-3 for K=1..512, TPR drops), Table 2 Standard and High-Diversity rows, Table 2 Balanced KL/ASR/L2 metrics, all Figure descriptions (underlying data confirmed in JSON result files), and all results-section claims. All values are directly confirmed in EXPERIMENT_RESULTS JSON files with exact or near-exact matches.
- **obvious_hallucination / result_fabrication (2 claims):** Claims 34 and 35 — Table 2 "Balanced" row ΔTPR values. The paper claims ~10pp (K=32) and ~15pp (K=64) for the "Balanced" encryptor. The repository data unambiguously identifies the "Balanced" encryptor as the original encryptor (matching KL=0.0227, ASR=0.1869, L2=0.2624 from `encryptor_training/RESULTS.json`), but `optimize_trace/iteration_0/RESULTS.json` explicitly records `original_k32_tpr_drop_fpr_1e3: 0.022` (2.2pp) and `original_k64_tpr_drop_fpr_1e3: 0.028` (2.8pp). This is a 4.5–5.4× discrepancy with the claimed ~10pp and ~15pp. No result file in the repository supports the paper's "Balanced" ΔTPR values.

**Key finding:**
The "Balanced" row in Table 2 appears to be a fabricated intermediate configuration designed to suggest a smooth tradeoff between Standard and High-Diversity configurations, with inflated ΔTPR values (~10pp, ~15pp) that are not supported by any experiment artifact in the repository.

**Next session should:** scoring (fs_execution.json or final scoring step).
