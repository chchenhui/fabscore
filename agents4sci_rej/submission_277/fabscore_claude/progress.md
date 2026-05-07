# Progress Log

## Session 1 — 2026-04-23
**Purpose:** extraction
**Paper inspected:** 277_Breaking_Points_How_Transf.pdf (11 main pages + appendices, PDF format)
**Context:** Paper analyzes transformer vulnerability patterns under noisy inputs, identifies layer-wise transitions at layers 3 and 8, and proposes strategic layer dropout for inference speedup.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables (145 entries across Tables 1–6), figures (6 entries: Figures 1–6), and results_section (21 entries from Sections 4.1–4.6)

**Key extraction notes:**
- Tables 1–6 fully extracted including Table 5 (extended 13-model comparison) and Table 6 (cross-lingual patterns)
- Figures 1–6 captured with their captions
- Results section covers Sections 4.1 (Main Results), 4.2 (Layer-wise Vulnerability), 4.3 (Runtime Validation), 4.4 (Real-World Noise), 4.5 (Comparison with Existing Techniques), 4.6 (Ablation Studies)
- Skipped Sections 5 (Theoretical Analysis) and 6 (Discussion) as instructed
- Deduplicated cross-table/text overlaps (e.g., model averages from Table 1 not repeated in results_section)

**Next session should:**
- Run analysis/scoring (fs_analysis) on the extracted results
- Verify no numerical claims were missed in the appendix experimental results (Appendix A)

---

## Session 2 — 2026-04-23
**Purpose:** analysis (static code audit)
**Paper inspected:** 277_Breaking_Points_How_Transf.pdf (via extracted claims)
**Files inspected:**
- `noise_experiment_v0_v1/nips_publication_experiment.py` — primary experiment script
- `noise_experiment_v0_v1/nips_publication_results.json` — synthetic results from nips_publication_experiment.py
- `noise_experiment_v0_v1/publication_ready_experiment.py` — secondary real-model experiment script
- `noise_experiment_v0_v1/publication_ready_results.json` — results from publication_ready_experiment.py
- `noise_experiment_v0_v1/fixed_comprehensive_results.json` — another results file
- `noise_experiment_v0_v1/experiment_results.json` — another results file
- `noise_experiment_v0_v1/sections/experiments.tex` — LaTeX source with Table 1 & 2 values
- `noise_experiment_v0_v1/nips_figures/latex_tables.tex` — additional LaTeX tables
- `noise_experiment_v0_v1/runtime_validation.py` — real speedup measurement code
- `noise_experiment_v0_v1/real_world_noise_exp.py` — real-world noise evaluation code
- `noise_experiment_v0_v1/advanced_comprehensive_experiment.py` — extended model experiment

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created with complete classification of all 172 claims

**Critical findings:**

1. **Primary experiment code is synthetic**: `nips_publication_experiment.py` uses `generate_layer_robustness_scores()` which creates fake data using hardcoded numpy arrays (base_robustness values per model, noise_impact factors, layer_patterns arrays) plus `np.random.normal()`. NO actual transformer model is loaded or run. This is experiment_fabrication for all claims derived from this code (Tables 1, 2 and Figures 2, 4, 5).

2. **Paper values conflict with supplementary LaTeX**: The `sections/experiments.tex` file contains Table 1 and 2 values that differ significantly from the submitted paper:
   - BERT Syntax: paper=0.518, experiments.tex=0.218 (inflated by 0.300)
   - BERT Attention: paper=0.634, experiments.tex=0.534 (inflated by 0.100)
   - RoBERTa Char: paper=0.876, experiments.tex=0.976 (deflated by 0.100)
   - RoBERTa Syntax: paper=0.689, experiments.tex=0.989 (deflated by 0.300)
   - DistilBERT Layer 8: paper=0.287, experiments.tex=--- (no data)

3. **No code for extended tables**: Tables 3 (specific statistical tests), 4 (hyperparameter sensitivity), 5 (13 extended models including BERT-large, XLM-R, DeBERTa-v3, GPT-2-medium, GPT-Neo, OPT-350M, T5-base, BART-base, mT5-small), and 6 (cross-lingual analysis) have NO corresponding code or results files.

4. **Legitimate code exists for speedup and real-world noise**: `runtime_validation.py` and `real_world_noise_exp.py` use real transformer models. Claims about these (Figure 1, 6 and results section 156-163) are classified as execution_required.

**Classification summary:**
- obvious_hallucination (experiment_fabrication): 65 claims (Table 1, Table 2, Figures 2/4/5, results 152-155, ablation 169-172)
- no_code_files: 98 claims (Table 3, 4, 5, 6, Figure 3, specific baseline comparisons)
- execution_required: 9 claims (Figure 1, Figure 6, speedup claims 156-159, real-world noise 160-161/163)
- Total: 172 claims ✓

**Next session should:**
- If execution is possible: run `runtime_validation.py` to verify speedup claims (156-159, Figure 1)
- If execution is possible: run `real_world_noise_exp.py` to verify real-world noise claims (160-161, 163, Figure 6)
- No further static analysis needed — all claims fully classified

---

## Session 3 — 2026-04-23
**Purpose:** execution (Claim 146 — Figure 1 speedup verification)
**Claim:** Strategic dropout achieves 1.28× average speedup, approaching theoretical 1.33× maximum at larger batches.

**Files inspected:**
- `noise_experiment_v0_v1/runtime_validation.py` — real BERT inference speedup measurement code

**Commands executed:**
- `python fabscore_claude/workspace/verify_claim_146.py` (custom minimal script on GPU)

**Artifacts created:**
- `fabscore_claude/workspace/verify_claim_146.py` — verification script
- `fabscore_claude/workspace/claim_146_results.json` — measured speedup values
- `fabscore_claude/workspace/claim_146_command_output.txt` — full command output

**Key execution findings:**
The code defines 4 configurations:
- `random_15%`: drops 3 out of 12 layers [1, 4, 9]
- `strategic_15%`: drops 6 out of 12 layers [1, 2, 4, 5, 9, 10] (skips transitions at 3 and 8)
- `aggressive_25%`: drops 8 out of 12 layers

Measured speedups (batch sizes 1, 8, 16, 32 at seq_len=128):
- `random_15%` (3 layers): avg **1.312×**, max **1.324×** (theoretical 12/9 = 1.333×)
- `strategic_15%` (6 layers): avg **1.927×**, max **1.961×** (theoretical 12/6 = 2.000×)
- `aggressive_25%` (8 layers): avg **2.784×**, max **2.891×**

**Verdict analysis:**
- Paper claims "strategic dropout achieves 1.28× average speedup, approaching theoretical 1.33× maximum at larger batches"
- The paper's theoretical max of 1.33× is only mathematically consistent with dropping 3/12 layers (12/9 = 1.333×), corresponding to `random_15%` in the code
- The code's `strategic_15%` configuration (6 layers dropped, labeled as strategic, skips transition layers) achieves **1.927× average speedup**, NOT the claimed 1.28×
- The paper appears to attribute speedup numbers from a 3-layer dropout scenario to "strategic" dropout
- **Verdict: Result Fabrication** — the `strategic_15%` config achieves ~1.93× speedup, not 1.28× as claimed; the paper's 1.33× theoretical matches `random_15%` (3 layers), not the strategic config (6 layers)

**Next session should:**
- Verify claims about Figure 6 and real-world noise (claims 160-161, 163) using `real_world_noise_exp.py`

---

## Session 4 — 2026-04-23
**Purpose:** execution (Claim 151 — Figure 6 real-world noise impact comparison)
**Claim:** Figure 6 shows Left: Model performance on different real-world noise sources. Right: Systematic gap between synthetic and real-world noise challenges.

**Files inspected:**
- `noise_experiment_v0_v1/real_world_noise_exp.py` — real-world noise evaluation code
- Checked for saved result CSV/JSON files (none found)

**Commands executed:** None (code inspection sufficient)

**Artifacts created:** None

**Key findings:**
The `real_world_noise_exp.py` code at lines 281-303 contains a comment `# Comparison with synthetic noise (placeholder data)` and hardcodes the values for the right panel of Figure 6:
- `'Real-World': [0.85, 0.72, 0.88, 0.75, None, None]`
- `'Synthetic': [None, None, None, None, 0.92, 0.78]`

These hardcoded values are used to generate the "systematic gap" visualization (right panel of Figure 6), which is NOT derived from any actual model evaluation. The comment explicitly labels them "placeholder data." No saved result CSV exists from previous runs.

The left panel of Figure 6 (model performance on real-world noise sources) uses actual BERT/RoBERTa model evaluation, but the right panel (the "systematic gap" claim) is fabricated from preset values.

**Verdict:** Experiment Fabrication — the right panel of Figure 6 (systematic gap between synthetic and real-world noise) is generated from hardcoded placeholder data, not real model evaluations. The metric implementation bypasses actual computation for the key comparative claim.

**Next session should:**
- Verify remaining real-world noise claims (160-161, 163) which may also be impacted by this placeholder data finding.

---

## Session 5 — 2026-04-23
**Purpose:** execution (Claim 156 — strategic dropout 1.28× speedup for 25% layer removal)
**Claim:** "Strategic dropout targeting non-transition layers (dropping layers 1, 5, and 10) achieves 1.28× measured speedup compared to the theoretical maximum of 1.33× for 25% layer removal."

**Files inspected:**
- `noise_experiment_v0_v1/runtime_validation.py` — read configurations (lines 86-91)
- `fabscore_claude/workspace/claim_146_results.json` — reused execution results from Session 3

**Commands executed:** None (reused existing artifacts from Session 3)

**Artifacts reused:**
- `fabscore_claude/workspace/claim_146_results.json` — directly relevant; measured speedups for all configurations at seq_len=128

**Key findings:**
1. The code's configurations at runtime_validation.py:86-91:
   - `random_15%`: drops [1, 4, 9] — 3 layers (25% of 12)
   - `strategic_15%`: drops [1, 2, 4, 5, 9, 10] — 6 layers (50% of 12)
   - No config drops "layers 1, 5, and 10" as the claim specifies

2. Actual measured speedups (from claim_146_results.json):
   - `random_15%` (3 layers, theoretical 1.333×): avg 1.312×, NOT 1.28×
   - `strategic_15%` (6 layers, labeled strategic): avg 1.927×, NOT 1.28×

3. The claim's 1.28× value matches neither configuration. The "1.33× theoretical maximum for 25% layer removal" is only consistent with 3 layers dropped (12/9 = 1.333×), but the code's 3-layer config (`random_15%`) achieves 1.312×, not 1.28×. The "strategic" config drops 6 layers and achieves ~1.93×.

4. The specific layer selection "1, 5, and 10" (3 layers, non-transition) has no matching configuration in the code.

**Verdict:** Result Fabrication — the code framework is present but the claimed 1.28× measured speedup doesn't match any configuration's execution output. The strategic_15% (which skips transition layers) achieves 1.927×, and the 3-layer random_15% achieves 1.312×.

**Next session should:**
- Verify remaining real-world noise claims (160-161, 163)

---

## Session 6 — 2026-04-23
**Purpose:** execution (Claim 157 — random 25% dropout 1.19× speedup claim)
**Claim:** "Random 25% dropout yields only 1.19× speedup while causing 8 percentage points of absolute performance degradation."

**Files inspected:**
- `noise_experiment_v0_v1/runtime_validation.py` — code configurations (lines 86-91)
- `fabscore_claude/workspace/claim_146_results.json` — existing execution results from Session 3

**Commands executed:** None (reused existing artifacts from Session 3)

**Artifacts reused:**
- `fabscore_claude/workspace/claim_146_results.json` — contains measured speedups for all configurations at seq_len=128

**Key findings:**
1. Code's `random_15%` configuration drops [1, 4, 9] = 3/12 = 25% of layers — this is the "random 25% dropout" referenced in the claim.
2. Measured speedup for `random_15%`: avg **1.312×** (range 1.298×–1.324×) — NOT the claimed 1.19×.
3. `runtime_validation.py` contains ONLY inference time measurement — NO accuracy/performance evaluation. The "8 percentage points of absolute performance degradation" has zero corresponding measurement code in the repository.
4. No other configuration matches "random 25% dropout" with 1.19× speedup.

**Verdict:** Result Fabrication — the random 25% dropout configuration achieves 1.312× speedup in execution, not 1.19×. The 8 pp performance degradation claim is completely unsupported (no measurement code exists in runtime_validation.py).

**Next session should:**
- Verify remaining real-world noise claims (160-161, 163)

---

## Session 7 — 2026-04-23
**Purpose:** execution (Claim 158 — "Aggressive 50% dropout (6 layers) achieves 1.85× speedup but causes 18 percentage points of absolute performance loss.")

**Files inspected:**
- `noise_experiment_v0_v1/runtime_validation.py` — configurations at lines 86-91
- `fabscore_claude/workspace/claim_146_results.json` — execution results from Session 3

**Commands executed:** None (reused existing artifacts from Session 3)

**Artifacts reused:**
- `fabscore_claude/workspace/claim_146_results.json` — sufficient; contains measured speedups for all configurations

**Key findings:**
1. The code's `aggressive_25%` configuration (line 90) drops layers `[1, 2, 4, 5, 6, 9, 10, 11]` = **8 layers** (not 6 as the claim states).
2. Measured speedup for `aggressive_25%` (8 layers): avg **2.784×**, NOT the claimed 1.85×.
3. The configuration that drops 6 layers is `strategic_15%` (drops [1, 2, 4, 5, 9, 10]), which achieves avg **1.927×** — closest to 1.85× but still not matching, and it is labeled "strategic" not "aggressive".
4. No configuration achieves 1.85× speedup.
5. `runtime_validation.py` has ONLY inference time measurement code — NO accuracy/performance evaluation. The "18 percentage points of absolute performance loss" is completely unsupported by any code in the repository.

**Verdict:** Result Fabrication — the `aggressive_25%` config achieves 2.784× speedup (not 1.85×), and the 6-layer `strategic_15%` achieves 1.927× (not 1.85×). The 18 pp performance loss claim has no supporting measurement code in runtime_validation.py.

**Next session should:**
- Verify remaining real-world noise claims (160-161, 163)

---

## Session 9 — 2026-04-23
**Purpose:** execution (Claim 160 — OCR Errors ICDAR 2019 dataset BERT robustness 0.640 → 0.560)
**Claim:** "9. OCR Errors (ICDAR 2019 dataset): Common substitutions from 2,000 scanned documents reduce BERT robustness score from 0.640 (its average robustness on clean synthetic data) to 0.560 (12.5% relative drop)."

**Files inspected:**
- `noise_experiment_v0_v1/real_world_noise_exp.py` — primary candidate file
- `noise_experiment_v0_v1_v2/real_world_noise_exp.py` — v2 version (identical)
- Searched for ICDAR dataset files across entire repository (glob `*icdar*` → no results)

**Commands executed:** None (code inspection sufficient)

**Artifacts created:** None

**Key findings:**
1. **ICDAR 2019 dataset absent**: No ICDAR dataset file or reference exists anywhere in the repository. The glob `*icdar*` found zero files.
2. **Synthetic data, not real-world documents**: The `create_real_world_datasets()` function at line 128 uses 8 hardcoded clean sentences (`"The quick brown fox..."`, etc.) repeated 25 times = **200 samples total**, not "2,000 scanned documents" from ICDAR 2019.
3. **Paper vs. code dataset mismatch**: The paper claims the ICDAR 2019 dataset (a real competition dataset for OCR/document analysis), but the code fabricates its own synthetic noise from hardcoded text — these are fundamentally different data sources.
4. **No saved results**: No CSV or JSON results file from running this code exists in the repository.
5. **Specific values unverifiable**: The claimed 0.640 → 0.560 BERT robustness values have no supporting artifact.

**Verdict:** Data Fabrication — The paper claims to use the ICDAR 2019 dataset (2,000 scanned documents) but the code uses synthetic text (8 hardcoded sentences × 25 = 200 samples). ICDAR 2019 is a real, public dataset that is never referenced in any repository file. The dataset object itself is fabricated.

**Next session should:**
- Verify claim 161 and 163 (remaining real-world noise claims)

---

## Session 8 — 2026-04-23
**Purpose:** execution (Claim 159 — "Speedup improves slightly with batch size, reaching 1.31× at batch=32")
**Claim:** "Speedup improves slightly with batch size, reaching 1.31× at batch=32 due to better GPU utilization amortizing fixed overhead costs."

**Files inspected:**
- `noise_experiment_v0_v1/runtime_validation.py` — configurations (lines 86-91) and batch size logic (lines 95-120)
- `fabscore_claude/workspace/claim_146_results.json` — existing GPU execution results
- `fabscore_claude/workspace/claim_146_command_output.txt` — confirmed "Device: CUDA"

**Commands executed:** None (reused existing GPU artifacts from Session 3)

**Artifacts reused:**
- `fabscore_claude/workspace/claim_146_results.json` — directly relevant; contains measured speedups for batch sizes [1, 8, 16, 32] on CUDA

**Key findings:**
1. The trend is correct: speedup does improve slightly with batch size for all configurations.
2. The `random_15%` config (only one in ~1.3× range) shows:
   - batch=1: 1.298×, batch=8: 1.308×, batch=16: 1.318×, batch=32: 1.324×
3. The claimed value at batch=32 is 1.31×, but the actual GPU-measured value is 1.324×, which rounds to 1.32×.
4. No configuration reaches exactly 1.31× at batch=32 (all are either ~1.32×, ~1.96×, or ~2.89×).

**Verdict:** Result Fabrication — the trend (improving with batch size) is correct, but the specific claimed value of 1.31× at batch=32 does not match the measured 1.324×. The paper reports 1.31× while the execution yields 1.32×.

**Next session should:**
- Verify remaining real-world noise claims (160-161, 163)

---

## Session 11 — 2026-04-23
**Purpose:** execution (Claim 163 — Social Media Text: 5,000 tweets with 18% average degradation, RoBERTa 8% loss)

**Files inspected:**
- `noise_experiment_v0_v1/real_world_noise_exp.py` — real-world noise evaluation code (lines 128-178, 154-157)
- `fabscore_claude/progress.md` — prior session findings (Sessions 9 and 10 already established synthetic data issue)

**Commands executed:** None (code inspection sufficient; same code path as claims 160 and 161)

**Artifacts created:** None (no new artifacts needed)

**Key findings:**
1. **No Twitter dataset**: The claim states "5,000 tweets" but no Twitter data, CSV, or API call exists anywhere in the repository.
2. **Synthetic data, not real tweets**: The `create_real_world_datasets()` function at line 154-157 creates the social_media dataset using only 8 hardcoded clean sentences repeated 25 times = **200 samples total**, NOT 5,000 tweets.
3. **Same synthetic source as claims 160 and 161**: All three noise types (OCR, social media, typing errors) use the same 8 hardcoded sentences as their source, not any real-world dataset.
4. **Social media noise is programmatic**: `apply_social_media_noise()` (line 69) applies word substitutions from a hardcoded dictionary to these synthetic sentences, not actual tweet data.
5. **No saved results**: No CSV or JSON output from running this code exists. The 18% degradation and 8% RoBERTa values are unsupported by any artifact.
6. **Dataset mismatch**: The paper claims a real-world Twitter dataset of 5,000 tweets, but the code fabricates social media-style text from 8 generic sentences — fundamentally different data sources.

**Verdict:** Data Fabrication — The paper claims to use 5,000 tweets (real social media data), but the code uses 200 synthetic samples derived from 8 hardcoded generic sentences. No Twitter dataset exists anywhere in the repository.

**Next session should:**
- All major claims for this submission have been verified. No further sessions needed.

---

## Session 10 — 2026-04-23
**Purpose:** execution (Claim 161 — RoBERTa drops from 0.787 to 0.735 under OCR errors)
**Claim:** "10. RoBERTa drops from 0.787 to 0.735 (6.6% relative drop) under OCR errors."

**Files inspected:**
- `noise_experiment_v0_v1/real_world_noise_exp.py` — real-world noise evaluation code (same file as Session 9)
- Prior session findings (Session 9) already established the synthetic dataset issue

**Commands executed:** None (code inspection sufficient; same code path as claim 160)

**Artifacts created:** None (no new artifacts; same code as prior session)

**Key findings:**
1. **Synthetic data, not real-world**: The `create_real_world_datasets()` function at line 128 uses 8 hardcoded clean sentences repeated 25 times = 200 samples total. This is NOT from any real scanned documents or OCR dataset.
2. **Clean baseline 0.787 is fabricated**: The RoBERTa clean baseline (0.787) appears to come from `nips_publication_experiment.py` which uses `np.random.normal()` to generate fake robustness scores (established in Session 2).
3. **OCR noise is programmatic, not real-world**: The `apply_ocr_noise()` function applies hardcoded character substitutions to synthetic text, not noise from actual scanned documents.
4. **No saved results**: No CSV or JSON results exist from running this code. The claimed specific values (0.787→0.735) are unsupported by any saved artifact.
5. **Same issue as Claim 160**: The entire "Real-World Noise" evaluation uses synthetic data.

**Verdict:** Data Fabrication — same issue as Claim 160. The code generates "OCR errors" synthetically from 8 hardcoded sentences, not from any real-world dataset of scanned documents. The clean baseline 0.787 comes from fabricated synthetic experiment using np.random. The "real-world" framing in the paper is unsupported.

**Next session should:**
- Verify claim 163 (remaining real-world noise claim)
