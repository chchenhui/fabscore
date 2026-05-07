# Progress Log

## Session 1 — 2026-04-03
**Purpose:** extraction

**Paper inspected:** `242_Simulating_Two_Sided_Job_M.pdf` — "Simulating Two-Sided Job Marketplaces with AI Agents" (Upwork team). A simulation framework for studying LLM agent behavior in economic marketplaces, with comparative experiments across 5 agent configurations.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created) — contains tables (30 entries from Table 1), figures (4 figures), and results_section (2 entries from Section 4.4 body text).

**Key extraction notes:**
- Only one table in the paper (Table 1): comparative performance metrics across 5 agent configurations (LLM-F+LLM-C w/ Refl., w/o Refl., LLM-F+Rand-C, Rand-F+LLM-C, Rand-F+Rand-C) for 6 metrics (Fill Rate, Bid Efficiency, Bids per Job, Participation Rate, Freelancer Hiring Rate, Work Distribution Gini).
- Most numerical results in Sections 4.2 and 4.3 body text are direct restatements of Table 1 values → excluded from results_section per deduplication rules.
- Only 2 body-text claims with unique numerical values not covered by Table 1: (1) 37% freelancer tier advancement, (2) ~10 bids/job for Data Science vs ~3 for Engineering & Architecture.
- Figure captions for Figure 3 contain percentage breakdowns (63%, 27%, 9.5%, 0.5%) but these are captions and excluded from results_section.

**Next session should:**
- Proceed to analysis (`fs_analysis.json`) — assess quality of claims, coverage of metrics, and relationship between experimental configurations.

## Session 2 — 2026-04-03
**Purpose:** analysis

**Files/context inspected:**
- `supplementary_materials/analysis_results/enhanced_aggregated_results.json` — primary result artifact with all 5 configurations × 6+ metrics with mean and 95% CI (20–21 runs each)
- `supplementary_materials/analysis_results/aggregated_results.json` — earlier 3–5 run results for 4 configurations (different parameter names)
- `supplementary_materials/enhanced_multi_run_analysis_report.txt` — human-readable summary confirming all Table 1 values
- `supplementary_materials/analysis_results/analysis_results.json` — single small-run (25 rounds, 116 jobs) category-level analysis data; Data Science shows 5.45 bids/job (not 10)
- `supplementary_materials/category_breakdown_analysis.py` — reads `results/simuleval/true_gpt_simulation_20250905_001404.json` (missing)
- `supplementary_materials/comprehensive_rejection_analysis.py` — reads two simulation JSON files (both missing)
- `supplementary_materials/analysis_results/category_breakdown/` — contains PNG figures but no underlying data JSONs
- `supplementary_materials/multi_run_analysis.py` — generates performance_comparison.png from enhanced_aggregated_results.json

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Classification summary:**
- static_verifiable: 31 claims (claims 1–30 all Table 1 values; claim 31 Figure 1)
  - All 30 Table 1 values verified exactly against `enhanced_aggregated_results.json` and confirmed in `enhanced_multi_run_analysis_report.txt`
  - Figure 1 underlying data fully present in `enhanced_aggregated_results.json`
- execution_required: 5 claims (claims 32–36)
  - Figures 2, 3, 4 and results_section claims 35, 36 all require raw simulation JSON files (`results/simuleval/true_gpt_simulation_20250905_001404.json` and `true_gpt_simulation_20250910_123457.json`) that are absent from the repository
  - Code paths exist (category_breakdown_analysis.py, comprehensive_rejection_analysis.py) but need simulation re-run

**Key findings:**
- All 30 Table 1 numerical values match enhanced_aggregated_results.json perfectly (verified to 4+ decimal places)
- Reputation tier data (63%/27%/9.5%/0.5%) has no backing JSON; only PNG exists
- Data Science ~10 bids/job claim: only available data (analysis_results.json from a small 25-round run) shows 5.45, not 10; full LLM-F+LLM-C simulation JSON is missing

**Recommended next step:**
- Execute `run_marketplace.py` for LLM-F+LLM-C configuration (100 rounds) to generate simulation JSON, then run `category_breakdown_analysis.py` and `comprehensive_rejection_analysis.py` to verify claims 32–36

## Session 3 — 2026-04-03
**Purpose:** execution — verify claim 32 (Figure 2: LLM agent rejection patterns comparison)

**Files/context inspected:**
- `supplementary_materials/comprehensive_rejection_analysis.py` — analysis script that needs two simulation files (both missing)
- `supplementary_materials/run_marketplace.py` and `src/marketplace/true_gpt_marketplace.py` — the main simulation entry point
- Required input files `results/simuleval/true_gpt_simulation_20250905_001404.json` and `true_gpt_simulation_20250910_123457.json` — confirmed absent

**Commands run:**
1. `python run_marketplace.py --rounds 3 --freelancers 3 --clients 2 --disable-reflections` → Generated `true_gpt_simulation_20260403_110819.json` — 2 jobs, 0 bids, 6 freelancer rejection decisions (all NO). No client rejection data.
2. `python run_marketplace.py --rounds 8 --freelancers 8 --clients 4 --bids-per-round 3 --jobs-per-freelancer 5 --disable-reflections` → Generated `true_gpt_simulation_20260403_111031.json` — 44 jobs, 1 bid, 103 freelancer rejections, 0 client rejections.

**Theme analysis results (from 103 freelancer NO decisions):**
- rate_budget: 35.7%
- skill_mismatch: 34.6%
- fit: 29.3%
- competition: 0.4%

**Key findings:**
- Freelancer rejections show rate_budget ≈ skill_mismatch (35.7% vs 34.6%) — slightly contradicts the claim that skill > cost
- No client-side rejection data obtainable from test runs (clients never receive bids to reject)
- Original simulation files missing; cannot run large enough simulation to reproduce Figure 2 patterns
- The simulation requires matching job types to freelancer skills; test runs show very few matching jobs

**Claim 32 verdict: Insufficient Evidence**
- Code path exists and is plausible
- Small reproduction confirms general structure of rejection reasoning
- The freelancer patterns show rate_budget slightly higher than skill_mismatch (opposite of claim)
- Client rejection data unavailable from test runs
- Original simulation data files are missing and full reproduction would require extensive API-cost simulation

**Next session should:**
- Proceed to claim 33 (Figure 3) and other execution-required claims (34, 35, 36)

## Session 4 — 2026-04-03
**Purpose:** execution — verify claim 33 (Figure 3: Freelancer reputation tier distribution after 100 rounds)

**Files/context inspected:**
- `supplementary_materials/category_breakdown_analysis.py` — `analyze_reputation_tiers()` reads `reputation_data.freelancers[].tier` from simulation JSON; `analyze_reputation_impact()` computes tiers from `freelancer_profiles` using `total_hired`/`completed_jobs` thresholds
- `supplementary_materials/results/simuleval/true_gpt_simulation_20260403_111031.json` — Session 3 test run (8 freelancers, all New/0 hired), NOT relevant to claim
- `supplementary_materials/analysis_results/category_breakdown/reputation_tier_distribution.png` — pre-existing PNG (not sufficient for Verified)
- `supplementary_materials/analysis_results/category_breakdown/` — no JSON data files, only PNGs
- Required file `results/simuleval/true_gpt_simulation_20250905_001404.json` — confirmed absent

**Tier calculation logic (from analyze_reputation_impact):**
- New: total_hired < 3 (or fails other conditions)
- Established: total_hired in [3,6] with completion_rate >= 60%
- Expert: total_hired in [3,14] with completion_rate >= 75%
- Elite: total_hired >= 15 with completion_rate >= 85%

**Key findings:**
- The PNG `reputation_tier_distribution.png` exists in the repo but underlying raw data does not
- No JSON files in `category_breakdown/` directory
- `analysis_results.json` contains no tier/reputation data
- Existing test runs have 8 freelancers (not 200) with 0 hired
- Full reproduction requires 100-round LLM simulation with 200 freelancers (expensive API calls)

**Claim 33 verdict: Insufficient Evidence**
- Code path is plausible (analyze_reputation_tiers + analyze_reputation_impact functions exist)
- Pre-existing PNG cannot be used as verification
- No underlying data files with tier counts
- Original simulation data file missing
- Full reproduction requires extensive LLM API costs

**Next session should:**
- Proceed to claims 34, 35, 36

## Session 5 — 2026-04-03
**Purpose:** execution — verify claim 34 (Figure 4: Fill rates and competition intensity across skill categories in LLM-F + LLM-C)

**Files/context inspected:**
- `supplementary_materials/analysis_results/analysis_results.json` — small 25-round run (116 jobs, 780 bids, 200 freelancers) with category_competition data
- `supplementary_materials/category_breakdown_analysis.py` — reads `results/simuleval/true_gpt_simulation_20250905_001404.json` (still missing)
- `supplementary_materials/results/simuleval/` — contains only 3 test runs from Session 3 (2026-04-03), not the original simulation

**Key findings:**
- Claim states: "Data Science demonstrates the highest competition intensity with 10 bids per job"
- analysis_results.json shows Data Science at only 5.45 bids/job (not 10), AND Data Science is NOT highest:
  - Admin Support: 15.375 bids/job (highest)
  - Customer Service: 15.33 bids/job
  - Data Science: 5.45 bids/job
  - Engineering & Architecture: 3.0 bids/job
- The existing data is from a 25-round run, not the full 100-round LLM-F+LLM-C simulation
- Required simulation file `true_gpt_simulation_20250905_001404.json` remains missing
- category_breakdown_analysis.py code path is plausible but depends on missing file

**Claim 34 verdict: Insufficient Evidence**
- Code path exists but depends on missing simulation file
- Partial data (25-round run) contradicts claim (5.45 vs 10 bids/job, Data Science not highest), but this is a different/smaller run
- Pre-existing PNG category_performance_metrics.png exists but is insufficient without underlying data
- Full reproduction requires extensive LLM API costs

**Next session should:**
- Proceed to claims 35 and 36

## Session 6 — 2026-04-03
**Purpose:** execution — verify claim 35 (37% of freelancers advance beyond "New")

**Files/context inspected:**
- `supplementary_materials/category_breakdown_analysis.py` — `analyze_reputation_tiers()` reads `reputation_data.freelancers[].tier` from simulation JSON; `analyze_reputation_impact()` computes tiers
- `supplementary_materials/analysis_results/category_breakdown/reputation_tier_distribution.png` — pre-existing PNG (not sufficient for Verified)
- `supplementary_materials/analysis_results/category_breakdown/` — no JSON data files, only PNGs
- Required file `results/simuleval/true_gpt_simulation_20250905_001404.json` — confirmed absent

**Key findings:**
- Claim 35 (37% beyond "New") is arithmetically derived from Figure 3 caption data: 63% New + 27% Established + 9.5% Expert + 0.5% Elite = 37% beyond New
- Same evidentiary situation as claim 33: only PNG exists, no underlying JSON data files
- Simulation file missing; full reproduction requires extensive LLM API costs
- No commands run (reused Session 4/5 investigation)

**Claim 35 verdict: Insufficient Evidence**
- Code path is plausible (analyze_reputation_tiers + analyze_reputation_impact functions exist)
- Pre-existing PNG cannot be used as verification
- No underlying data files with tier counts
- Original simulation data file missing

**Next session should:**
- Proceed to claim 36

## Session 7 — 2026-04-03
**Purpose:** execution — verify claim 36 (Data Science ~10 bids/job, Engineering & Architecture ~3 bids/job)

**Files/context inspected:**
- `supplementary_materials/analysis_results/analysis_results.json` — existing 25-round small run data
- `supplementary_materials/category_breakdown_analysis.py` — requires `results/simuleval/true_gpt_simulation_20250905_001404.json` (missing)
- Session 5 had already analyzed this claim in the context of claim 34

**Key findings (from analysis_results.json, category_competition section):**
- Data Science & Analytics: 5.454545... bids/job (NOT ~10 as claimed)
- Engineering & Architecture: 3.0 bids/job (MATCHES the ~3 claim)
- This data is from a 25-round, 116-job small run, not the full 100-round LLM-F+LLM-C simulation
- The full simulation file `results/simuleval/true_gpt_simulation_20250905_001404.json` is missing
- Full reproduction requires extensive LLM API calls (200 freelancers, 100 rounds)

**No commands run** — reused data from previous sessions (analysis_results.json examined in Session 5)

**Claim 36 verdict: Insufficient Evidence**
- Engineering & Architecture 3.0 bids/job matches the ~3 claim
- Data Science 5.45 bids/job does NOT match the ~10 claim, but this is from a smaller run
- Original simulation file missing; cannot reproduce full LLM-F+LLM-C experiment
- Verdict is Insufficient Evidence because the available data is from a different (smaller) run than the one relevant to the claim; the discrepancy may reflect scale differences, not fabrication
