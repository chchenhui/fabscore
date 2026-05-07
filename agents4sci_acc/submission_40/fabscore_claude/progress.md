# Progress Log

## Session 1 — 2026-04-23
**Purpose:** extraction

**Paper inspected:** `40_Feasibility_Guided_Fair_Ada.pdf` (submission_40)

**Files created/updated:**
- `fabscore_claude/fs_extracted.json` — created with tables, figures, and results_section entries

**Summary of extraction:**
- Extracted 44 table entries from Table 1 (performance comparison across 7 policies), Table 2 (ablation study with 4 variants), and Table 3 (sensitivity to sliding window size)
- Extracted 1 figure (Figure 1: bar chart comparing intervention policies on acute event reduction and fairness disparity)
- Extracted 3 results_section claims not already captured in tables: 21% reduction vs IQL, p < 0.01 for OPE, and ~0.2s inference time

**Next session should:**
- Verify extracted values against the actual PDF if needed
- Check if any figures were missed (the paper may contain more figures beyond Figure 1)
- Review whether the results_section claims accurately reflect the exact wording in the paper body

---

## Session 3 — 2026-04-23
**Purpose:** execution — verify Claim 1 (risk-based prioritization acute event rate 12.3%)

**Files inspected:**
- `fcaf_rl_code/run_experiment.py` — baseline_rate computed as test_df['acute_event'].mean() at line 92
- `fcaf_rl_code/synthetic_data_generation.py` — base_rate = 0.05 + 0.02*comorbidity + 0.01*prior_ed
- `fcaf_rl_code/synthetic_data.csv` — 36,963 rows (not ~130,000 as claimed in reason)

**Execution performed:**
- Computed rates from existing synthetic_data.csv: overall rate = 10.88%, per-state range 10.47-11.09%
- Generated dataset with 5000 patients, seed 42 (suggested entrypoint): overall rate = 10.85%, per-state range 10.36-11.36%
- Tested various seeds (0,1,42,123) and patient counts (2000,5000,10000): all rates in range 10.47-11.36%

**Artifacts created:**
- `fabscore_claude/workspace/claim_1_command_output.txt` — raw command outputs

**Verdict summary:**
- Claimed rate of 12.3% (95% CI: 11.9-12.7) is NOT reproducible from the code
- All tested configurations produce rates of 10.47-11.36%, well below 12.3%
- The data generation formula (base_rate = 0.05 + 0.02*comorbidity + 0.01*prior_ed) cannot produce 12.3% average rates given Poisson(2) comorbidities and Poisson(1) ED visits
- **Verdict: Result Fabrication** — code/data match paper's description but reported numbers don't match execution

**Next session:** No further work needed on Claim 1.

---

## Session 2 — 2026-04-23
**Purpose:** analysis (static code inspection and claim classification)

**Files inspected:**
- `fcaf_rl_code/fcaf_rl.py` — FCAF-RL algorithm implementation (TransitionDataset, DiffusionModel, QNetwork, FCAFConfig, FCAFTrainer, select_action)
- `fcaf_rl_code/run_experiment.py` — Experiment runner with evaluate_policy and main()
- `fcaf_rl_code/synthetic_data_generation.py` — Synthetic data generation (5000 patients, 3 states)
- `fcaf_rl_code/synthetic_data.csv` — Pre-generated synthetic data (~130,000 rows)

**Files created/updated:**
- `fabscore_claude/fs_analysis.json` — Created with full classification of all 48 claims

**Critical findings:**

1. **Broken evaluate_policy (run_experiment.py:49-53)**: The function calls `select_action()` but the returned action index is NEVER used. Reward is always `1 - df_test.iloc[i]['acute_event']` (observed outcome). This means `policy_reward = 1 - baseline_rate` always, and relative reduction = 0%. This makes ALL FCAF-RL performance claims (event rates, reductions, fairness) unreproducible by this code.

2. **Broken adaptive switching (fcaf_rl.py:250)**: `current_disparity = 0.0` is hardcoded and never updates. The `if current_disparity <= fairness_threshold` branch is always taken, so the algorithm always uses the first Q-network (λ=0.0, no fairness). Fairness disparity claims cannot be reproduced.

3. **Action dimension mismatch**: `action_to_idx` maps 10 interventions (0-9, including "None"=9) but `FCAFConfig.action_dim=9`. `F.one_hot(9, num_classes=9)` would raise IndexError for "None" rows (which are common in the data).

4. **No baseline implementations**: IQL, FISOR, OGSRL, CAPS, FairDICE are not implemented anywhere.

5. **No ablation code**: NOAUG, NOFAIR, NOSWITCH variants (Table 2) have no implementation.

6. **No window size code**: Table 3 sensitivity analysis has no window_size parameter.

7. **No NNT, CI, OPE, or timing code**: These measurements are entirely absent.

**Classification summary:**
- `obvious_hallucination` (experiment_fabrication): 13 claims — FCAF-RL full results (7, 13, 26, 30, 34, 38), Table 3 window sensitivity (39-44), results section comparison (46)
- `no_code_files`: 34 claims — all baseline method results, NNT, ablation variants, Figure 1, OPE, inference time
- `execution_required`: 1 claim — Claim 1 (risk-based baseline rate, computable from synthetic data)

**Next session should:**
- Execution phase: run `python run_experiment.py --n_patients 5000 --seed 42` to verify Claim 1 (risk-based baseline rate ~12.3%)
- Confirm the IndexError for "None" intervention during TransitionDataset construction by running the script
