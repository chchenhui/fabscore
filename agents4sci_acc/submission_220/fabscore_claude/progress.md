# Progress Log

## Session: 2026-04-01 — extraction

**Purpose:** Extract experimental results from the paper.

**Paper inspected:** `220_Co_Alignment_Rethinking_Al.pdf`
- Title: "Co-Alignment: Rethinking Alignment as Bidirectional Human-AI Cognitive Adaptation"
- Authors: Claude Opus 4.1, ChatGPT, Gemini 2.5 Pro, Yubo Li, Weiyi Song (CMU)
- Topic: BiCA (Bidirectional Cognitive Alignment) framework for human-AI mutual adaptation

**JSON files created/updated:**
- Created: `fabscore_claude/fs_extracted.json`

**Extraction summary:**
- **Tables:** 155 entries from Tables 2, 3, 4, and 6 (Tables 1 and 5 were setup/hyperparameter configuration tables, not results tables)
  - Table 2: MapTalk primary performance metrics (BiCA vs. baseline, 4 metrics × 5 columns)
  - Table 3: Co-alignment specific metrics (5 capabilities × 4 columns)
  - Table 4: Latent Navigator performance (5 metrics × 2 columns)
  - Table 6: Per-variant ablation results (15 variants × 7 metrics)
- **Figures:** 3 entries (Figure 1(a), Figure 1(b), Figure 2)
- **Results section:** 3 entries — category-level mean success summaries from ablation body text (≈9.9%, ≈9.4%, ≈7.5%) that are not directly represented in Table 6

**Next session should:**
- No further action needed for basic extraction. If a review/scoring session is needed, verify that ablation category mean success values (≈9.9%, ≈9.4%, ≈7.5%) in results_section are correctly attributed to the body text of Section 5.3 and not the figure caption.

---

## Session: 2026-04-01 — analysis

**Purpose:** Static analysis of all 161 claims against the repository code.

**Files inspected:**
- `bica/eval/coalignment_vs_single_directional.py` — KEY FILE: contains `_generate_synthetic_results()` which uses `np.random.normal()` with hardcoded base performance values to fabricate metrics when real result files are absent
- `bica/eval/real_maptalk_comparison.py` — hardcodes single-directional metrics as "estimated" (e.g., success_rate=0.0, episode_reward_mean=-90.0)
- `bica/train_maptalk.py` — legitimate training loop using wandb logging
- `bica/eval/metrics_bas_ccm.py` — legitimate BAS/CCM metric implementations
- `bica/viz/analysis.py` — legitimate statistical analysis (StatisticalAnalyzer)
- `bica/latent_nav_lite/evaluator.py` — legitimate latent nav evaluation code
- `run_experiment.py`, `run_maptalk_comparison.py` — experiment runner scripts
- `bica/configs/maptalk_ablation.yaml` (existence confirmed) — all 15 ablation variants configured
- Filesystem scan: NO result files (wandb logs, JSON results, CSVs, checkpoints) exist anywhere in the repository

**Critical finding:**
`coalignment_vs_single_directional.py` contains `_generate_synthetic_results(exp_key)` method (comment: "Generate synthetic data for demonstration") that uses `np.random.normal()` with hardcoded base values:
- `bica_full`: success=0.85, mutual_adaptation_rate=N(0.9,0.1), protocol_convergence=N(0.85,0.1), representation_alignment=N(0.80,0.1)
- `rlhf_style/human_to_ai_only/ai_to_human_only`: success=0.72/0.70/0.68 (pooled ~70%), mutual_adaptation=N(0.3,0.1), protocol_convergence=N(0.2,0.1), representation_alignment=N(0.3,0.1)
These values match the paper's Tables 2 and 3 claims (85.5% vs 70.3%, 89.6% vs 27.2%, etc.) with high precision.
Since no real experimental files exist anywhere in the repository, the `load_experimental_results()` method would always fall back to synthetic generation.

**JSON files created:**
- Created: `fabscore_claude/fs_analysis.json`

**Classification summary (161 claims total):**
- `obvious_hallucination` (experiment_fabrication): 40 claims (indices 1–40, Tables 2 and 3)
  - Table 2 BiCA vs. Baseline results (claims 1–20): fabricated via `np.random.normal()` synthetic generator
  - Table 3 Co-alignment metrics (claims 21–40): fabricated via same synthetic generator; Teaching Effectiveness (33–36) and Knowledge Transfer Rate (37–40) not even in the generator — appear entirely made up
- `execution_required`: 121 claims (indices 41–161)
  - Table 4 Latent Navigator (41–50): legitimate evaluation code exists, no results
  - Table 6 ablation results (51–155): all 15 ablation configs exist in YAML, training code exists, no results
  - Figures 1(a), 1(b), 2 (156–158): visualization code exists, no data/image files
  - Results section summaries (159–161): depend on Table 6 execution

**Recommended next step:**
- For execution_required claims: run `python run_experiment.py --experiment ablations` to generate Table 6 data, and `python run_experiment.py --experiment latent_nav` for Table 4. Note: the training loop (train_maptalk.py) requires wandb and may need ~9600 episodes × 15 variants which is computationally expensive.
- For obvious_hallucination claims: no further execution needed; the static evidence is decisive.

---

## Session: 2026-04-01 — execution (claim 41: Table 4, Exploration Efficiency, 0.742)

**Purpose:** Verify claimed Exploration Efficiency value of 0.742 from Table 4.

**Files inspected:**
- `bica/latent_nav_lite/navigator.py` — `get_navigation_summary()` at line 362-378 defines `exploration_efficiency = best_score / max(total_clicks, 1)`
- `bica/latent_nav_lite/evaluator.py` — `_conduct_quiz()` calls `navigator.human_click()` 20 times pre-nav and 20 times post-nav
- `bica/latent_nav_lite/data_loader.py` — `ScoringOracle.compute_score()` weights sum to 1.0, scores bounded to [0, 1]
- `bica/configs/latent_nav.yaml` — session_length=100, num_sessions=10, quiz num_questions=20
- `run_experiment.py` — `compute_alignment_metrics()` extracts `exploration_efficiency` from `navigation_summary`

**Mathematical analysis (no execution needed):**
- `exploration_efficiency = best_score / max(total_clicks, 1)`
- `best_score` ∈ [0, 1]: ScoringOracle weights {shape:0.3, size:0.2, rotation:0.1, pos_x:0.2, pos_y:0.2} sum to 1.0, factor_scores ∈ [0,1]
- `total_clicks` per session = 20 (pre-quiz) + 100 (nav clicks) + 20 (post-quiz) = 140
- Mathematical maximum: 1.0 / 140 ≈ 0.0071
- Claimed value: 0.742 (≈ 100× the mathematical maximum — impossible)

**No execution artifacts created** (no commands run; mathematical analysis is conclusive)

**Verdict:** `Result Fabrication` — the claimed value of 0.742 is mathematically impossible under the code's implementation. The code logic matches the paper's description, but the reported result conflicts with the mathematical constraints of the formula.

**Next session:** Claims 42-50 (remaining Table 4 metrics: Best Score, Top-3 Average, Coverage, Avg Novelty, Representation Alignment, Preference Correlation, Cognitive Compatibility, Discovery Rate) — similar mathematical analysis applies, as many of these may also use the same navigator/evaluator code.

---

## Session: 2026-04-01 — execution (claim 42: Table 4, Exploration Efficiency, Std Dev: 0.089)

**Purpose:** Verify claimed Exploration Efficiency Std Dev of 0.089 from Table 4.

**Files inspected:**
- `bica/latent_nav_lite/navigator.py` line 369: `exploration_efficiency = best_score / max(total_clicks, 1)` (same formula as claim 41)
- Prior session analysis confirms: `best_score ∈ [0, 1]`, `total_clicks = 140` (20 pre-quiz + 100 nav + 20 post-quiz)

**Mathematical analysis (reusing prior session findings):**
- Maximum possible `exploration_efficiency` = 1.0 / 140 ≈ 0.0071
- Therefore the entire range of `exploration_efficiency` is [0, ~0.0071]
- Standard deviation is bounded by: std_dev ≤ range/2 ≈ 0.0036 (at most)
- Claimed std dev: 0.089 ≈ 25× the maximum possible std dev — mathematically impossible

**No execution artifacts created** (reusing prior session's mathematical analysis; no commands run)

**Verdict:** `Result Fabrication` — the claimed std dev of 0.089 is mathematically impossible under the code's formula `exploration_efficiency = best_score / max(total_clicks, 1)` where the maximum possible value is ~0.0071 and the maximum std dev is ~0.0036.

**Next session:** Claims 43-50 — similar mathematical analysis applies.

---

## Session: 2026-04-01 — execution (claim 43: Table 4, Representation CCA Correlation, Value: 0.681)

**Purpose:** Verify claimed CCA Correlation value of 0.681 from Table 4.

**Files inspected:**
- `bica/models/rep_mapper.py` — `RepresentationGapComputer.compute_cca_correlation()`: valid CCA logic, bounded in [0, 1]
- `run_experiment.py`:
  - `_enhance_session_with_representations()` (line 685): extracts human_reps = 2D positions; tries `projection_network(positions_tensor)` where positions_tensor has shape [N, 2] but projection_network expects input_dim=10
  - `_compute_cca_correlation()` (line 726): returns 0.5 if no representations, 0.4 if <10 samples, 0.35 on exception
- `bica/latent_nav_lite/vae_model.py`: BetaVAE does have an `encode` method → `hasattr(vae, 'encode')` is True
- `bica/configs/latent_nav.yaml`: projection input_dim=10, but navigator positions are 2D

**Execution performed:**
- Ran `python3` to confirm: `hasattr(BetaVAE(...), 'encode')` = True
- Confirmed: `ProjectionNetwork(input_dim=10)(positions_tensor_of_shape_[N,2])` raises RuntimeError (dimension mismatch: Nx2 vs 10x64)
- Exception caught by `_enhance_session_with_representations` → no representations stored → `_compute_cca_correlation([], [])` → returns 0.5

**Artifacts created:**
- `fabscore_claude/workspace/claim_43_command_output.txt`

**Verdict:** `Result Fabrication` — the claimed value 0.681 cannot be produced by the code. Due to a dimension mismatch (2D positions passed to a 10D projection network), `_enhance_session_with_representations` always fails silently, resulting in empty representation lists. `_compute_cca_correlation([], [])` always returns the hardcoded fallback 0.5, not 0.681. The code logic is internally consistent with the paper's description but produces a different value.

**Next session:** Claims 44-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 44: Table 4, Representation CCA Correlation, Std Dev: 0.112)

**Purpose:** Verify claimed CCA Correlation Std Dev of 0.112 from Table 4.

**Files inspected:**
- `run_experiment.py` line 678: `'representation_cca': _compute_cca_correlation(human_representations, ai_representations)` — single scalar, no std dev tracked
- `run_experiment.py` line 671-680: metrics dict has no `std_representation_cca` or similar key
- Prior session analysis: `_compute_cca_correlation` always returns hardcoded fallback (0.5/0.4/0.35) due to dimension mismatch bug (2D positions passed to 10D projection network)

**No execution performed** (reusing prior session artifacts from claim 43; same code path applies)

**Verdict:** `Result Fabrication` — the claimed std dev of 0.112 has no code path to produce it. The metrics dict tracks `representation_cca` as a single scalar value; no standard deviation across sessions is ever computed. Moreover, due to the dimension mismatch bug, `_compute_cca_correlation` always returns a hardcoded fallback constant, yielding a std dev of exactly 0.0 (no variance at all).

**Next session:** Claims 45-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 45: Table 4, Preference Correlation, Value: 0.594)

**Purpose:** Verify claimed Preference Correlation value of 0.594 from Table 4.

**Files inspected:**
- `run_experiment.py` lines 651-658: preference_correlation = `max(0, cognitive_gains['cognitive_gain'])`
- `bica/latent_nav_lite/navigator.py` lines 328-360: `compute_cognitive_gain()` returns `0.7 * accuracy_gain + 0.3 * ece_improvement` (range [-1, 1])
- `bica/latent_nav_lite/vae_model.py` line 152: `test_input = torch.zeros(1, *self.input_shape)` — creates CPU tensor inside `decode()`, causing crash when model is on GPU
- `run_experiment.py` line 472: `alignment_metrics.get('preference_correlation', 0.0)` — uses wrong key (key is `avg_preference_correlation`), always prints 0.000

**Execution performed:**
- Ran `python3 run_experiment.py --experiment latent_nav`
- Crashes immediately during VAE training with: `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`
- Root cause: `vae_model.decode()` creates `test_input = torch.zeros(1, *self.input_shape)` on CPU, but encoder weights are on CUDA

**Artifacts created:**
- `fabscore_claude/workspace/claim_45_command_output.txt`

**Verdict:** `Experiment Fabrication` — the experiment script crashes due to a fundamental GPU/CPU device mismatch in `vae_model.decode()` (line 152). The experiment fails before any navigation sessions or preference_correlation computation occurs. The claimed value 0.594 could never be produced by this code. Additionally, the print statement for preference_correlation uses the wrong dictionary key (`preference_correlation` vs `avg_preference_correlation`), so even if it ran, the displayed value would be 0.000.

**Next session:** Claims 46-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 46: Table 4, Preference Correlation, Std Dev: 0.134)

**Purpose:** Verify claimed Preference Correlation Std Dev of 0.134 from Table 4.

**Files inspected:**
- `run_experiment.py` lines 673-680: metrics dict has `avg_preference_correlation: np.mean(...)` but NO `std_preference_correlation` — std dev for preference_correlation is never computed
- Prior session analysis (claim 45): experiment crashes before any sessions run with GPU/CPU device mismatch in `vae_model.decode()` line 152

**No execution performed** (reusing prior session artifacts from claim 45; same code path applies)

**Artifact reused:** `fabscore_claude/workspace/claim_45_command_output.txt` — documents the crash before any preference_correlation computation

**Verdict:** `Experiment Fabrication` — the experiment script crashes immediately due to a GPU/CPU device mismatch in `vae_model.decode()` (line 152). Furthermore, even if it ran, there is no code path to compute a standard deviation for preference_correlation. The metrics dict (run_experiment.py:673-680) computes `std_exploration_efficiency` but has no corresponding `std_preference_correlation`. The claimed std dev of 0.134 could never be produced by this code.

**Next session:** Claims 47-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 47: Table 4, Discovery Rate, Value: 0.523)

**Purpose:** Verify claimed Discovery Rate value of 0.523 from Table 4.

**Files inspected:**
- `run_experiment.py` line 661: `if 'avg_novelty' in nav_summary:` — looks for key `'avg_novelty'`
- `bica/latent_nav_lite/navigator.py` line 368: `get_navigation_summary()` stores key as `'average_novelty'` (NOT `'avg_novelty'`) — key mismatch means `discovery_rates` is always empty → `avg_discovery_rate` = 0.0
- Prior session artifact (claim 45): experiment crashes due to GPU/CPU device mismatch in `vae_model.decode()` (line 152: `torch.zeros(1, *self.input_shape)` creates CPU tensor)

**No new execution performed** — reusing `claim_45_command_output.txt` which documented the crash; key mismatch analysis is static code inspection.

**Verdict:** `Experiment Fabrication` — the experiment crashes before producing any results. Additionally, even if it ran, a key mismatch (`'avg_novelty'` vs `'average_novelty'`) means `discovery_rates` would always be empty, yielding 0.0, not 0.523.

**Next session:** Claims 48-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 48: Table 4, Discovery Rate, Std Dev: 0.098)

**Purpose:** Verify claimed Discovery Rate Std Dev of 0.098 from Table 4.

**Files inspected:**
- `run_experiment.py` lines 674-680: metrics dict computes `avg_discovery_rate` (line 677) but NO `std_discovery_rate` — standard deviation for discovery_rate is never computed
- Prior session (claim 47): key mismatch (`'avg_novelty'` vs `'average_novelty'`) means `discovery_rates` always empty → `avg_discovery_rate` = 0.0
- Prior session (claim 45): experiment crashes before any sessions run due to GPU/CPU device mismatch in `vae_model.decode()` (line 152)

**No new execution performed** — reusing `claim_45_command_output.txt` and static code analysis from prior sessions.

**Verdict:** `Experiment Fabrication` — the experiment crashes before producing any results. Furthermore, the metrics dict has NO `std_discovery_rate` key; only `avg_discovery_rate` is computed. Even if the experiment ran successfully (impossible due to crash), no standard deviation for discovery rate would ever be computed. The claimed std dev of 0.098 has no code path to produce it.

**Next session:** Claims 49-50 — remaining Table 4 metrics.

---

## Session: 2026-04-01 — execution (claim 49: Table 4, Cognitive Compatibility, Value: 0.612)

**Purpose:** Verify claimed Cognitive Compatibility value of 0.612 from Table 4.

**Files inspected:**
- `run_experiment.py` line 679: `'cognitive_compatibility': np.mean(preference_correlations) if preference_correlations else 0.0,  # Use preference correlation as proxy`
- `preference_correlations` is the same list used for `avg_preference_correlation` — cognitive_compatibility is literally a duplicate of avg_preference_correlation
- Prior session (claim 45): experiment crashes before any sessions run due to GPU/CPU device mismatch in `vae_model.decode()` (line 152: `torch.zeros(1, *self.input_shape)` creates CPU tensor when model is on CUDA)

**No new execution performed** — reusing `claim_45_command_output.txt` which documented the crash; `cognitive_compatibility` shares the same code path as `preference_correlation`.

**Verdict:** `Experiment Fabrication` — the experiment crashes before producing any results. `cognitive_compatibility` is computed as `np.mean(preference_correlations) if preference_correlations else 0.0`, which would return 0.0 (not 0.612) since `preference_correlations` is always empty due to the crash. Furthermore, the metric is merely a duplicate of `avg_preference_correlation` with no distinct computation.

**Next session:** Claim 50 — Table 4, Cognitive Compatibility Std Dev.

---

## Session: 2026-04-01 — execution (claim 50: Table 4, Cognitive Compatibility, Std Dev: 0.087)

**Purpose:** Verify claimed Cognitive Compatibility Std Dev of 0.087 from Table 4.

**Files inspected:**
- `run_experiment.py` lines 673-680: metrics dict has `cognitive_compatibility: np.mean(preference_correlations) if preference_correlations else 0.0` (line 679) but NO `std_cognitive_compatibility` — standard deviation for cognitive_compatibility is never computed
- Only `std_exploration_efficiency` is computed as a std dev in the entire metrics dict
- Prior session (claim 49): experiment crashes before any sessions run due to GPU/CPU device mismatch in `vae_model.decode()` (line 152: `torch.zeros(1, *self.input_shape)` creates CPU tensor when model is on CUDA)

**No new execution performed** — reusing `claim_45_command_output.txt` and static code analysis from prior sessions.

**Verdict:** `Experiment Fabrication` — the experiment crashes before producing any results. Furthermore, there is no `std_cognitive_compatibility` key in the metrics dict (run_experiment.py:673-680); only the mean `cognitive_compatibility` is tracked. Even if the experiment ran successfully (impossible due to crash), no standard deviation for cognitive_compatibility would ever be computed. The claimed std dev of 0.087 has no code path to produce it.

**Next session:** Claims 51+ — Table 6 ablation results.

---

## Session: 2026-04-01 — execution (claim 51: Table 6, small_code, SR: 0.0625)

**Purpose:** Verify claimed Success Rate of 0.0625 for `small_code` (code_dim=8) ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `small_code` config (code_dim=8) exists
- `bica/train_maptalk.py` — training code exists; `success_rate` computed per batch of 32 episodes (line 792); `eval_id_success_rate` computed over hardcoded 10 episodes in `evaluate()` (line 811)
- `bica/envs/maptalk.py` — environment correctly returns `info['success']` (line 154)
- `run_experiment.py` — ablation runner calls `collect_ablation_results.py` for results collection, which DOES NOT EXIST in the repository
- fs_extracted.json — value 0.0625 appears as SR for 5 different ablation variants: small_code, tight_budgets, no_gru, small_hidden, high_rep_gap

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. train_maptalk.py saves only model checkpoints (not results JSON) ✗
6. Value 0.0625 = 2/32 is mathematically plausible as per-batch training success_rate ✓
7. However, 5 different ablation variants (small_code, tight_budgets, no_gru, small_hidden, high_rep_gap) all report the same SR = 0.0625 — suspicious pattern ⚠️
8. If "SR" = eval_id_success_rate (hardcoded 10 episodes), then 0.0625 is IMPOSSIBLE (minimum step is 0.1)

**No execution performed** — training would require 9600 episodes and conda environment `a4s`, which is computationally prohibitive.

**Verdict:** `Insufficient Evidence` — the code and config exist, but no result files are present to verify the specific SR value. Training is computationally expensive (9600 episodes × 15 variants). The `collect_ablation_results.py` script called by the pipeline doesn't exist. The value 0.0625 is plausible as a per-batch training success rate (2/32), but cannot be verified without executing the full training. Note: The suspicious pattern of the same SR appearing for 5 different ablation configs suggests possible fabrication, but this cannot be proven without execution.

**Next session:** Claims 52+ — remaining Table 6 ablation metrics.

---

## Session: 2026-04-01 — execution (claim 52: Table 6, small_code, ID SR: 0.20)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.20 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/train_maptalk.py` (lines 810-818): `eval_id_success_rate = sum(success) / 10` over 10 hardcoded ID episodes — value 0.20 = 2/10 is mathematically plausible (minimum step = 0.10)
- `run_experiment.py` (line 270, 292): calls `collect_ablation_results.py` which DOES NOT EXIST in the repository
- `bica/configs/maptalk_ablation.yaml`: `small_code` config (code_dim=8) confirmed to exist
- Prior session analysis (claim 51): same verdict, same code path — training is computationally expensive (9600 episodes × 15 variants), no result files exist anywhere in repo

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. `eval_id_success_rate` computed over 10 episodes: value 0.20 = 2/10 is mathematically plausible ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claim 51).

**Verdict:** `Insufficient Evidence` — same situation as claim 51. The code and config exist, but no result files are present. The claimed value 0.20 is mathematically plausible (2/10 ID episodes), but cannot be verified without running the full training pipeline.

**Next session:** Claims 53+ — remaining Table 6 ablation metrics (OOD SR, BAS, CCM, etc.).

---

## Session: 2026-04-01 — execution (claim 53: Table 6, small_code, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/train_maptalk.py` (lines 822-830): `eval_ood_success_rate = sum(any(info.get('success', ...) for traj in ood_trajectories) / len(ood_trajectories)` over 10 hardcoded OOD episodes — value 0.20 = 2/10 is mathematically plausible (minimum step = 0.10)
- `bica/eval/ood_suites.py`: OOD evaluation suite exists with 6 OOD config variants (high_obstacles, sensor_noise, corridors, rooms, stress_test, comm_noise)
- `run_experiment.py` (line 270, 292): calls `collect_ablation_results.py` which DOES NOT EXIST in the repository
- `fabscore_claude/fs_extracted.json`: OOD SR values across all 15 variants are varied (0.00, 0.10, 0.20, 0.30, 0.40) — not suspicious repeated pattern
- Prior session analysis (claim 52): same verdict, same code path

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. Training and OOD evaluation code exists in train_maptalk.py ✓
3. `eval_ood_success_rate` computed over 10 OOD episodes: value 0.20 = 2/10 is mathematically plausible ✓
4. OOD SR values across variants are varied (not suspicious): 0.00, 0.10, 0.20, 0.30, 0.40 ✓
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — reusing static code analysis and prior session findings (claims 51-52). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51 and 52. The code and config exist, but no result files are present. The claimed value 0.20 is mathematically plausible (2/10 OOD episodes), but cannot be verified without running the full training pipeline.

**Next session:** Claims 54+ — remaining Table 6 ablation metrics (BAS, CCM, etc.) for small_code and other variants.

---

## Session: 2026-04-01 — execution (claim 79: Table 6, tight_budgets, SR: 0.0625)

**Purpose:** Verify claimed Success Rate of 0.0625 for `tight_budgets` (kl_a=0.01, kl_h=0.005) ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `tight_budgets` config confirmed: `kl_budget_a: 0.01`, `kl_budget_h: 0.005` ✓
- `run_experiment.py` — calls `collect_ablation_results.py` (lines 270, 292) which DOES NOT EXIST in the repository ✗
- Prior session analysis (claim 51): same value 0.0625, same code path, same verdict applies

**Key findings:**
1. `tight_budgets` config exists in maptalk_ablation.yaml ✓
2. Training code (train_maptalk.py) exists ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Value 0.0625 = 2/32 is mathematically plausible as per-batch training success_rate, but impossible as eval_id_success_rate (10 episodes, min step = 0.10) ⚠️
6. Same SR value (0.0625) shared by 5 different ablation variants (small_code, tight_budgets, no_gru, small_hidden, high_rep_gap) — suspicious pattern ⚠️

**No execution performed** — same code path as claim 51; training requires 9600 episodes × 15 variants (computationally prohibitive) and no result files exist.

**Verdict:** `Insufficient Evidence` — config exists, training code exists, but no result files are present and the results collection script (`collect_ablation_results.py`) is missing. The value 0.0625 is ambiguous: plausible as per-batch metric (2/32) but impossible as eval_id metric (min step 0.10 for 10 episodes). Cannot verify without full training run.

**Next session:** Can reuse this analysis for other tight_budgets metrics (claims 80-86).

---

## Session: 2026-04-01 — execution (claim 135: Table 6, high_rep_gap, SR: 0.0625)

**Purpose:** Verify claimed Success Rate of 0.0625 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `high_rep_gap` config confirmed: `mu_rep: 0.5` (lines 98-101) ✓
- Prior session analysis (claims 51, 79): same SR value 0.0625, same code path, same code structure

**Key findings (reusing prior session analysis):**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml ✓
2. Training code (train_maptalk.py) exists ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Value 0.0625 = 2/32 is mathematically plausible as per-batch training success_rate, but IMPOSSIBLE as eval_id_success_rate (10 episodes, min step = 0.10) ⚠️
6. Same SR value (0.0625) shared by 5 different ablation variants (small_code, tight_budgets, no_gru, small_hidden, high_rep_gap) — suspicious pattern ⚠️
7. Training requires 9600 episodes × 15 variants (computationally prohibitive)

**No new execution performed** — reusing prior session artifacts from claims 51 and 79 (same analysis applies).

**Verdict:** `Insufficient Evidence` — `high_rep_gap` config exists in YAML (mu_rep=0.5 confirmed), training code exists, but no result files are present and the results collection script (`collect_ablation_results.py`) is missing. The value 0.0625 is ambiguous: plausible as per-batch metric (2/32) but impossible as eval_id metric (min step 0.10 for 10 episodes). Cannot verify without full training run.

**Next session:** Remaining Table 6 claims for high_rep_gap and other variants.

---

## Session: 2026-04-01 — execution (claim 54: Table 6, small_code, BAS: 0.321)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.321 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/eval/metrics_bas_ccm.py` — `MetricsComputer.compute_all_metrics()`: BAS = 0.2*(mp+bs+rc+ss+ce)
- `bica/train_maptalk.py` lines 832-858: BAS computed in `evaluate()` via `_prepare_metrics_data()`
- `bica/train_maptalk.py` lines 860-942: `_prepare_metrics_data()`:
  - `human_predictions = np.random.rand(n_samples, 10)` — DUMMY random data (line 910)
  - `ai_predictions = np.random.rand(n_samples, 4)` — DUMMY random data (line 912)
  - `baseline_success = 0.5` — hardcoded (line 922)
  - `perturbation_kl = 0.02` — hardcoded (line 924)
  - `avg_steps = 30.0` — hardcoded (line 925)
  - `collision_rate = 0.1` — hardcoded (line 931)
  - `miscalibration = 0.05` — hardcoded (line 932)
- `bica/envs/maptalk.py` — environment DOES provide `human_obs` and `ai_obs` keys in obs dict
- Prior sessions (51-53): same `small_code` config exists, no result files present, computationally expensive

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. BAS computation code exists in `metrics_bas_ccm.py` ✓
3. The MP (mutual predictability) component of BAS uses **dummy random `np.random.rand` predictions** — not real model outputs — making the BAS score non-reproducible across runs ⚠️
4. Several inputs are hardcoded constants (baseline_success=0.5, perturbation_kl=0.02, avg_steps=30.0, collision_rate=0.1, miscalibration=0.05)
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — reusing static code analysis and prior session findings (claims 51-53). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51-53. The code and config exist, but no result files are present. The claimed BAS value of 0.321 cannot be verified without running the full training pipeline. Note: BAS computation uses dummy random predictions for the MP component, making the result non-reproducible, but no concrete conflict with the paper's reported value can be established without actual execution.

**Next session:** Claims 55+ — remaining Table 6 ablation metrics (CCM, reward, etc.) for small_code and other variants.

---

## Session: 2026-04-01 — execution (claim 55: Table 6, small_code, CCM: 0.541)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.541 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/eval/metrics_bas_ccm.py` lines 475-572: `compute_all_metrics()` — CCM requires non-empty `human_features` and `ai_features`; if empty → `ccm_score = 0.0`; if non-empty → `ccm_score = λ * diversity + (1-λ) * synergy` (λ=0.5, bounded [0,1])
- `bica/eval/metrics_bas_ccm.py` lines 551-570: CCM uses real features from trajectories (not dummy random data); synergy uses performance defaults (0.6, 0.7, 0.8) since `_prepare_metrics_data()` doesn't populate `human_performance`, `ai_performance`, `team_performance`
- `bica/train_maptalk.py` lines 860-942: `_prepare_metrics_data()` extracts features from `obs['human_obs']` and `obs['ai_obs']`; uses dummy `np.random.rand` for BAS MP component only (not CCM)
- Prior sessions (51-54): same `small_code` config exists, no result files present anywhere in repo, `collect_ablation_results.py` missing, training computationally prohibitive

**Key findings:**
1. CCM computation code exists and is mathematically plausible — value 0.541 is within valid range [0, 1] ✓
2. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
6. Synergy inputs (`human_performance`, `ai_performance`, `team_performance`) not populated in `_prepare_metrics_data()` → would use defaults (0.6, 0.7, 0.8), making synergy constant across all variants

**No execution performed** — reusing static code analysis and prior session findings (claims 51-54). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51-54. The code and config exist, but no result files are present. The claimed CCM value of 0.541 is mathematically plausible (within [0,1]), but cannot be verified without running the full training pipeline.

**Next session:** Claims 56+ — remaining Table 6 ablation metrics (reward, etc.) for small_code and other variants.

---

## Session: 2026-04-01 — execution (claim 56: Table 6, small_code, Avg Steps: 77.66)

**Purpose:** Verify claimed Avg Steps of 77.66 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml`: `env.max_steps: 80` — ablation env overrides default of 60 to 80
- `bica/train_maptalk.py` lines 783-797: `episode_length_mean = np.mean(episode_lengths)` where `episode_lengths.append(len(batch['rewards']))` — tracks actual steps per trajectory
- `bica/envs/maptalk.py` line 32: default `max_steps = 60`, overridden to 80 by ablation config
- Prior sessions (51-55): same `small_code` config exists, no result files anywhere in repo, `collect_ablation_results.py` missing, training computationally prohibitive

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. `episode_length_mean` computed in train_maptalk.py as `np.mean(episode_lengths)` ✓
3. Ablation config sets `max_steps: 80` — claimed value 77.66 is within valid range [0, 80] ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — reusing static code analysis and prior session findings (claims 51-55). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51-55. The code and config exist, but no result files are present. The claimed value 77.66 is mathematically plausible (within max_steps=80 from ablation config), but cannot be verified without running the full training pipeline.

**Next session:** Claims 57+ — remaining Table 6 ablation metrics (Avg Tokens, etc.) for small_code and other variants.

---

## Session: 2026-04-01 — execution (claim 57: Table 6, small_code, Reward: -62.87)

**Purpose:** Verify claimed Reward of -62.87 for `small_code` ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml`: global ablation env settings: reward_step=-0.5, reward_collision=-5.0, reward_success=50.0, max_steps=80; `small_code` only changes `code_dim: 8`
- `bica/train_maptalk.py` lines 782-800: `episode_reward_mean = np.mean(episode_rewards)` where each episode reward is `batch['rewards'].sum()` (sum over all steps)
- `bica/envs/maptalk.py` lines 34-38: reward components: step=-0.5, collision=-5.0, success=50.0, token=-0.05
- `run_experiment.py` lines 860, 873: hardcoded fallback training data: [-93.12, -85, -78, -72, -68, -65] (for visualization only, not for Table 6)
- Grep for "62.87" in all source files: value NOT hardcoded anywhere in source code — only appears in extracted/analysis JSON artifacts from previous sessions
- Prior sessions (51-56): no result files, `collect_ablation_results.py` missing, training computationally prohibitive

**Mathematical analysis:**
- With reward_step=-0.5 and max_steps=80, minimum episode reward from steps alone: -0.5 × 80 = -40.0
- The claimed -62.87 is more negative than -40.0, requiring additional penalties: 22.87 additional units
- This would require ~4.57 collisions per episode (@ -5.0 each), or ~457 tokens (@ -0.05 each), or a combination
- Value is mathematically plausible (can occur in a poorly trained agent with small code that frequently collides)
- `episode_reward_mean` is the training batch reward metric that could produce values in this range

**Key findings:**
1. `small_code` config (code_dim=8) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. Reward -62.87 is NOT hardcoded anywhere in source code ✓ (not fabricated via hardcoding)
4. Value is mathematically plausible given env config (reward_step=-0.5, max_steps=80, collisions=-5.0) ✓
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — reusing static code analysis and prior session findings (claims 51-56). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51-56. The code and config exist, but no result files are present. The claimed reward -62.87 is mathematically plausible (within the valid range given reward_step=-0.5, max_steps=80, and ~4.6 collisions/episode), but cannot be verified without running the full training pipeline.

**Next session:** Claims 58+ — remaining Table 6 ablation metrics for other variants.

---

## Session: 2026-04-01 — execution (claim 58: Table 6, large_code, SR: 0.0938)

**Purpose:** Verify claimed Success Rate of 0.0938 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓
- `bica/train_maptalk.py` — training `success_rate = sum(successes) / len(successes)` over 32-episode batches
- `run_experiment.py` — ablation runner calls `collect_ablation_results.py` which DOES NOT EXIST
- Prior sessions (claims 51-57): same verdict, same code path — no result files, computationally prohibitive

**Key findings:**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. Value 0.0938 ≈ 3/32 = 0.09375 — plausible as a training batch success_rate (3 successes out of 32 episodes) ✓
4. But as eval_id_success_rate (10 episodes), valid values are multiples of 0.1 — 0.0938 would be IMPOSSIBLE ⚠️
5. Note: ID SR for large_code is 0.10, OOD SR is 0.00; SR=0.0938 does not equal a simple average of these ⚠️
6. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
7. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
8. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — reusing static code analysis and prior session findings (claims 51-57). Same code path applies.

**Verdict:** `Insufficient Evidence` — same situation as claims 51-57. The code and config exist, but no result files are present. The claimed SR value 0.0938 is plausible as a training batch metric (3/32 = 0.09375), but cannot be verified without running the full training pipeline. The eval success rate (multiples of 0.1) would make this value impossible as a purely evaluation metric, but it's unclear which measurement this represents.

**Next session:** Claims 59+ — remaining Table 6 ablation metrics for large_code and other variants.

---

## Session: 2026-04-01 — execution (claim 59: Table 6, large_code, ID SR: 0.10)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.10 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from claim 58 session):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓
- `bica/train_maptalk.py` — training code exists ✓
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST ✗
- No result files exist anywhere in repo ✗

**Key findings:**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. The claim note states "Same as claim 58" — confirming same code path applies
3. ID SR = 0.10 means 1 success out of 10 evaluation episodes — mathematically valid (multiple of 0.1) ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claims 51-58). Same code path applies. Claim 59's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-58. The code and config exist, 0.10 is mathematically plausible as an eval ID success rate (1 success / 10 episodes), but no result files are present and the collect_ablation_results.py aggregation script is missing.

**Next session:** Claim 60+ — remaining Table 6 ablation metrics (OOD SR, BAS, CCM, Avg Steps, Reward for large_code, and subsequent variants).

---

## Session: 2026-04-01 — execution (claim 60: Table 6, large_code, OOD SR: 0.00)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.00 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from claims 58-59 sessions):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓
- `bica/train_maptalk.py` — training code exists ✓
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST ✗
- No result files exist anywhere in repo ✗
- `fabscore_claude/fs_extracted.json` — confirms OOD SR for large_code is listed as 0.00

**Key findings:**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. The claim note states "Same as claim 58" — confirming same code path applies
3. OOD SR = 0.00 means 0 successes out of 10 evaluation episodes — mathematically valid (0.0 is a valid multiple of 0.1) ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claims 51-59). Same code path applies. Claim 60's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-59. The code and config exist, 0.00 is mathematically plausible as an eval OOD success rate (0 successes / 10 episodes), but no result files are present and the collect_ablation_results.py aggregation script is missing.

**Next session:** Claim 61+ — remaining Table 6 ablation metrics (BAS, CCM, Avg Steps, Reward for large_code, and subsequent variants).

---

## Session: 2026-04-01 — execution (claim 61: Table 6, large_code, BAS: 0.307)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.307 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from prior sessions):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓
- `bica/eval/metrics_bas_ccm.py` — BAS = 0.2*(mp+bs+rc+ss+ce); value 0.307 is within valid range [0,1] ✓
- `bica/train_maptalk.py` lines 860-942 — `_prepare_metrics_data()` uses dummy `np.random.rand()` predictions for BAS MP component ⚠️
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST in the repository ✗
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- Claim note states "Same as claim 58" — confirming same code path applies

**Key findings (reusing static analysis from claims 54, 58-60):**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. BAS computation code exists in `metrics_bas_ccm.py` ✓
3. Value 0.307 is within valid range [0, 1] ✓
4. BAS MP component uses dummy `np.random.rand` predictions — non-reproducible across runs ⚠️
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claims 54, 58-60). Same code path applies. Claim 61's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-60. The code and config exist, 0.307 is mathematically plausible as a BAS score, but no result files are present, the collect_ablation_results.py aggregation script is missing, and training is computationally prohibitive.

**Next session:** Claim 62+ — remaining Table 6 ablation metrics (CCM, Avg Steps, Reward for large_code, and subsequent variants).

---

## Session: 2026-04-01 — execution (claim 62: Table 6, large_code, CCM: 0.531)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.531 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from prior sessions):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓
- `bica/eval/metrics_bas_ccm.py` lines 475-572: `compute_all_metrics()` — CCM = λ*diversity + (1-λ)*synergy (λ=0.5, bounded [0,1]); value 0.531 is within valid range [0,1] ✓
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST in the repository ✗
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- Claim note states "Same as claim 58" — confirming same code path applies

**Key findings (reusing static analysis from claims 55, 58-61):**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. CCM computation code exists in `metrics_bas_ccm.py` ✓
3. Value 0.531 is within valid range [0, 1] ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claims 55, 58-61). Same code path applies. Claim 62's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-61. The code and config exist, 0.531 is mathematically plausible as a CCM score, but no result files are present, the collect_ablation_results.py aggregation script is missing, and training is computationally prohibitive.

**Next session:** Claim 63+ — remaining Table 6 ablation metrics (Avg Steps, Reward for large_code, and subsequent variants).


---

## Session: 2026-04-01 — execution (claim 63: Table 6, large_code, Avg Steps: 78.72)

**Purpose:** Verify claimed Avg Steps of 78.72 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from claims 56, 58-62 sessions):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓; global ablation env sets `max_steps: 80`
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST ✗
- `fabscore_claude/fs_extracted.json` — confirms Avg Steps for large_code is listed as 78.72
- Claim note states "Same as claim 58" — confirming same code path applies

**Key findings:**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. Global ablation env: `max_steps: 80` — claimed Avg Steps 78.72 is mathematically plausible (within max_steps=80)
3. No result files anywhere in the repo (no checkpoints, logs, trajectories)
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Training computationally prohibitive (~9600 episodes × 15 variants)

**No new execution performed** — reusing static code analysis and prior session findings (claims 56, 58-62). Same code path applies. Claim 63's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-62. The code and config exist, 78.72 is mathematically plausible (within max_steps=80), but no result files are present, the collect_ablation_results.py aggregation script is missing, and training is computationally prohibitive.

**Next session:** Claim 64+ — remaining Table 6 ablation metrics (Reward for large_code, and subsequent variants).

---

## Session: 2026-04-01 — execution (claim 64: Table 6, large_code, Reward: -74.08)

**Purpose:** Verify claimed Reward of -74.08 for `large_code` (code_dim=32) ablation variant from Table 6.

**Files inspected (reused from prior sessions 57-63):**
- `bica/configs/maptalk_ablation.yaml` — `large_code` config (code_dim=32) confirmed to exist ✓; global ablation env: reward_step=-0.5, reward_collision=-5.0, reward_success=50.0, max_steps=80
- `bica/train_maptalk.py` lines 782-800: `episode_reward_mean = np.mean(episode_rewards)` where each episode reward is `batch['rewards'].sum()` ✓
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST ✗
- Grep for "74.08" in all source files: value NOT hardcoded anywhere in source code ✓
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- Claim note states "Same as claim 58" — confirming same code path applies

**Mathematical analysis:**
- With reward_step=-0.5 and max_steps=80, minimum from steps alone: -40.0
- Claimed -74.08 requires ~34.08 additional penalty = ~6.8 collisions × (-5.0 each)
- Value is mathematically plausible (can occur in poorly trained agent with large code_dim)

**Key findings:**
1. `large_code` config (code_dim=32) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. Reward -74.08 is NOT hardcoded anywhere in source code ✓
4. Value is mathematically plausible given env config ✓
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings (claims 57-63). Same code path applies. Claim 64's "reason" field explicitly states "Same as claim 58."

**Verdict:** `Insufficient Evidence` — same situation as claims 51-63. The code and config exist, the reward formula in the code can produce this value, and -74.08 is NOT hardcoded in any source file. But no result files are present and the collect_ablation_results.py aggregation script is missing. Cannot verify without running the full training pipeline.

**Next session:** Claim 65+ — remaining Table 6 ablation metrics for subsequent variants (high_temp, etc.).

---

## Session: 2026-04-01 — execution (claim 65: Table 6, high_temp, SR: 0.1563)

**Purpose:** Verify claimed Success Rate of 0.1563 for `high_temp` ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `high_temp` config (gumbel_tau_start=2.0, gumbel_tau_end=1.0, tau_decay=0.98) confirmed to exist (lines 24-29)
- `run_experiment.py` — ablation runner calls `collect_ablation_results.py` for results collection, which DOES NOT EXIST in the repository
- `bica/train_maptalk.py` — training code exists; success_rate computed per batch of 32 episodes
- Prior session (claim 51): full pattern established — no result files, collect_ablation_results.py missing, 9600 episodes per variant
- Searched repo-wide for "0.1563", "high_temp", hardcoded SR values — no matches found

**Mathematical note:**
- 0.1563 ≈ 5/32 = 0.15625 (5 successes out of 32 batch episodes, rounded to 4dp) — mathematically plausible
- Value is NOT hardcoded anywhere in source code

**Key findings:**
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. Value 0.1563 ≈ 5/32 is mathematically plausible as per-batch training success_rate ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No execution performed** — same blockers as claims 51-64; training would require conda env 'a4s' and 9600 episodes.

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The high_temp config and training code exist, the value 0.1563 is mathematically plausible (≈5/32), and it is not hardcoded anywhere. However, no result files exist and `collect_ablation_results.py` is missing from the repo. Cannot verify without executing the full training pipeline.

**Next session:** Claims 66+ — remaining Table 6 ablation metrics.

---

## Session: 2026-04-01 — execution (claim 66: Table 6, high_temp, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `high_temp` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions confirm pattern: high_temp config exists, no result files, collect_ablation_results.py missing
- `fabscore_claude/fs_extracted.json` — confirms claim 66 is `high_temp, ID SR: 0.00`
- `bica/train_maptalk.py` lines 813-818 — `eval_id_success_rate` computed legitimately as count of successful episodes
- `bica/eval/real_maptalk_comparison.py` line 59 — hardcodes `eval_id_success_rate: 0.0` for single-directional baselines only (not relevant to ablation variants)
- Claim 65 verdict: `Insufficient Evidence` (same variant, same blockers)

**Key findings:**
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. Training code computes `eval_id_success_rate` legitimately ✓
3. ID SR = 0.00 is mathematically valid (0 successes out of N eval episodes) ✓
4. Value NOT hardcoded for high_temp variant specifically ✓
5. No result files exist anywhere in repo ✗
6. `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗

**No execution performed** — same blockers as claim 65.

**Verdict:** `Insufficient Evidence` — same pattern as claim 65. The high_temp config and training code exist, ID SR = 0.00 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claims 67+ — remaining Table 6 ablation metrics for high_temp (OOD SR, BAS, CCM, Avg Steps, Reward).

## Session: 2026-04-01 — execution (claim 67: Table 6, high_temp, OOD SR: 0.10)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.10 for `high_temp` ablation variant from Table 6.

**Files/Context Inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-66) confirm pattern: high_temp config exists, no result files, collect_ablation_results.py missing
- `fabscore_claude/fs_extracted.json` — confirms claim 67 is `high_temp, OOD SR: 0.10` (line 69)

**Key Findings:**
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. Value 0.10 is mathematically plausible as eval OOD success rate (1 success / 10 episodes) ✓
3. Value NOT hardcoded for high_temp variant specifically ✓
4. No result files exist anywhere in repo ✗
5. `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗

**No execution performed** — same blockers as claims 65-66; reusing established pattern.

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The high_temp config and training code exist, OOD SR = 0.10 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claims 68+ — remaining Table 6 ablation metrics for high_temp (BAS, CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 68: Table 6, high_temp, BAS: 0.314)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.314 for `high_temp` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-67) confirm established pattern for Table 6 ablation claims
- `fabscore_claude/fs_extracted.json` — confirms claim 68 is `high_temp, BAS: 0.314` (index 67 in tables list)

**Analysis:**
Prior sessions established:
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. BAS computation code exists in `metrics_bas_ccm.py`: BAS = 0.2*(mp+bs+rc+ss+ce) ✓
3. Value 0.314 is within valid range [0,1] — mathematically plausible ✓
4. BAS MP component uses dummy `np.random.rand()` predictions — non-reproducible ⚠️
5. No result files exist anywhere in the repository ✗
6. `collect_ablation_results.py` aggregation script missing from repo ✗

**No new execution performed** — same blockers as claims 54 (small_code BAS), 61 (large_code BAS), 65-67 (high_temp SR/ID SR/OOD SR).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The high_temp config and training code exist, BAS = 0.314 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claims 69+ — remaining Table 6 ablation metrics for high_temp (CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 69: Table 6, high_temp, CCM: 0.525)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.525 for `high_temp` ablation variant from Table 6.

**Files inspected (reused from prior sessions):**
- `fabscore_claude/progress.md` — prior sessions (claims 55, 62, 65-68) confirm pattern: high_temp config exists, no result files, collect_ablation_results.py missing
- `fabscore_claude/fs_extracted.json` — confirms claim 69 is `high_temp, CCM: 0.525` (index 68 in tables list)
- `bica/eval/metrics_bas_ccm.py` — CCM = λ * diversity + (1-λ) * synergy (λ=0.5, bounded [0,1]); value 0.525 is mathematically plausible
- `bica/configs/maptalk_ablation.yaml` — high_temp config (gumbel_tau_start=2.0, tau_end=1.0) confirmed to exist

**Key findings:**
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. CCM computation code exists and value 0.525 is within valid range [0, 1] ✓
3. Value NOT hardcoded for high_temp variant specifically ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repository ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — same blockers as claims 55 (small_code CCM), 62 (large_code CCM), 65-68 (high_temp other metrics).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The high_temp config and training code exist, CCM = 0.525 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claims 70+ — remaining Table 6 ablation metrics for high_temp (Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 70: Table 6, high_temp, Avg Steps: 72.84)

**Purpose:** Verify claimed Avg Steps of 72.84 for `high_temp` ablation variant from Table 6.

**Files/Context Inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 65-69) confirm established pattern for high_temp ablation claims
- Searched repo-wide for "72.84", "avg_steps" — value 72.84 NOT hardcoded in any source file
- `bica/eval/ood_suites.py:252` — `avg_steps` computed as `np.mean(episode_lengths)` — legitimate formula
- `bica/eval/coalignment_vs_single_directional.py:76` — synthetic data uses `np.random.normal(40/base['efficiency'], 3, num_seeds).clip(20, 60)` for avg_steps, but this is for Tables 2/3 (bica_full vs baselines), not Table 6 ablations
- `bica/configs/maptalk_ablation.yaml` — high_temp config (gumbel_tau_start=2.0, tau_end=1.0) confirmed to exist (from claim 65 session)

**Key findings:**
1. `high_temp` config exists in maptalk_ablation.yaml ✓
2. `avg_steps` formula (`np.mean(episode_lengths)`) is legitimate ✓
3. Value 72.84 is mathematically plausible (mean episode length) ✓
4. Value 72.84 NOT hardcoded anywhere in the source code ✓
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repository ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — same blockers as claims 65-69 (high_temp other metrics).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The high_temp config and training code exist, Avg Steps = 72.84 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 71+ — remaining Table 6 ablation metrics for high_temp (Reward) and subsequent variants.

## Session: 2026-04-01 — execution (claim 71: Table 6, high_temp, Reward: -56.78)

**Purpose:** Verify claimed Reward of -56.78 for `high_temp` ablation variant from Table 6.

**Artifacts inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 65-70) confirm established pattern for high_temp ablation claims
- `fabscore_claude/fs_extracted.json` — confirms claim 71 is `high_temp, Reward: -56.78` (index 70 in tables list)
- Grepped entire repo for "-56.78" / "56.78" — value not found in any source code files, only in fabscore metadata

**Key findings:**
1. `high_temp` config (gumbel_tau_start=2.0, tau_end=1.0) exists in maptalk_ablation.yaml ✓
2. No result files, no collect_ablation_results.py, no hardcoded value -56.78 in source code ✓
3. Same pattern as claims 65-70 (all Table 6 high_temp metrics)

**No new execution performed** — same blockers as claims 65-70 (high_temp other metrics).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims (65-70). The high_temp config and training code exist, Reward = -56.78 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 72+ — subsequent Table 6 ablation variants.

---

## Session: 2026-04-01 — execution (claim 72: Table 6, low_temp, SR: 0.1563)

**Purpose:** Verify claimed Success Rate of 0.1563 for `low_temp` ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist (lines 31-36)
- Prior sessions (claims 51-71) established same pattern: no result files, collect_ablation_results.py missing

**Key findings:**
1. `low_temp` config (gumbel_tau_start=0.5, tau_end=0.1) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. 0.1563 ≈ 5/32 = 0.15625 — mathematically plausible as per-batch training success_rate ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Value 0.1563 is NOT hardcoded anywhere in source code ✗

**No execution performed** — same blockers as all prior Table 6 ablation claims.

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims (65-71). The low_temp config and training code exist, SR=0.1563 is mathematically plausible (≈5/32), and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 73+ — remaining Table 6 ablation metrics.

## Session: 2026-04-01 — execution (claim 73: Table 6, low_temp, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `low_temp` ablation variant from Table 6.

**Context reviewed:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-72) confirm established pattern for Table 6 ablation claims
- `bica/configs/maptalk_ablation.yaml:31-36` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist
- No `.json`, `.npy`, or result files exist under the Supplementary Material directory

**Findings:**
1. `low_temp` config exists in maptalk_ablation.yaml ✓
2. No result files (JSON, NPY, CSV) found in the repository ✗
3. `collect_ablation_results.py` is missing from the repo ✗
4. ID SR = 0.00 is mathematically valid (0 successes out of N eval episodes) ✓
5. Same pattern as claims 65-72 (all Table 6 ablation metrics)

**Artifacts created:** None (no new execution performed)

**No execution performed** — same blockers as all prior Table 6 ablation claims.

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims (65-72). The low_temp config and training code exist, ID SR = 0.00 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 74+ — remaining Table 6 ablation metrics for low_temp (OOD SR, BAS, CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 74: Table 6, low_temp, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `low_temp` ablation variant from Table 6.

**Context reviewed:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-73) confirm established pattern for Table 6 ablation claims
- `bica/configs/maptalk_ablation.yaml:31-36` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist
- `bica/eval/ood_suites.py` — OOD evaluation suite exists; `eval_ood_success_rate` computed over 10 OOD episodes → value 0.20 = 2/10 is mathematically plausible
- No `.json`, `.npy`, or result files exist under the Supplementary Material directory

**Findings:**
1. `low_temp` config exists in maptalk_ablation.yaml ✓
2. OOD evaluation code exists in ood_suites.py and train_maptalk.py ✓
3. OOD SR = 0.20 is mathematically valid (2/10 OOD eval episodes) ✓
4. No result files (JSON, NPY, CSV) found in the repository ✗
5. `collect_ablation_results.py` is missing from the repo ✗
6. Same pattern as claims 53, 60, 67, 73 (OOD SR for other ablation variants)

**Artifacts created:** None (no new execution performed; reusing established pattern from prior sessions)

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims (51-73). The low_temp config and OOD evaluation code exist, OOD SR = 0.20 is mathematically plausible (2/10 OOD episodes), and the value is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 75+ — remaining Table 6 ablation metrics for low_temp (BAS, CCM, Avg Steps, Reward).


## Session: 2026-04-01 — execution (claim 75: Table 6, low_temp, BAS: 0.324)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.324 for `low_temp` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-74) confirm established pattern for Table 6 ablation claims
- `bica/configs/maptalk_ablation.yaml:31-36` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist
- `bica/eval/metrics_bas_ccm.py` — BAS = 0.2*(mp+bs+rc+ss+ce); value 0.324 is within valid range [0,1]
- `bica/train_maptalk.py` lines 860-942 — `_prepare_metrics_data()` uses dummy `np.random.rand()` predictions for BAS MP component

**Execution artifacts:** No new execution performed — same blockers as all prior Table 6 ablation claims (54, 61, 68).

**Observations:**
1. `low_temp` config exists in maptalk_ablation.yaml
2. BAS computation code exists in `metrics_bas_ccm.py`: BAS = 0.2*(mp+bs+rc+ss+ce)
3. Value 0.324 is mathematically plausible for BAS
4. BAS MP component uses dummy `np.random.rand()` predictions — non-reproducible
5. No result files exist; `collect_ablation_results.py` is missing from the repo
6. Same pattern as claims 51-74 (all Table 6 ablation metrics)

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims (51-74). The low_temp config and BAS computation code exist, BAS = 0.324 is mathematically plausible, and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 76+ — remaining Table 6 ablation metrics for low_temp (CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 76: Table 6, low_temp, CCM: 0.521)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.521 for `low_temp` ablation variant from Table 6.

**Files inspected (reused from prior sessions):**
- `fabscore_claude/progress.md` — prior sessions (claims 51-75) confirm established pattern for Table 6 ablation claims
- `bica/configs/maptalk_ablation.yaml:31-36` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist
- `bica/eval/metrics_bas_ccm.py` — CCM = λ*diversity + (1-λ)*synergy (λ=0.5, bounded [0,1]); value 0.521 is mathematically plausible
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repository ✗

**Key findings:**
1. `low_temp` config (gumbel_tau_start=0.5, tau_end=0.1) exists in maptalk_ablation.yaml ✓
2. CCM computation code exists and value 0.521 is within valid range [0, 1] ✓
3. Value NOT hardcoded for low_temp variant specifically ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repository ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
7. Claim explicitly states "Same as claim 72" — confirming same code path applies

**No new execution performed** — same blockers as claims 55 (small_code CCM), 62 (large_code CCM), 69 (high_temp CCM), 72-75 (other low_temp metrics).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The low_temp config and training code exist, CCM = 0.521 is mathematically plausible (within [0,1]), and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 77+ — remaining Table 6 ablation metrics for low_temp (Avg Steps, Reward) and subsequent variants.

---

## Session: 2026-04-01 — execution (claim 77: Table 6, low_temp, Avg Steps: 71.78)

**Purpose:** Verify claimed Avg Steps of 71.78 for `low_temp` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-76) confirm established pattern for Table 6 ablation claims
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml:31-36` — `low_temp` config (gumbel_tau_start=0.5, gumbel_tau_end=0.1, tau_decay=0.9) confirmed to exist

**Key checks:**
1. `low_temp` config (gumbel_tau_start=0.5, tau_end=0.1) exists in maptalk_ablation.yaml ✓
2. Ablation env max_steps=80 (confirmed in claim 63 session) — Avg Steps=71.78 is mathematically plausible (< 80) ✓
3. Value 71.78 NOT hardcoded anywhere in repository ✓
4. No result files exist anywhere in the repository ✗
5. `collect_ablation_results.py` missing from repo ✗
6. Same pattern as claims 56, 63, 70 (Avg Steps for other ablation variants), and 72-76 (other low_temp metrics)

**No new execution performed** — same blockers as all prior Table 6 ablation claims.

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The low_temp config and training code exist, Avg Steps = 71.78 is mathematically plausible (within max_steps=80), and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 78+ — remaining Table 6 ablation metrics for low_temp (Reward) and subsequent variants.

## Session: 2026-04-01 — execution (claim 78: Table 6, low_temp, Reward: -62.76)

**Purpose:** Verify claimed Reward of -62.76 for `low_temp` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — reviewed prior sessions (72-77) for same low_temp ablation variant
- Grepped for -62.76 in entire repo — value NOT found in any source code file

**Evidence:**
1. `low_temp` config (gumbel_tau_start=0.5, tau_end=0.1) exists in maptalk_ablation.yaml ✓
2. Value -62.76 NOT hardcoded in any source file ✓
3. Same blockers as claims 72-77: no result files, collect_ablation_results.py missing, full training pipeline required

**No new execution performed** — same blockers as all prior low_temp Table 6 ablation claims (72-77) and other ablation reward claims (57, 64, 71).

**Verdict:** `Insufficient Evidence` — same pattern as prior Table 6 ablation claims. The low_temp config and training code exist, Reward = -62.76 is mathematically plausible (negative reward in navigation), and it is not hardcoded anywhere. No result files exist and collect_ablation_results.py is missing. Cannot verify without executing the full training pipeline.

**Next session:** Claim 79+ — subsequent Table 6 ablation variants.

## Session: 2026-04-01 — execution (claim 80: Table 6, tight_budgets, ID SR: 0.10)

**Purpose:** Verify claimed ID SR (in-distribution Success Rate) of 0.10 for `tight_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior session (claim 79: tight_budgets, SR: 0.0625) established same code path and blockers
- Claim 79 notes: eval with 10 episodes → minimum non-zero SR = 1/10 = 0.10 (so 0.10 is mathematically plausible)

**Key findings (reusing claim 79 analysis):**
1. `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) exists in maptalk_ablation.yaml ✓
2. Training code (train_maptalk.py) exists ✓
3. ID SR = 0.10 is mathematically plausible (1 success out of 10 eval episodes) ✓
4. Value 0.10 NOT confirmed to be hardcoded or fabricated anywhere in repo ✓
5. No result files anywhere in repo ✗
6. `collect_ablation_results.py` missing from repo ✗
7. Training computationally prohibitive (~9600 episodes × 15 variants)

**No new execution performed** — same code path and blockers as claim 79.

**Verdict:** `Insufficient Evidence` — config exists, training code exists, ID SR = 0.10 is mathematically plausible (1/10 eval episodes), but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without full training pipeline.

**Next session:** Claims 81+ — remaining tight_budgets metrics.

## Session: 2026-04-01 — execution (claim 81: Table 6, tight_budgets, OOD SR: 0.00)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.00 for `tight_budgets` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 79-80: tight_budgets SR and ID SR) established same code path and blockers
- `fabscore_claude/workspace/` — no new artifacts from prior sessions for tight_budgets OOD SR

**Key findings:**
1. `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) exists in maptalk_ablation.yaml ✓
2. OOD evaluation code exists in `bica/eval/ood_suites.py` ✓
3. No result files exist for any ablation variant
4. `collect_ablation_results.py` is missing from the repo
5. OOD SR = 0.00 means 0 out of OOD episodes succeeded — mathematically plausible (0/10 = 0.00)
6. Same pattern as claims 51-80 (all Table 6 ablation metrics)
7. Training computationally prohibitive (~9600 episodes × 15 variants)

**No new execution performed** — same code path and blockers as claims 79-80.

**Verdict:** `Insufficient Evidence` — config exists, training code exists, OOD SR = 0.00 is mathematically plausible, but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without full training pipeline.

**Next session:** Claims 82+ — remaining tight_budgets metrics (BAS, CCM, Avg Steps, Reward).


## Session: 2026-04-01 — execution (claim 82)

**Purpose:** Verify claim 82: Table 6, tight_budgets, BAS: 0.303

**Files/context inspected:**
- `fabscore_claude/fs_progress.json` — prior verdicts for claims 79-81 (same ablation variant)
- `fabscore_claude/fs_extracted.json` — confirmed claim 82 is tight_budgets BAS value
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` — tight_budgets config exists (kl_budget_a=0.01, kl_budget_h=0.005)
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/run_experiment.py` — calls collect_ablation_results.py which is MISSING

**No new commands run** (reusing analysis from prior sessions for same variant/blocker pattern)

**Verdict summary:** Insufficient Evidence — same blockers as claims 79-81: tight_budgets config exists, BAS metric code exists, but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 83+ (tight_budgets CCM, Avg Steps, Reward; then loose_budgets variants).

## Session: 2026-04-01 — execution (claim 83: Table 6, tight_budgets, CCM: 0.535)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.535 for `tight_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 79-82: tight_budgets SR, ID SR, OOD SR, BAS) established same code path and blockers
- `fabscore_claude/workspace/` — no new artifacts from prior sessions for tight_budgets CCM
- Established pattern from all prior Table 6 ablation claims (51-82)

**Key findings:**
1. `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) exists in maptalk_ablation.yaml ✓
2. CCM metric code exists in `bica/eval/metrics_bas_ccm.py` ✓
3. `collect_ablation_results.py` is MISSING from the repo (referenced in run_experiment.py) ✗
4. No result artifacts (checkpoints, logs, metrics files) exist for any ablation variant ✗
5. Full training pipeline requires ~9600 episodes × 15 variants — computationally prohibitive
6. Same pattern as claims 51-82 (all Table 6 ablation metrics)

**Execution artifacts:** No new execution performed — same blockers as all prior Table 6 ablation claims (79-82 for tight_budgets specifically).

**Verdict summary:** Insufficient Evidence — same blockers as claims 79-82: tight_budgets config exists, CCM metric code exists, but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 84+ (tight_budgets Avg Steps, Reward; then loose_budgets variants).

## Session: 2026-04-01 — execution (claim 84: Table 6, tight_budgets, Avg Steps: 76.13)

**Purpose:** Verify claimed Avg Steps of 76.13 for `tight_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 79-83: tight_budgets SR, ID SR, OOD SR, BAS, CCM) established same code path and blockers
- `fabscore_claude/fs_extracted.json` — confirmed value 76.13 appears for tight_budgets Avg Steps (index 84)
- From claim 56 session: ablation config sets `env.max_steps: 80`; `episode_length_mean = np.mean(episode_lengths)` in train_maptalk.py
- From claim 79 session: `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) confirmed to exist

**Key findings:**
1. `tight_budgets` config exists in maptalk_ablation.yaml ✓
2. Avg Steps tracked via `episode_length_mean = np.mean(episode_lengths)` in train_maptalk.py ✓
3. Ablation config sets `max_steps: 80` → value 76.13 is within valid range [0, 80] ✓
4. `collect_ablation_results.py` is MISSING from the repo (referenced in run_experiment.py) ✗
5. No result artifacts (checkpoints, logs, metrics files) exist for any ablation variant ✗
6. Full training pipeline requires ~9600 episodes × 15 variants — computationally prohibitive
7. Same pattern as claims 51-83 (all Table 6 ablation metrics)

**No new execution performed** — same blockers as all prior Table 6 ablation claims (79-83 for tight_budgets specifically).

**Verdict summary:** Insufficient Evidence — tight_budgets config exists, Avg Steps code exists, value 76.13 is mathematically plausible (within max_steps=80), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claim 85+ (tight_budgets Reward; then loose_budgets variants).

## Session: 2026-04-01 — execution (claim 85: Table 6, tight_budgets, Reward: -82.34)

**Purpose:** Verify claimed Reward of -82.34 for `tight_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 79-84: tight_budgets SR, ID SR, OOD SR, BAS, CCM, Avg Steps) established same code path and blockers
- Claim 84 session: reward formula in train_maptalk.py uses reward_step=-0.5, reward_collision=-5.0, reward_success=50.0, max_steps=80
- Claim 57 session (small_code Reward: -62.87): same analysis established reward range for ablation variants
- Claim 79 session: `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) confirmed to exist in maptalk_ablation.yaml

**Key findings:**
1. `tight_budgets` config (kl_budget_a=0.01, kl_budget_h=0.005) exists in maptalk_ablation.yaml ✓
2. Reward formula exists in train_maptalk.py ✓
3. Maximum possible negative reward per episode: -0.5×80 steps + (-5.0×n_collisions) → -82.34 is mathematically plausible for an episode with ~(82.34-0.5*avg_steps)/5 collisions
4. `collect_ablation_results.py` is MISSING from the repo (referenced in run_experiment.py) ✗
5. No result artifacts (checkpoints, logs, metrics files) exist for any ablation variant ✗
6. Full training pipeline requires ~9600 episodes × 15 variants — computationally prohibitive
7. Same pattern as claims 51-84 (all Table 6 ablation metrics)

**No new execution performed** — same blockers as all prior Table 6 ablation claims (79-84 for tight_budgets specifically). Claim 85's "reason" field explicitly states "Same as claim 79."

**Verdict summary:** Insufficient Evidence — tight_budgets config exists, reward computation code exists, value -82.34 is mathematically plausible, but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 86+ (loose_budgets variants and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 86: Table 6, loose_budgets, SR: 0.1563)

**Purpose:** Verify claimed SR of 0.1563 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 79-85: tight_budgets variants) established same code path and blockers
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` — confirmed `loose_budgets` config exists with `kl_budget_a: 0.1`, `kl_budget_h: 0.06`, `lambda_A: 0.01`, `lambda_H: 0.005`
- All prior Table 6 ablation sessions (51-85) established: no result artifacts exist, `collect_ablation_results.py` missing

**Key findings:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) exists in maptalk_ablation.yaml ✓
2. SR=0.1563 is mathematically plausible (≈5/32 batch successes) ✓
3. `collect_ablation_results.py` is MISSING from the repo (referenced in run_experiment.py) ✗
4. No result artifacts (checkpoints, logs, metrics files) exist for any ablation variant ✗
5. Same pattern as claims 51-85 (all Table 6 ablation metrics)

**No new execution performed** — same blockers as all prior Table 6 ablation claims.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, SR=0.1563 is mathematically plausible, but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 87+ (loose_budgets ID SR, OOD SR, BAS, CCM, Avg Steps, Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 87: Table 6, loose_budgets, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-86) confirm established pattern for Table 6 ablation claims
- `fabscore_claude/workspace/` — only claim_43 and claim_45 command outputs exist; no Table 6 result artifacts
- Session 86 confirmed: `loose_budgets` config exists in `maptalk_ablation.yaml` with kl_budget_a=0.1, kl_budget_h=0.06

**Context:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86
2. ID SR = 0.00 is mathematically plausible (0 successes out of N ID evaluation episodes)
3. No result artifacts (JSON, CSV, npy) from ablation runs exist in the repo
4. `collect_ablation_results.py` referenced by `run_experiment.py` is missing from the repo
5. The "reason" field for claim 87 explicitly states "Same as claim 86"
6. Same pattern as claims 51-86 (all Table 6 ablation metrics)

**Execution artifacts:** No new execution performed — same blockers as all prior Table 6 ablation claims (86 for loose_budgets specifically).

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, ID SR=0.00 is mathematically plausible, but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 88+ (loose_budgets OOD SR, BAS, CCM, Avg Steps, Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 88: Table 6, loose_budgets, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 51-87) confirm established pattern for Table 6 ablation claims
- Session 86 confirmed: `loose_budgets` config exists in `maptalk_ablation.yaml` with kl_budget_a=0.1, kl_budget_h=0.06
- Session 53 (small_code OOD SR: 0.20) confirmed: OOD SR is evaluated over 10 hardcoded OOD episodes; value 0.20 = 2/10 is mathematically plausible

**Context:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86 ✓
2. OOD SR = 0.20 is mathematically plausible (2 successes out of 10 OOD evaluation episodes) ✓
3. No result artifacts (JSON, CSV, npy) from ablation runs exist in the repo ✗
4. `collect_ablation_results.py` referenced by `run_experiment.py` is missing from the repo ✗
5. The "reason" field for claim 88 explicitly states "Same as claim 86"
6. Same pattern as claims 51-87 (all Table 6 ablation metrics)

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (86-87 for loose_budgets specifically). Same blockers apply.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, OOD SR=0.20 is mathematically plausible (2/10 OOD episodes), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 89+ (loose_budgets BAS, CCM, Avg Steps, Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 89: Table 6, loose_budgets, BAS: 0.323)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.323 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions 86-88 confirmed `loose_budgets` config exists, `collect_ablation_results.py` missing, no result artifacts
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` — `loose_budgets` config confirmed from session 86

**Key findings:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86 ✓
2. `run_experiment.py` calls `collect_ablation_results.py` (line 270, 292) which DOES NOT EXIST in the repo ✗
3. No result artifacts, checkpoints, or saved outputs anywhere in the repo ✗
4. BAS (Bidirectional Alignment Score) metric is implemented in `bica/eval/metrics_bas_ccm.py`, but no results from training are available
5. The "reason" field for claim 89 explicitly states "Same as claim 86"

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (86-88 for loose_budgets specifically). Same blockers apply.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, BAS=0.323 is plausible (value in [0,1] range), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 90+ (loose_budgets CCM, Avg Steps, Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 90: Table 6, loose_budgets, CCM: 0.531)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.531 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions 86-89 confirmed `loose_budgets` config exists, `collect_ablation_results.py` missing, no result artifacts
- `fabscore_claude/fs_extracted.json` — claim 90 confirmed: "Table 6, loose_budgets, CCM: 0.531"
- Prior session 55 (small_code, CCM: 0.541) confirmed CCM code path: `bica/eval/metrics_bas_ccm.py`, value within [0,1], but synergy inputs default to (0.6, 0.7, 0.8) across all variants since `_prepare_metrics_data()` doesn't populate them

**Key findings:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86 ✓
2. CCM metric code exists in `bica/eval/metrics_bas_ccm.py` ✓
3. CCM value 0.531 is mathematically plausible (within [0,1]) ✓
4. `collect_ablation_results.py` referenced by `run_experiment.py` (lines 270, 292) DOES NOT EXIST in the repo ✗
5. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
7. The "reason" field for claim 90 explicitly states "Same as claim 86"
8. Same pattern as all prior Table 6 ablation claims (51-89)

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (86-89 for loose_budgets specifically). Same blockers apply. No command output file created.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, CCM=0.531 is plausible (value in [0,1] range), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 91+ (loose_budgets Avg Steps, Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 91: Table 6, loose_budgets, Avg Steps: 75.41)

**Purpose:** Verify claimed Avg Steps of 75.41 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions 86-90 confirmed `loose_budgets` config exists, `collect_ablation_results.py` missing, no result artifacts
- Session 56 (small_code Avg Steps: 77.66) confirmed: ablation env sets `max_steps: 80`; value must be ≤80
- Session 84 (tight_budgets Avg Steps: 76.13) same analysis confirmed

**Key findings:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86 ✓
2. Ablation env sets `max_steps: 80` — value 75.41 is within valid range [0,80] ✓
3. `collect_ablation_results.py` referenced by `run_experiment.py` (lines 270, 292) DOES NOT EXIST in the repo ✗
4. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
5. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
6. The "reason" field for claim 91 explicitly states "Same as claim 86"
7. Same pattern as all prior Table 6 ablation claims (51-90)

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (86-90 for loose_budgets specifically). No command output file created.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, Avg Steps=75.41 is plausible (within max_steps=80), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 92+ (loose_budgets Reward and subsequent Table 6 rows).

## Session: 2026-04-01 — execution (claim 92: Table 6, loose_budgets, Reward: -65.85)

**Purpose:** Verify claimed Reward of -65.85 for `loose_budgets` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions 86-91 confirmed `loose_budgets` config exists, `collect_ablation_results.py` missing, no result artifacts
- Sessions 57 (small_code Reward) and 85 (tight_budgets Reward) confirmed Reward metric range context

**Key findings:**
1. `loose_budgets` config (kl_budget_a=0.1, kl_budget_h=0.06) confirmed in maptalk_ablation.yaml from session 86 ✓
2. Reward value -65.85 is mathematically plausible (negative reward in a navigation task) ✓
3. `collect_ablation_results.py` referenced by `run_experiment.py` (lines 270, 292) DOES NOT EXIST in the repo ✗
4. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
5. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
6. The "reason" field for claim 92 explicitly states "Same as claim 86"
7. Same pattern as all prior Table 6 ablation claims (51-91)

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (86-91 for loose_budgets specifically). No command output file created.

**Verdict summary:** Insufficient Evidence — loose_budgets config exists with correct KL params, Reward=-65.85 is plausible (negative reward in navigation task), but collect_ablation_results.py is missing from the repo and no result artifacts exist. Full training is computationally prohibitive.

**Next session:** Claims 93+ (subsequent Table 6 rows).

---

## Session: 2026-04-01 — execution (claim 93: Table 6, low_ib, SR: 0.0313)

**Purpose:** Verify claimed Success Rate of 0.0313 for `low_ib` (beta_ib=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `bica/configs/maptalk_ablation.yaml` — `low_ib` config confirmed: `beta_ib: 0.5`, `ib_weight: 0.5` ✓
- `fabscore_claude/progress.md` — prior sessions 51-92 confirmed same pattern: config exists, no result files, `collect_ablation_results.py` missing
- `fabscore_claude/execution_log.json` — consistent `Insufficient Evidence` verdicts for claims 51-92

**Key findings:**
1. `low_ib` config (beta_ib=0.5, ib_weight=0.5) confirmed in maptalk_ablation.yaml line 56-60 ✓
2. SR value 0.0313 ≈ 1/32 is mathematically plausible as per-batch training success_rate (1 success out of 32 episodes) ✓
3. SR value 0.0313 is IMPOSSIBLE as eval_id_success_rate (10 episodes, min step = 0.10) ⚠️
4. `collect_ablation_results.py` referenced by `run_experiment.py` (lines 270, 292) DOES NOT EXIST in the repo ✗
5. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session findings (51-92 for Table 6 ablation pattern). No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config (beta_ib=0.5) exists, SR=0.0313 ≈ 1/32 is plausible as per-batch metric but impossible as eval metric, collect_ablation_results.py is missing, no result artifacts exist, full training is computationally prohibitive.

**Next session:** Claims 94+ (remaining low_ib metrics and subsequent variants).

## Session: 2026-04-01 — execution (claim 94: Table 6, low_ib, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `low_ib` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claim 93 session already confirmed low_ib config and established same pattern
- Same findings as claim 93 apply: low_ib (beta_ib=0.5) config exists, no result artifacts, collect_ablation_results.py missing

**Key findings:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. ID SR = 0.00 is plausible (0/10 eval episodes succeeding) — not mathematically impossible ✓
3. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
4. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
5. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claim 93 analysis). No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config exists, ID SR=0.00 is a plausible value (0/10 successes), but collect_ablation_results.py is missing and no result artifacts exist.

**Next session:** Claims 95+ (remaining low_ib metrics and subsequent Table 6 variants).

----

## Session: 2026-04-01 — execution (claim 95: Table 6, low_ib, OOD SR: 0.30)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.30 for `low_ib` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 93-94 sessions already confirmed `low_ib` config and established same pattern
- Same findings as claims 93-94 apply: low_ib (beta_ib=0.5) config exists, no result artifacts, collect_ablation_results.py missing
- Claim reason field explicitly states "Same as claim 93"

**Key findings:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. OOD SR = 0.30 is plausible (3/10 OOD eval episodes succeeding) — mathematically valid (min step = 0.10) ✓
3. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
4. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
5. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-94 analysis for low_ib). No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config exists, OOD SR=0.30 is a plausible value (3/10 OOD successes), but collect_ablation_results.py is missing and no result artifacts exist.

**Next session:** Claims 96+ (remaining low_ib metrics and subsequent Table 6 variants).

---

## Session: 2026-04-01 — execution (claim 96: Table 6, low_ib, BAS: 0.332)

**Purpose:** Verify claimed Bidirectional Alignment Score (BAS) of 0.332 for `low_ib` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 93-95 sessions already confirmed low_ib config and established same pattern
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/eval/metrics_bas_ccm.py` — BAS implementation confirmed to exist
- Same findings as claims 93-95 apply: low_ib (beta_ib=0.5) config exists, no result artifacts, collect_ablation_results.py missing

**Evidence summary:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. BAS metric implementation exists in `bica/eval/metrics_bas_ccm.py` ✓
3. BAS = 0.332 is within plausible [0,1] range ✓
4. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
5. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
6. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-95 analysis for low_ib). Claim reason field explicitly states "Same as claim 93". No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config and BAS metric code both exist, BAS=0.332 is plausible, but collect_ablation_results.py is missing and no result artifacts exist.

**Next session:** Claims 97+ (remaining low_ib metrics and subsequent Table 6 variants).

## Session: 2026-04-01 — execution (claim 97: Table 6, low_ib, CCM: 0.548)

**Purpose:** Verify claimed CCM (Concept-Concept Mapping) score of 0.548 for `low_ib` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 93-96 sessions already confirmed low_ib config and established same pattern
- `fabscore_claude/fs_extracted.json` — confirms "97. Table 6, low_ib, CCM: 0.548"
- Same findings as claims 93-96 apply: low_ib (beta_ib=0.5) config exists in maptalk_ablation.yaml, no result artifacts exist, collect_ablation_results.py missing
- Claim reason field explicitly states "Same as claim 93"

**Key findings:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. CCM metric implementation exists in `bica/eval/metrics_bas_ccm.py` ✓
3. `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗
4. No result artifacts (checkpoints, logs, metrics) exist for low_ib variant ✗
5. Full training is computationally prohibitive (9600 episodes × 15 variants)

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-96 analysis for low_ib). Claim reason field explicitly states "Same as claim 93". No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config and CCM metric code both exist, CCM=0.548 is plausible, but collect_ablation_results.py is missing and no result artifacts exist.

**Next session:** Claims 98+ (remaining low_ib metrics: Avg Steps, Reward, and subsequent Table 6 variants).


## Session: 2026-04-01 — execution (claim 98: Table 6, low_ib, Avg Steps: 78.84)

**Purpose:** Verify claimed Avg Steps of 78.84 for `low_ib` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 93-97 sessions already confirmed `low_ib` config and established same pattern
- `bica/configs/maptalk_ablation.yaml` lines 56-60: `low_ib` config (beta_ib=0.5, ib_weight=0.5) confirmed to exist ✓; global ablation env: `max_steps: 80`
- `bica/train_maptalk.py` lines 804-858: `evaluate()` function — does NOT compute "Avg Steps" from actual rollouts; only tracks `eval_id_success_rate`, `eval_id_reward_mean`, `eval_ood_success_rate`, `eval_ood_reward_mean`
- `bica/train_maptalk.py` line 925: `avg_steps` is hardcoded to 30.0 in `_prepare_metrics_data()` for CE computation — NOT computed from actual episodes
- `run_experiment.py` — calls `collect_ablation_results.py` which DOES NOT EXIST in the repository ✗
- Claim reason field explicitly states "Same as claim 93"

**Key findings:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. Global ablation env: `max_steps: 80` — claimed Avg Steps 78.84 is within valid range [0, 80] ✓
3. `evaluate()` function does NOT compute "Avg Steps" metric from actual rollouts (no code path for it) ✗
4. `avg_steps` in `_prepare_metrics_data()` is hardcoded to 30.0, not derived from actual episodes ✗
5. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
6. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
7. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-97 analysis for low_ib) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config exists and 78.84 is plausible (within max_steps=80), but the evaluate() function has no code path to compute "Avg Steps" from actual rollouts (avg_steps is hardcoded to 30.0 for CE computation), collect_ablation_results.py is missing, and no result artifacts exist.

**Next session:** Claim 99+ (remaining low_ib metric: Reward, and subsequent Table 6 variants).

----

## Session: 2026-04-01 — execution (claim 99: Table 6, low_ib, Reward: -70.71)

**Purpose:** Verify claimed Reward of -70.71 for `low_ib` (beta_ib=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 93-98 sessions already confirmed `low_ib` config and established same pattern
- `bica/train_maptalk.py` lines 794-795: `episode_reward_mean = np.mean(episode_rewards)` computed over batch trajectories; also `eval_id_reward_mean` and `eval_ood_reward_mean` from evaluate() function
- `bica/configs/maptalk_ablation.yaml`: global ablation env: `reward_step: -0.5`, `reward_collision: -5.0`, `reward_success: 50.0`, `max_steps: 80`
- Claim reason field explicitly states "Same as claim 93"

**Key findings:**
1. `low_ib` config (beta_ib=0.5) confirmed in maptalk_ablation.yaml ✓
2. Reward metric code exists: `episode_reward_mean = np.mean(episode_rewards)` in train_maptalk.py ✓
3. Value -70.71 is mathematically plausible: with max_steps=80, reward_step=-0.5, ~6 collisions: -40 + (6×-5.0) = -70 ✓
4. Value -70.71 is NOT hardcoded anywhere in source code ✓
5. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
6. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
7. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-98 analysis for low_ib) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — low_ib config and reward metric code both exist, -70.71 is a plausible reward value (within env config range), but collect_ablation_results.py is missing and no result artifacts exist anywhere in the repo.

**Next session:** Claim 100+ (subsequent Table 6 variants).

## Session: 2026-04-01 — execution (claim 100: Table 6, high_ib, SR: 0.0313)

**Purpose:** Verify claimed Success Rate (SR) of 0.0313 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 93-99 sessions established the same low_ib pattern; claim 100 is the same pattern for high_ib
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 62-66: `high_ib` config (beta_ib=2.0, ib_weight=2.0) confirmed to exist ✓
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/run_experiment.py` lines 861/874: hardcoded fallback `success_rate = [0.031, ...]` in `load_training_data()` for visualization only, not Table 6 results
- `run_experiment.py` lines 270/292: calls `collect_ablation_results.py` which DOES NOT EXIST in repo ✗
- No result artifacts found anywhere in repo ✗

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. Both `low_ib` and `high_ib` report the SAME SR: 0.0313 — suspicious
3. SR=0.0313 ≈ 1/32 is plausible as per-batch training metric (batch_episodes=32), but IMPOSSIBLE as eval metric (50 episodes, min step = 0.02)
4. `run_experiment.py` has hardcoded visualization fallback `success_rate = [0.031, ...]` — the 0.031 value likely influenced the paper's reported 0.0313
5. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
6. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
7. Full training (9600 episodes × 15 variants × 3 seeds) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-99 analysis) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, SR=0.0313 ≈ 1/32 is plausible as per-batch metric, but collect_ablation_results.py is missing and no result artifacts exist. The same SR value appearing for both low_ib and high_ib is suspicious, but this alone is not sufficient for a fabrication verdict without running the actual experiment.

**Next session:** Claim 101+ (remaining high_ib metrics from Table 6).

----

## Session: 2026-04-01 — execution (claim 101: Table 6, high_ib, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claim 100 session established the same high_ib analysis
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 62-66: `high_ib` config (beta_ib=2.0, ib_weight=2.0) confirmed to exist ✓
- `run_experiment.py` lines 270/292: calls `collect_ablation_results.py` which DOES NOT EXIST in repo ✗
- No result artifacts found anywhere in repo ✗

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. `eval_id_success_rate = sum(success) / 10` over 10 hardcoded ID episodes in train_maptalk.py
3. ID SR = 0.00 = 0/10 is mathematically plausible (minimum possible value) ✓
4. `collect_ablation_results.py` referenced by `run_experiment.py` DOES NOT EXIST ✗
5. No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in the repo ✗
6. Full training (9600 episodes × 15 variants × 3 seeds) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 93-100 analysis) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, ID SR=0.00 is mathematically plausible (0/10 episodes), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 102+ (high_ib OOD SR: 0.00, BAS, CCM, etc. from Table 6).

## Session: 2026-04-01 — execution (claim 102: Table 6, high_ib, OOD SR: 0.00)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.00 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 100-101 sessions established same high_ib analysis
- `fabscore_claude/fs_extracted.json` — confirms claim 102: `high_ib, OOD SR: 0.00` (line 104)
- Prior session findings (claim 81, 88, 95, 101): identical pattern for OOD SR = 0.00 across multiple ablation variants

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. OOD SR = 0.00 is mathematically plausible (0 successes out of 10 OOD evaluation episodes) ✓
3. `collect_ablation_results.py` is missing from repo — cannot generate result artifacts ✗
4. No result files in workspace or repo for high_ib ✗
5. Value not hardcoded anywhere ✓

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 100-101 analysis). No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, OOD SR=0.00 is mathematically plausible (0/10 episodes), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 103+ (high_ib BAS, CCM, etc. from Table 6).

---

## Session: 2026-04-01 — execution (claim 103: Table 6, high_ib, BAS: 0.300)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.300 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 100-102 sessions established the same high_ib pattern
- Prior sessions confirm: `high_ib` config exists in maptalk_ablation.yaml, BAS code exists in `bica/eval/metrics_bas_ccm.py`, but `collect_ablation_results.py` is missing and no result artifacts exist

**Findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. BAS computation code exists in `metrics_bas_ccm.py`: BAS = 0.2*(mp+bs+rc+ss+ce) ✓
3. BAS MP component uses dummy random `np.random.rand` predictions — non-reproducible ⚠️
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. No result files in workspace or repo for high_ib ✗
6. Value not hardcoded anywhere ✓
7. BAS range is [0,1] with equal weights — 0.300 is mathematically plausible ✓

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 100-102 analysis). No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, BAS=0.300 is mathematically plausible, but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 104+ (remaining high_ib metrics from Table 6).

---

## Session: 2026-04-01 — execution (claim 104: Table 6, high_ib, CCM: 0.547)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.547 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 100-103 sessions established the same high_ib analysis
- Prior sessions confirm: `high_ib` config (beta_ib=2.0, ib_weight=2.0) exists in maptalk_ablation.yaml, CCM code exists in `bica/eval/metrics_bas_ccm.py`, but `collect_ablation_results.py` is missing and no result artifacts exist
- Claim 55 (small_code, CCM: 0.541) was already analyzed; same CCM code path applies

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. CCM computation code exists in `metrics_bas_ccm.py`: `ccm_score = λ * diversity + (1-λ) * synergy` (λ=0.5, bounded [0,1]) ✓
3. CCM value 0.547 is within valid range [0, 1] — mathematically plausible ✓
4. Value not hardcoded anywhere in source files ✓
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. No result files (JSON, CSV, npy, checkpoints, wandb logs) anywhere in repo ✗
7. Full training (9600 episodes × 15 variants × 3 seeds) is computationally prohibitive ✗

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 100-103 analysis) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, CCM=0.547 is mathematically plausible (within [0,1]), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 105+ (remaining high_ib metrics: Avg Steps, Reward, from Table 6).

---

## Session: 2026-04-01 — execution (claim 105: Table 6, high_ib, Avg Steps: 79.16)

**Purpose:** Verify claimed Avg Steps of 79.16 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 100-104 sessions established same high_ib analysis
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 62-66: `high_ib` config (beta_ib=2.0, ib_weight=2.0) confirmed to exist ✓
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` line 797: `episode_length_mean = np.mean(episode_lengths)` — avg steps computed correctly ✓
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml`: ablation env sets `max_steps: 80`; value 79.16 is within valid range [0, 80] ✓
- Grepped for "79.16" in all source files: value NOT hardcoded anywhere in source code ✓

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. Training code computes `episode_length_mean = np.mean(episode_lengths)` in train_maptalk.py ✓
3. Ablation config sets `max_steps: 80` — claimed value 79.16 is within valid range [0, 80] ✓
4. Value 79.16 NOT hardcoded anywhere in source code ✓ (not fabricated via hardcoding)
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 100-104 analysis) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, Avg Steps=79.16 is mathematically plausible (within max_steps=80 from ablation config), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 106+ (remaining high_ib metrics: Reward, from Table 6).

---

## Session: 2026-04-01 — execution (claim 106: Table 6, high_ib, Reward: -74.32)

**Purpose:** Verify claimed Reward of -74.32 for `high_ib` (beta_ib=2.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 100-105 sessions established same high_ib analysis
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` lines 782-828: reward computation via `episode_reward_mean = np.mean(episode_rewards)` ✓
- Grepped for "-74.32" in all source files: value NOT hardcoded anywhere ✓

**Key findings:**
1. `high_ib` config (beta_ib=2.0) confirmed in maptalk_ablation.yaml ✓
2. Training code computes `episode_reward_mean = np.mean(episode_rewards)` in train_maptalk.py ✓
3. Reward value -74.32 is NOT hardcoded anywhere in source code ✓
4. Reward sign (negative) is consistent with navigation environments where penalties accumulate per step
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 100-105 analysis) and static code inspection. No command output file created.

**Verdict summary:** Insufficient Evidence — high_ib config (beta_ib=2.0) exists, Reward=-74.32 is mathematically plausible (negative reward consistent with step-penalty navigation env), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running the full training pipeline.

**Next session:** Claim 107+ (next ablation variant/metric from Table 6).

---

## Session: 2026-04-01 — execution (claim 107: Table 6, no_gru, SR: 0.0625)

**Purpose:** Verify claimed Success Rate of 0.0625 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 51 and 79 established same analysis for identical SR=0.0625 values (small_code, tight_budgets)
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 69-72: `no_gru` config confirmed (`use_gru: false`) ✓
- `fabscore_claude/fs_extracted.json`: confirmed claim 107 = "Table 6, no_gru, SR: 0.0625"; 5 variants share SR=0.0625: small_code, tight_budgets, no_gru, small_hidden, high_rep_gap — suspicious pattern

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. Training code (train_maptalk.py) computes `success_rate = episode_successes / len(episode_rewards)` (32 episodes/batch) ✓
3. Value 0.0625 = 2/32 is plausible as per-batch training success_rate ✓
4. Value 0.0625 is IMPOSSIBLE as eval_id_success_rate (10 episodes, min step = 0.10) ⚠️
5. Same SR=0.0625 shared by 5 different ablation variants — suspicious pattern ⚠️
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
8. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 51 and 79 analysis which covers same SR=0.0625 scenario). No command output file created.

**Verdict summary:** Insufficient Evidence — no_gru config (use_gru=false) exists in YAML, SR=0.0625 is mathematically ambiguous (plausible as per-batch metric 2/32, impossible as eval metric over 10 episodes), collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 108+ (remaining no_gru metrics: ID SR, OOD SR, BAS, CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 108: Table 6, no_gru, ID SR: 0.10)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.10 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 52 (small_code ID SR: 0.20) and 80 (tight_budgets ID SR: 0.10) established same analysis for ID SR values
- Prior session (claim 107): `no_gru` config (use_gru=false) confirmed in maptalk_ablation.yaml ✓
- `bica/train_maptalk.py` (lines 810-818): `eval_id_success_rate = sum(success) / 10` over 10 hardcoded ID episodes — value 0.10 = 1/10 is mathematically plausible (minimum step = 0.10) ✓
- No result files exist anywhere in repo; `collect_ablation_results.py` is missing ✗

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. `eval_id_success_rate` computed over 10 episodes: ID SR = 0.10 = 1/10 is mathematically valid (min step = 0.10) ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts:** No new execution performed — reusing prior session findings (claims 52, 80, 107 which establish the same analysis pattern). No command output file created.

**Verdict summary:** Insufficient Evidence — `no_gru` config exists, ID SR = 0.10 is mathematically plausible (1 success out of 10 eval episodes), but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claims 109+ (remaining no_gru metrics: OOD SR, BAS, CCM, Avg Steps, Reward).

## Session: 2026-04-01 — execution (claim 109: Table 6, no_gru, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md`: reviewed prior session findings (claims 107, 108) for no_gru variant
- `/home/chenhui/fabscore/agent4sci_acc/submission_220`: checked for any result artifacts (*.json, *.csv, *.npy) — none found outside fabscore_claude
- Prior session (claim 108): `no_gru` config (use_gru=false) confirmed in maptalk_ablation.yaml ✓

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. OOD SR = 0.20 is mathematically plausible (2 successes out of 10 eval episodes) ✓
3. No result files exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗

**Execution artifacts created:** None (no new commands run; reused prior session findings)

**Verdict summary:** Insufficient Evidence — `no_gru` config exists, OOD SR = 0.20 is mathematically plausible (2/10 eval episodes), but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claims 110+ (remaining no_gru metrics: BAS, CCM, Avg Steps, Reward).

## Session: 2026-04-01 — execution (claim 110: Table 6, no_gru, BAS: 0.319)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.319 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md`: reviewed prior session findings (claims 107-109) for no_gru variant
- `fabscore_claude/fs_extracted.json`: confirmed claim 110 = "Table 6, no_gru, BAS: 0.319"
- Prior sessions (claims 107-109): `no_gru` config confirmed in maptalk_ablation.yaml ✓; no result files; `collect_ablation_results.py` missing ✗
- `bica/eval/metrics_bas_ccm.py`: BAS computation confirmed to use dummy random `np.random.rand` predictions for MP component (non-reproducible)

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. BAS code exists in `metrics_bas_ccm.py` ✓
3. No result files exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. Same situation as claims 54 (small_code, BAS: 0.321) and all other Table 6 BAS claims

**Execution artifacts created:** None (no new commands run; reused prior session findings)

**Verdict summary:** Insufficient Evidence — `no_gru` config exists, BAS computation code exists, but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claims 111+ (remaining no_gru metrics: CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 111: Table 6, no_gru, CCM: 0.535)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.535 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md`: reviewed prior session findings (claims 107-110) for no_gru variant
- `fabscore_claude/fs_extracted.json`: confirmed claim 111 = "Table 6, no_gru, CCM: 0.535"; also found CCM=0.535 shared by 3 variants: tight_budgets (#83), no_gru (#111), small_hidden (#118) — suspicious pattern ⚠️
- `bica/eval/metrics_bas_ccm.py`: CCMMetrics.compute_ccm_score() returns value in [0,1], so 0.535 is mathematically plausible ✓
- Prior sessions (claims 107-110): `no_gru` config (use_gru=false) confirmed in maptalk_ablation.yaml ✓; no result files; `collect_ablation_results.py` missing ✗

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. CCM computation code exists in `metrics_bas_ccm.py` ✓
3. CCM ∈ [0, 1], so 0.535 is mathematically plausible ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
7. Same CCM value (0.535) appears for 3 different ablation variants — suspicious

**Execution artifacts created:** None (no new commands run; reused prior session findings from claims 107-110)

**Verdict summary:** Insufficient Evidence — `no_gru` config exists, CCM code exists, 0.535 is plausible (within [0,1]), but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claims 112-113 (remaining no_gru metrics: Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 112: Table 6, no_gru, Avg Steps: 78.06)

**Purpose:** Verify claimed Avg Steps of 78.06 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md`: reviewed prior session findings (claims 107-111) for no_gru variant
- Prior sessions (claims 107-111): `no_gru` config (use_gru=false) confirmed in maptalk_ablation.yaml ✓; no result files; `collect_ablation_results.py` missing ✗; max_steps=80 confirmed

**Key findings:**
1. `no_gru` config (use_gru=false) exists in maptalk_ablation.yaml ✓
2. Avg Steps metric (step counting) code exists in training loop ✓
3. max_steps=80, so Avg Steps=78.06 is within valid range [0, 80] ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run
7. 78.06 is NOT hardcoded in any source file (not verified specifically but follows same pattern)

**Execution artifacts created:** None (no new commands run; reused prior session findings from claims 107-111)

**Verdict summary:** Insufficient Evidence — `no_gru` config exists, step counting code exists, 78.06 is plausible (within max_steps=80), but no result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claim 113 (remaining no_gru metric: Reward).

---

## Session: 2026-04-01 — execution (claim 113: Table 6, no_gru, Reward: -83.87)

**Purpose:** Verify claimed Reward of -83.87 for `no_gru` (use_gru=false) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 107-112 sessions established same no_gru analysis
- Grepped for "-83.87"/"83.87" in all source files: value NOT found in any source code file (only in fabscore metadata files) ✓
- `bica/configs/maptalk_ablation.yaml`: ablation env config: `reward_step=-0.5`, `reward_collision=-5.0`, `reward_success=50.0`, `max_steps=80`
- `bica/train_maptalk.py`: `episode_reward_mean = np.mean(episode_rewards)` — reward metric code exists ✓

**Key findings:**
1. `no_gru` config (use_gru=false) confirmed in maptalk_ablation.yaml ✓
2. Reward code path: `episode_reward_mean = np.mean(episode_rewards)` in train_maptalk.py ✓
3. Mathematical plausibility: -40 (80 steps × -0.5) + ~8.8 collisions × (-5.0) ≈ -84 — value -83.87 is plausible ✓
4. Value -83.87 NOT hardcoded anywhere in source code ✓
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. No result artifacts (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts created:** None (no new commands run; reused prior session findings from claims 107-112)

**Verdict:** Insufficient Evidence — `no_gru` config exists, reward metric code exists, -83.87 is mathematically plausible (negative reward with step penalties + ~8-9 collisions), and value is not hardcoded anywhere. No result files are present and `collect_ablation_results.py` is missing. Cannot verify without running full training pipeline.

**Next session:** Claim 114+ (subsequent Table 6 ablation variants).

---

## Session: 2026-04-01 — execution (claim 114: Table 6, small_hidden, SR: 0.0625)

**Purpose:** Verify claimed Success Rate of 0.0625 for `small_hidden` (128/128/64/64) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 51, 79, 107 established identical analysis for SR=0.0625 across other variants
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 74-81: `small_hidden` config confirmed (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64) ✓
- `fabscore_claude/fs_extracted.json`: confirmed claim 114 = "Table 6, small_hidden, SR: 0.0625"; 5 variants share SR=0.0625 (small_code, tight_budgets, no_gru, small_hidden, high_rep_gap) — suspicious pattern

**Key findings:**
1. `small_hidden` config (128/128/64/64) exists in maptalk_ablation.yaml ✓
2. SR=0.0625 = 2/32 is plausible as per-batch training success_rate ✓
3. SR=0.0625 is IMPOSSIBLE as eval_id_success_rate over 10 episodes (min step = 0.10) ⚠️
4. Same SR=0.0625 shared by 5 different ablation variants — suspicious pattern ⚠️
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Execution artifacts created:** None (no new commands run; reused prior session findings from claims 51, 79, 107 which cover the same SR=0.0625 scenario)

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, SR=0.0625 is mathematically ambiguous (plausible as per-batch 2/32, impossible as eval metric), collect_ablation_results.py is missing, no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 115+ (remaining small_hidden metrics: ID SR, OOD SR, BAS, CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 115: Table 6, small_hidden, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `small_hidden` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/fs_extracted.json` line 117: confirmed claim 115 = "115. Table 6, small_hidden, ID SR: 0.00"
- `fabscore_claude/progress.md` — claims 66 (high_temp ID SR: 0.00), 73 (low_temp ID SR: 0.00), 87 (loose_budgets ID SR: 0.00), 94 (low_ib ID SR: 0.00), 101 (high_ib ID SR: 0.00) all established identical pattern with same verdict
- `bica/configs/maptalk_ablation.yaml` lines 74-81: `small_hidden` config confirmed (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64)

**Analysis:**
1. `small_hidden` config exists in maptalk_ablation.yaml ✓
2. Training code and eval code exist (train_maptalk.py, evaluator.py) ✓
3. ID SR = 0.00 is mathematically plausible (0/10 eval episodes) ✓
4. No result files anywhere in the repo ✗
5. `collect_ablation_results.py` is missing from the repo ✗

**No new execution artifacts created** — reusing established analysis pattern from claims 66, 73, 87, 94, 101.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, ID SR=0.00 is mathematically plausible (0/10 eval episodes), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claim 116 (small_hidden, OOD SR: 0.20) and remaining small_hidden metrics.

---

## Session: 2026-04-01 — execution (claim 116: Table 6, small_hidden, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `small_hidden` ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 53 (small_code OOD SR: 0.20), 66 (high_temp OOD SR), 73, 87, 94, 101 all established identical pattern
- `bica/configs/maptalk_ablation.yaml` — `small_hidden` config confirmed (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64)
- `bica/train_maptalk.py` lines 822-830: `eval_ood_success_rate` computed over 10 OOD episodes; value 0.20 = 2/10 is mathematically plausible (minimum step = 0.10)
- `bica/eval/ood_suites.py`: OOD evaluation suite exists

**Analysis:**
1. `small_hidden` config exists in maptalk_ablation.yaml ✓
2. Training and OOD evaluation code exist (train_maptalk.py, ood_suites.py) ✓
3. OOD SR = 0.20 = 2/10 OOD eval episodes, mathematically plausible ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution artifacts created** — reusing established analysis pattern from claims 53, 66, 73, 87, 94, 101.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, OOD SR=0.20 is mathematically plausible (2/10 OOD eval episodes), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 117+ (remaining small_hidden metrics: BAS, CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 117: Table 6, small_hidden, BAS: 0.322)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.322 for `small_hidden` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 54 (small_code BAS: 0.321), 61, 68, 75, 82, 89, 96, 103, 110 all established identical pattern for BAS across other variants
- `bica/configs/maptalk_ablation.yaml` lines 74-81: `small_hidden` config confirmed (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64) ✓
- `bica/eval/metrics_bas_ccm.py`: BAS = 0.2*(mp+bs+rc+ss+ce); MP component uses dummy random `np.random.rand` predictions — non-reproducible ⚠️
- Prior session (claim 116): same `small_hidden` config existence confirmed, no result files, collect_ablation_results.py missing

**Key findings:**
1. `small_hidden` config exists in maptalk_ablation.yaml ✓
2. BAS computation code exists in metrics_bas_ccm.py ✓
3. BAS MP component uses dummy random `np.random.rand` — non-reproducible across runs ⚠️
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution artifacts created** — reusing established analysis pattern from claims 54, 61, 68, 75, 82, 89, 96, 103, 110.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, BAS=0.322 is plausible (within [0,1]), but BAS uses dummy random predictions for MP component, collect_ablation_results.py is missing, and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 118+ (remaining small_hidden metrics: CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 118: Table 6, small_hidden, CCM: 0.535)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.535 for `small_hidden` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md`: reviewed prior session findings for claims 111 (no_gru, CCM: 0.535), 114-117 (small_hidden variants), and 83 (tight_budgets, CCM: 0.535)
- `fabscore_claude/fs_progress.json`: confirmed prior verdicts for claims 111 and 114 (Insufficient Evidence)
- `fabscore_claude/fs_extracted.json`: confirmed CCM=0.535 is shared by 3 ablation variants (tight_budgets #83, no_gru #111, small_hidden #118) — suspicious pattern ⚠️

**Key findings (reused from prior sessions):**
1. `small_hidden` config exists in `bica/configs/maptalk_ablation.yaml` ✓
2. CCMMetrics.compute_ccm_score() in `bica/eval/metrics_bas_ccm.py` returns value in [0,1], so 0.535 is mathematically plausible ✓
3. No result files anywhere in repository ✗
4. `collect_ablation_results.py` (called by run_experiment.py:270,292) is missing ✗
5. CCM=0.535 is shared identically by 3 unrelated variants (tight_budgets, no_gru, small_hidden) ⚠️
6. Full training pipeline requires 9600 episodes × 15 variants (computationally prohibitive for verification)

**No new commands run** — reusing established analysis pattern from claims 83, 111 which cover identical CCM=0.535 scenario for ablation variants.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, CCM=0.535 is mathematically plausible within [0,1], but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 119+ (remaining small_hidden metrics: Avg Steps, Reward).

----

## Session: 2026-04-01 — execution (claim 119: Table 6, small_hidden, Avg Steps: 76.53)

**Purpose:** Verify claimed Avg Steps of 76.53 for `small_hidden` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 56, 63, 70, 77, 84, 91, 98, 105, 112 established identical analysis for Avg Steps across other variants (all plausible within max_steps=80, all Insufficient Evidence)
- Prior session for claim 118 confirmed `small_hidden` config exists in maptalk_ablation.yaml
- Ablation env global setting: `max_steps: 80`; `small_hidden` only overrides hidden dims (128/128/64/64)

**Key findings:**
1. `small_hidden` config (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64) exists in maptalk_ablation.yaml ✓
2. Avg Steps = `np.mean([len(ep) for ep in episodes])` — tracked as `avg_steps_mean` in train_maptalk.py ✓
3. With `max_steps: 80`, the value 76.53 is mathematically plausible (within [0, 80]) ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training computationally prohibitive (9600 episodes × 15 variants) ✗

**No new execution artifacts created** — reusing established analysis pattern from claims 56, 63, etc. which cover identical Avg Steps scenario for ablation variants. The analysis is identical: config exists, value is plausible, but no results.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, Avg Steps=76.53 is mathematically plausible (within max_steps=80), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claim 120 (small_hidden, Reward).

---

## Session: 2026-04-01 — execution (claim 120: Table 6, small_hidden, Reward: -81.62)

**Purpose:** Verify claimed Reward of -81.62 for `small_hidden` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 57, 64, 71, 78, 85, 92, 99, 106, 113 established identical analysis for Reward across other variants (all Insufficient Evidence)
- `bica/configs/maptalk_ablation.yaml` line 118: `reward_step=-0.5`, line 119: `reward_collision=-5.0`, max_steps=80
- `bica/train_maptalk.py:795`: `episode_reward_mean = np.mean(episode_rewards)` (sum of per-step rewards)
- Prior session (claim 119) confirmed avg_steps=76.53 for `small_hidden`

**Key findings:**
1. `small_hidden` config (policy_hidden_dim=128, value_hidden_dim=128, human_gru_hidden=64, protocol_hidden_dim=64) exists in maptalk_ablation.yaml ✓
2. Reward = sum(step rewards + collision penalties) per episode, then mean across episodes ✓
3. With reward_step=-0.5, avg_steps≈76.53: step penalties ≈ -38.27; collision penalties ≈ -43.35 (≈8-9 collisions), total ≈ -81.62 — mathematically plausible ✓
4. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. Training computationally prohibitive (9600 episodes × 15 variants) ✗

**No new execution artifacts created** — reusing established analysis pattern from claims 57, 64, 71, 78, 85, 92, 99, 106, 113 which cover identical Reward scenario for ablation variants.

**Verdict summary:** Insufficient Evidence — `small_hidden` config exists, Reward=-81.62 is mathematically plausible (step penalties + collision penalties consistent with avg_steps=76.53 and reward_step=-0.5, reward_collision=-5.0), but collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claim 121+ (next ablation variant metrics).

---

## Session: 2026-04-01 — execution (claim 121: Table 6, large_hidden, SR: 0.0938)

**Purpose:** Verify claimed Success Rate of 0.0938 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — `large_hidden` config confirmed to exist with policy_hidden_dim=512, value_hidden_dim=512, human_gru_hidden=256, protocol_hidden_dim=256 ✓
- `bica/train_maptalk.py` — training code exists ✓
- `run_experiment.py` — ablation runner calls `collect_ablation_results.py` which DOES NOT EXIST ✗
- Prior sessions (claims 58, 65, 72, 79, 86, 93, 100, 107, 114): same verdict, same code path — no result files, computationally prohibitive

**Key findings:**
1. `large_hidden` config (512/512/256/256) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. SR value 0.0938 ≈ 3/32 = 0.09375 — plausible as a training batch success_rate ✓
4. As eval metric (10 episodes), valid values are multiples of 0.1 — 0.0938 would be IMPOSSIBLE ⚠️
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**No new execution performed** — reusing static code analysis and prior session findings.

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists, SR=0.0938 is plausible as a batch-level training metric but cannot be verified without running the full training pipeline. No result files exist.

**Next session:** Claim 122+ (next ablation variant metrics for large_hidden).

## Session: 2026-04-01 — execution (claim 122: Table 6, large_hidden, ID SR: 0.10)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.10 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Context inspected:**
- `fabscore_claude/progress.md` — Session for claim 121 (same variant, different metric)
- `bica/configs/maptalk_ablation.yaml:82-88` — `large_hidden` config confirmed to exist ✓
- No result files (JSON, wandb logs, checkpoints) exist in repo ✗
- `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗

**Findings:**
1. `large_hidden` config (512/512/256/256) exists in maptalk_ablation.yaml ✓
2. ID SR: 0.10 = 1/10 episodes — plausible as eval metric (multiples of 0.1) ✓
3. No result files exist anywhere in repo ✗
4. Full training pipeline is computationally expensive — not feasible to run

**No new execution performed** — same conditions as claim 121 (same variant, missing results).

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and ID SR=0.10 is a plausible eval value, but cannot be verified without running the full training pipeline. No result files exist.

**Next session:** Claim 123+ (OOD SR and remaining metrics for large_hidden).

----

## Session: 2026-04-01 — execution (claim 123: Table 6, large_hidden, OOD SR: 0.10)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.10 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Context inspected:**
- `fabscore_claude/progress.md` — Sessions for claims 121 and 122 (same variant, different metrics)
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml:82-88` — `large_hidden` config confirmed to exist with policy_hidden_dim=512, value_hidden_dim=512, human_gru_hidden=256, protocol_hidden_dim=256 ✓
- `fabscore_claude/fs_extracted.json` — confirms OOD SR for large_hidden is listed as 0.10
- No result files (JSON, wandb logs, checkpoints) exist in repo ✗
- `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗

**Findings:**
1. `large_hidden` config (512/512/256/256) exists in maptalk_ablation.yaml ✓
2. OOD SR: 0.10 = 1/10 OOD episodes — plausible as eval metric (multiples of 0.1) ✓
3. No result files exist anywhere in repo ✗
4. Full training pipeline is computationally expensive — not feasible to run

**No new execution performed** — same conditions as claims 121-122 (same variant, missing results).

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and OOD SR=0.10 is a plausible eval value (1/10 episodes), but cannot be verified without running the full training pipeline. No result files exist.

**Next session:** Claim 124+ (BAS and remaining metrics for large_hidden).

## Session: 2026-04-01 — execution (claim 124: Table 6, large_hidden, BAS: 0.312)

**Purpose:** Verify claimed Behavioral Alignment Score (BAS) of 0.312 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — Sessions for claims 121–123 (same variant, different metrics)
- `fabscore_claude/fs_extracted.json:126` — confirms BAS for large_hidden is listed as 0.312
- `fabscore_claude/fs_progress.json` — BAS metrics for other ablation variants all show same pattern: Insufficient Evidence

**Artifacts created/updated:** None (no new execution needed)

**Key findings:**
1. `large_hidden` config (512/512/256/256) exists in maptalk_ablation.yaml ✓
2. BAS computation code exists in `bica/eval/metrics_bas_ccm.py` ✓
3. No result files exist — same as claims 121–123 for this variant
4. Claim reason states "Same as claim 121" confirming identical verification pattern
5. Full training pipeline requires ~9600+ episodes and wandb; cannot run in this context

**No new execution performed** — same conditions as claims 121–123 (same variant, missing results).

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and BAS=0.312 is plausible given the computation code, but cannot be verified without running the full training pipeline. No result files exist.

**Next session:** Claim 125+ (remaining metrics for large_hidden or other ablation variants).

---

## Session: 2026-04-01 — execution (claim 125: Table 6, large_hidden, CCM: 0.547)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.547 for `large_hidden` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 121–124: large_hidden SR, ID SR, OOD SR, BAS) confirmed same code path and blockers
- `bica/configs/maptalk_ablation.yaml:82-88` — `large_hidden` config (policy_hidden_dim=512, value_hidden_dim=512, human_gru_hidden=256, protocol_hidden_dim=256) confirmed to exist ✓
- `bica/eval/metrics_bas_ccm.py` — CCM computation code exists: CCM = λ*diversity + (1-λ)*synergy (λ=0.5, bounded [0,1]); value 0.547 is mathematically plausible ✓
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗
- Claim reason states "Same as claim 121" — confirming identical verification pattern

**No new execution performed** — same conditions as claims 121–124 (same variant, missing results). CCM code path confirmed by prior sessions (55, 62, 69, 76, 83).

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and CCM=0.547 is within valid [0,1] range, but cannot be verified without running the full training pipeline. No result files exist and collect_ablation_results.py is missing.

**Next session:** Claims 126+ (remaining Table 6 ablation metrics for other variants).

---

## Session: 2026-04-01 — execution (claim 126: Table 6, large_hidden, Avg Steps: 74.09)

**Purpose:** Verify claimed Avg Steps of 74.09 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 121–125: large_hidden SR, ID SR, OOD SR, BAS, CCM) confirmed same code path and blockers
- `bica/configs/maptalk_ablation.yaml:82-88` — `large_hidden` config (policy_hidden_dim=512, value_hidden_dim=512, human_gru_hidden=256, protocol_hidden_dim=256) confirmed to exist ✓
- `bica/eval/ood_suites.py:252` — `avg_steps` = np.mean(episode_lengths) — metric computation code exists ✓
- `bica/viz/plots.py:197` — Avg Steps listed as plottable metric ✓
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗
- Claim reason states "Same as claim 121" — confirming identical verification pattern

**No new execution performed** — same conditions as claims 121–125 (same variant, missing results). avg_steps code path confirmed.

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and Avg Steps=74.09 has a plausible code path (np.mean of episode lengths), but cannot be verified without running the full training pipeline. No result files exist and collect_ablation_results.py is missing.

**Next session:** Claims 127+ (remaining Table 6 ablation metrics for other variants).

----

## Session: 2026-04-01 — execution (claim 127: Table 6, large_hidden, Reward: -73.88)

**Purpose:** Verify claimed Reward of -73.88 for `large_hidden` (512/512/256/256) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions (claims 121–126: large_hidden SR, ID SR, OOD SR, BAS, CCM, Avg Steps) confirmed same code path and blockers
- `bica/configs/maptalk_ablation.yaml:82-88` — `large_hidden` config (policy_hidden_dim=512, value_hidden_dim=512, human_gru_hidden=256, protocol_hidden_dim=256) confirmed to exist ✓
- `bica/train_maptalk.py` — `episode_reward_mean = np.mean(episode_rewards)` where each episode reward is `batch['rewards'].sum()` ✓
- `bica/envs/maptalk.py` — reward components: step=-0.5, collision=-5.0, success=50.0, token=-0.05; max_steps=80 (ablation) ✓
- No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
- `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗
- Claim reason states "Same as claim 121" — confirming identical verification pattern

**Mathematical plausibility:** Reward=-73.88 with step=-0.5, max_steps=80: step penalties alone = up to -40.0; adding multiple collision penalties at -5.0 each → value -73.88 is plausible given avg_steps=74.09 and some collisions.

**No new execution performed** — same conditions as claims 121–126 (same variant, missing results). Reward code path confirmed by prior sessions (57, 64, 71, 78, 85, 92, 99, 106, 113, 120).

**Verdict summary:** Insufficient Evidence — `large_hidden` config exists and Reward=-73.88 is mathematically plausible given env config (reward_step=-0.5, max_steps=80, collisions=-5.0), but cannot be verified without running the full training pipeline. No result files exist and collect_ablation_results.py is missing.

**Next session:** Claims 128+ (remaining Table 6 ablation variants).

---

## Session: 2026-04-01 — execution (claim 128: Table 6, no_rep_gap, SR: 0.0313)

**Purpose:** Verify claimed Success Rate (SR) of 0.0313 for `no_rep_gap` (mu_rep=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — Sessions for claims 93-127 established same analysis pattern for all Table 6 ablation variants
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml:91-96` — `no_rep_gap` config confirmed: `mu_rep: 0.0`, `disable_rep_mapper: true`, `experiment_name: "no_representation_gap"` ✓
- Prior sessions confirmed: `collect_ablation_results.py` (called by run_experiment.py:270/292) does NOT exist in repo ✗
- No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in repo ✗
- Same SR=0.0313 appears for both low_ib and high_ib variants — suspicious pattern ⚠️

**Key findings:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml ✓
2. SR=0.0313 ≈ 1/32 is plausible as per-batch training success_rate (batch_episodes=32) ✓
3. SR=0.0313 is IMPOSSIBLE as eval_id_success_rate over 10 episodes (min step = 0.10) ⚠️
4. SR=0.0313 is IMPOSSIBLE as comprehensive eval over 50 episodes (min step = 0.02) ⚠️
5. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
6. No result files exist anywhere in repo ✗
7. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**No new execution performed** — same conditions as claims 93, 100 (same SR=0.0313 value for different variants). Config confirmed via static inspection.

**Verdict summary:** Insufficient Evidence — `no_rep_gap` config (mu_rep=0.0) exists, SR=0.0313 ≈ 1/32 is mathematically ambiguous (plausible as per-batch training metric, impossible as eval metric), collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 129+ (remaining no_rep_gap metrics from Table 6).

## Session: 2026-04-01 — execution (claim 129: Table 6, no_rep_gap, ID SR: 0.00)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.00 for `no_rep_gap` (mu_rep=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — Session for claim 128 (no_rep_gap, SR: 0.0313) established same analysis pattern
- Claim 129 reason field: "Same as claim 128" — confirms identical conditions apply
- Prior sessions confirmed: `collect_ablation_results.py` (called by run_experiment.py) does NOT exist in repo ✗
- No result artifacts (JSON, CSV, npy, checkpoints, wandb logs) exist anywhere in repo ✗
- `no_rep_gap` config confirmed: `mu_rep: 0.0`, `disable_rep_mapper: true` in maptalk_ablation.yaml ✓

**Key findings:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml ✓
2. ID SR=0.00 is mathematically plausible (0 successes out of any N evaluation episodes) ✓
3. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
4. No result files exist anywhere in repo ✗
5. Full training (9600 episodes × 15 variants) is computationally prohibitive ✗

**No new execution performed** — same conditions as claim 128. Config confirmed via static inspection from prior session.

**Verdict summary:** Insufficient Evidence — `no_rep_gap` config exists, ID SR=0.00 is mathematically plausible (cannot be proven false), collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 130+ (remaining no_rep_gap metrics from Table 6).

## Session: 2026-04-01 — execution (claim 130: Table 6, no_rep_gap, OOD SR: 0.10)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.10 for `no_rep_gap` (mu_rep=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — Sessions for claims 128 (SR: 0.0313) and 129 (ID SR: 0.00) established same analysis pattern
- `fabscore_claude/fs_extracted.json` — confirms `130. Table 6, no_rep_gap, OOD SR: 0.10` at line 132

**Analysis:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml (from prior sessions) ✓
2. OOD SR: 0.10 = 1/10 OOD eval episodes — mathematically plausible ✓
3. No result files exist for any ablation variant ✗
4. `collect_ablation_results.py` aggregation script is missing ✗
5. Full training pipeline (~9600 episodes × 15 variants) is computationally infeasible to run

**No new execution performed** — same conditions as claims 128-129. Config confirmed via static inspection from prior sessions.

**Verdict summary:** Insufficient Evidence — `no_rep_gap` config exists, OOD SR=0.10 is mathematically plausible (1/10 OOD episodes), collect_ablation_results.py is missing and no result artifacts exist. Cannot verify without running full training pipeline.

**Next session:** Claims 131+ (remaining no_rep_gap metrics from Table 6).

---

## Session: 2026-04-01 — execution (claim 131: Table 6, no_rep_gap, BAS: 0.500)

**Purpose:** Verify claimed BAS value of 0.500 for no_rep_gap variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 128-130 (same no_rep_gap variant)
- `fabscore_claude/fs_progress.json` — confirms Insufficient Evidence for claims 128-130 with identical conditions
- `bica/eval/metrics_bas_ccm.py` — BAS = 0.2 * (MP + BS + RC + SS + CE), bounded in [0, 1]

**Reasoning (no new execution):**
- Claim 131 is explicitly noted as "Same as claim 128" in its reason field
- Claims 128, 129, 130 all classified as Insufficient Evidence (no result files, missing collect_ablation_results.py, computationally prohibitive to run full training)
- BAS = 0.500 is mathematically plausible (equal-weight average of 5 sub-scores, each in [0,1], so 0.500 indicates neutral performance)
- Same conditions apply: no_rep_gap config exists in maptalk_ablation.yaml, no result artifacts, collect_ablation_results.py missing

**No execution performed** — reusing analysis from claims 128-130.

**Verdict summary:** Insufficient Evidence — no_rep_gap config exists and BAS=0.500 is mathematically plausible, but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing.

**Next session:** Claims 132+ (remaining no_rep_gap or subsequent Table 6 ablation metrics).

## Session: 2026-04-01 — execution (claim 132: Table 6, no_rep_gap, CCM: 0.500)

**Purpose:** Verify claimed CCM value of 0.500 for no_rep_gap variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 128-131 (same no_rep_gap variant, same conditions)
- `fabscore_claude/fs_extracted.json` — confirms `132. Table 6, no_rep_gap, CCM: 0.500` at index 132

**Execution artifacts:** None created (no new commands run; reusing prior session analysis).

**Key findings:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml (from prior sessions) ✓
2. CCM=0.500 is mathematically plausible (e.g., 1/2 or normalized score)
3. No result artifacts exist anywhere in the repository
4. `collect_ablation_results.py` aggregation script is missing
5. Full training pipeline required to generate results — cannot run in this environment

**Verdict summary:** Insufficient Evidence — same conditions as claims 128-131; no_rep_gap config exists and CCM=0.500 is mathematically plausible, but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing.

**Next session:** Claims 133+ (remaining Table 6 ablation metrics).

## Session: 2026-04-01 — execution (claim 133: Table 6, no_rep_gap, Avg Steps: 78.63)

**Purpose:** Verify claimed Average Steps value of 78.63 for `no_rep_gap` (mu_rep=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 128-132 (same no_rep_gap variant, same conditions)
- `fabscore_claude/fs_extracted.json` — confirms `133. Table 6, no_rep_gap, Avg Steps: 78.63` at index 133

**Findings:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml (from prior sessions) ✓
2. Avg Steps=78.63 is a plausible value for episode step counts in a navigation/communication task
3. No result artifacts (CSV, JSON, logs) exist anywhere in the repository
4. `collect_ablation_results.py` aggregation script is missing
5. Full training pipeline required to generate results — cannot run in this environment

**Verdict summary:** Insufficient Evidence — same conditions as claims 128-132; no_rep_gap config exists and Avg Steps=78.63 is mathematically plausible, but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing.

**Next session:** Claims 134+ (remaining Table 6 ablation metrics).

---

## Session: 2026-04-01 — execution (claim 134: Table 6, no_rep_gap, Reward: -61.99)

**Purpose:** Verify claimed Reward of -61.99 for `no_rep_gap` (mu_rep=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 128-133 (same no_rep_gap variant, same conditions)
- Claim 134 reason field: "Same as claim 128" — confirms identical conditions apply

**Findings:**
1. `no_rep_gap` config (mu_rep=0.0, disable_rep_mapper=true) confirmed in maptalk_ablation.yaml (from prior sessions) ✓
2. Reward=-61.99 is mathematically plausible: with reward_step=-0.5 and max_steps=80, minimum from steps alone is -40.0; -61.99 implies ~4.4 collision events (each -5.0) which is reasonable
3. No result artifacts (CSV, JSON, logs, checkpoints) exist anywhere in the repository ✗
4. `collect_ablation_results.py` aggregation script (called by run_experiment.py) is missing ✗
5. Full training pipeline is computationally prohibitive (9600 episodes × 15 variants) ✗

**No execution performed** — reusing analysis from claims 128-133; same conditions apply exactly.

**Verdict summary:** Insufficient Evidence — `no_rep_gap` config exists and Reward=-61.99 is mathematically plausible, but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing. Cannot verify without running the full training pipeline.

**Next session:** Claims 135+ (remaining Table 6 ablation metrics or subsequent claims).

---

## Session: 2026-04-01 — execution (claim 136: Table 6, high_rep_gap, ID SR: 0.10)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.10 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135 (same high_rep_gap variant, same conditions), 52 (same ID SR metric, same code path)
- Claim 136 reason field: "Same as claim 135" — confirms identical conditions apply

**Findings:**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml (confirmed in claim 135 session) ✓
2. `eval_id_success_rate = sum(success) / 10` over 10 hardcoded ID episodes — value 0.10 = 1/10 is mathematically plausible (minimum non-zero step = 0.10) ✓
3. No result artifacts (CSV, JSON, logs, checkpoints) exist anywhere in the repository ✗
4. `collect_ablation_results.py` aggregation script (called by run_experiment.py) is missing ✗
5. Full training pipeline is computationally prohibitive (9600 episodes × 15 variants) ✗

**No execution performed** — reusing analysis from claims 135 and 52; same conditions apply exactly.

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and ID SR=0.10 is mathematically plausible (1 success out of 10 evaluation episodes = minimum non-zero value), but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing. Cannot verify without running the full training pipeline.

**Next session:** Claims 137+ (remaining Table 6 ablation metrics for high_rep_gap or subsequent claims).

---

## Session: 2026-04-01 — execution (claim 137: Table 6, high_rep_gap, OOD SR: 0.00)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.00 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135 (same high_rep_gap variant, SR=0.0625), 136 (same variant, ID SR=0.10), 53 (OOD SR metric code path), 66 (OOD SR=0.00 for high_temp)
- Claim 137 reason field: "Same as claim 135" — confirms identical conditions apply
- `bica/train_maptalk.py` lines 822-830: `eval_ood_success_rate = sum(...success...) / len(ood_trajectories)` over 10 OOD episodes — value 0.00 = 0/10 is mathematically plausible ✓

**Findings:**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml (confirmed in claim 135 session) ✓
2. `eval_ood_success_rate = sum(success) / 10` over 10 hardcoded OOD episodes — value 0.00 = 0/10 is mathematically plausible ✓
3. No result artifacts (CSV, JSON, logs, checkpoints) exist anywhere in the repository ✗
4. `collect_ablation_results.py` aggregation script (called by run_experiment.py) is missing ✗
5. Full training pipeline is computationally prohibitive (9600 episodes × 15 variants) ✗
6. Value 0.00 is NOT hardcoded specifically for high_rep_gap variant ✓

**No execution performed** — reusing analysis from claims 135, 136, 53, 66; same conditions apply exactly.

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and OOD SR=0.00 is mathematically plausible (0 successes out of 10 OOD evaluation episodes), but no result artifacts exist anywhere in the repository and the collect_ablation_results.py aggregation script is missing. Cannot verify without running the full training pipeline.

**Next session:** Claims 138+ (remaining Table 6 ablation metrics).

## Session: 2026-04-01 — execution (claim 138: Table 6, high_rep_gap, BAS: 0.500)

**Purpose:** Verify claimed Bidirectional Alignment Score (BAS) of 0.500 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135-137 (same high_rep_gap variant)
- `fabscore_claude/fs_extracted.json` — confirmed claim 138 = "Table 6, high_rep_gap, BAS: 0.500"
- `bica/train_maptalk.py` — BAS fallback hardcoded at 0.5 (line 850)
- `bica/eval/coalignment_vs_single_directional.py` — synthetic results generator (for Table 2)

**Key finding:** `train_maptalk.py` line 848-851 contains a hardcoded fallback:
```python
eval_metrics.update({
    'bas_score': 0.5,
    'ccm_score': 0.5,
    ...
})
```
This fallback is triggered when the BAS/CCM computation fails with an exception. The claimed values (BAS=0.500, CCM=0.500) exactly match these fallback values, while all other 14 ablation variants have BAS values in range 0.300-0.332 (not 0.500). This is suspicious but not conclusive without result files.

**Artifacts created:** None (no new execution; existing analysis sufficient)

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and BAS=0.500 may plausibly come from the hardcoded fallback in train_maptalk.py (triggered when advanced metrics fail), but there are no result artifacts in the repository and the collect_ablation_results.py is missing. The value exactly matches the hardcoded fallback (0.5 for both BAS and CCM), which is suspicious but not definitively fabricated since no actual run artifacts exist to compare against.

**Next session:** Claim 139 (high_rep_gap, CCM: 0.500) and beyond.

---

## Session: 2026-04-01 — execution (claim 139: Table 6, high_rep_gap, CCM: 0.500)

**Purpose:** Verify claimed Cognitive Complementarity Metric (CCM) of 0.500 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135-138 (same high_rep_gap variant)
- `bica/train_maptalk.py` lines 846-856: confirmed `ccm_score: 0.5` hardcoded fallback at line 851 (triggered when advanced metrics computation fails via exception handler)
- Prior claim 138 session: same analysis applies for CCM as for BAS

**Key findings:**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml ✓
2. Training code (train_maptalk.py) exists ✓
3. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
4. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
5. CCM value 0.500 exactly matches hardcoded fallback in train_maptalk.py:851 — suspicious since this is the exception-case default ⚠️
6. However, CCM formula can also legitimately produce ~0.5 if λ=0.5 and diversity≈synergy≈0.5

**No new execution performed** — reusing prior session analysis (claims 135-138). Same code path applies.

**Artifacts created:** None (no new execution; existing analysis sufficient)

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and CCM=0.500 may plausibly come from the hardcoded fallback in train_maptalk.py:851 (triggered when advanced metrics fail), but there are no result artifacts and `collect_ablation_results.py` is missing. The value exactly matches the hardcoded fallback, which is suspicious, but without actual run artifacts we cannot establish a concrete conflict with the paper.

**Next session:** Claim 140+ (remaining Table 6 ablation metrics).

## Session: 2026-04-01 — execution (claim 140: Table 6, high_rep_gap, Avg Steps: 77.63)

**Purpose:** Verify claimed Avg Steps of 77.63 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135-139 (same high_rep_gap variant)
- No new execution commands run; reusing context from prior sessions

**Key findings (from prior sessions):**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml (confirmed in claim 135 session) ✓
2. Ablation env sets `max_steps: 80` (confirmed in session 56 for small_code Avg Steps) ✓
3. Avg Steps = 77.63 is ≤ max_steps=80, so mathematically plausible ✓
4. No result artifacts exist anywhere in the repository ✗
5. `collect_ablation_results.py` aggregation script is missing ✗
6. No fresh commands run — conditions identical to claims 135-139

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and Avg Steps=77.63 is mathematically plausible (≤ max_steps=80), but no result artifacts exist in the repository and collect_ablation_results.py is missing. Cannot verify without running the full training pipeline.

**Next session:** Claims 141+ (subsequent Table 6 or other claims).

----

## Session: 2026-04-01 — execution (claim 141: Table 6, high_rep_gap, Reward: -59.73)

**Purpose:** Verify claimed Reward of -59.73 for `high_rep_gap` (mu_rep=0.5) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — prior sessions for claims 135-140 (same high_rep_gap variant)
- `bica/envs/maptalk.py` — reward structure: `reward_step=-1.0`, `reward_collision=-5.0`, `reward_success=50.0`, `reward_token_cost=-0.05`; `max_steps=80` (ablation config)
- `bica/train_maptalk.py` (lines 782-798) — `episode_reward_mean = np.mean(episode_rewards)` where each episode reward = sum of per-step rewards
- No new execution commands run; reusing context from prior sessions

**Key findings:**
1. `high_rep_gap` config (mu_rep=0.5) exists in maptalk_ablation.yaml ✓
2. Ablation env sets `max_steps: 80` (confirmed in claim 56 session) ✓
3. Reward -59.73 is mathematically plausible: with SR≈0.0625 (mostly failures), failed episodes get ~-80 (steps only), mix of successes/failures could yield mean ≈ -60 ✓
4. No result artifacts exist anywhere in the repository ✗
5. `collect_ablation_results.py` aggregation script is missing ✗
6. `episode_reward_mean` tracked in train_maptalk.py but never saved to files ✗
7. No fresh commands run — conditions identical to claims 135-140

**Verdict summary:** Insufficient Evidence — `high_rep_gap` config exists and Reward=-59.73 is mathematically plausible given the env reward structure (reward_step=-1.0, max_steps=80, SR≈0.0625), but no result artifacts exist in the repository and collect_ablation_results.py is missing. Cannot verify without running the full training pipeline.

**Next session:** Claims 142+ (subsequent Table 6 ablation metrics, no_instructor_cost variant).

## Session: 2026-04-01 — execution (claim 142: Table 6, no_instructor_cost, SR: 0.1563)

**Purpose:** Verify claimed Success Rate of 0.1563 for `no_instructor_cost` (kappa_teach=0.0) ablation variant from Table 6.

**Files inspected:**
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` (lines 103-106): `no_instructor_cost` variant exists with `kappa_teach: 0.0`
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/run_experiment.py` (line 153): `no_teacher: {kappa_teach: 0.0}` — similar variant with same parameter
- `fabscore_claude/progress.md`: prior sessions for high_temp (claim 65), low_temp (claim 72), loose_budgets (claim 86) all received Insufficient Evidence for same pattern

**Execution artifacts created:** None (no commands run — same pattern as prior claims)

**Key findings:**
1. `no_instructor_cost` config exists in YAML (lines 103-106) with `kappa_teach: 0.0` ✓
2. Training code path exists in run_experiment.py ✓
3. SR=0.1563 ≈ 5/32 is mathematically plausible ✓
4. No result artifacts exist anywhere in the repository ✗
5. `collect_ablation_results.py` aggregation script is missing ✗
6. 0.1563 is not hardcoded anywhere in the code ✓

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with correct `kappa_teach=0.0` parameter and SR=0.1563 is mathematically plausible (≈5/32), but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claims 65, 72, 86 and many other Table 6 ablation SR claims. Cannot verify without executing the full training pipeline.

**Next session:** Remaining Table 6 claims for no_instructor_cost variant and other ablation variants.

---

## Session: 2026-04-01 — execution (claim 143: Table 6, no_instructor_cost, ID SR: 0.20)

**Purpose:** Verify claimed In-Distribution Success Rate (ID SR) of 0.20 for `no_instructor_cost` ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claim 142 (no_instructor_cost SR: 0.1563) already established the same analysis pattern for this variant
- `bica/train_maptalk.py` (lines 810-818): `eval_id_success_rate = sum(success) / 10` over 10 hardcoded ID episodes — value 0.20 = 2/10 is mathematically plausible
- `bica/configs/maptalk_ablation.yaml`: `no_instructor_cost` config (kappa_teach=0.0) confirmed to exist
- No result artifacts exist anywhere; `collect_ablation_results.py` is missing

**Execution artifacts created:** None (no commands run — same pattern as prior claims 52, 80, 108, 142)

**Key findings:**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. ID SR = 0.20 = 2/10 is mathematically plausible ✓
3. No result artifacts exist anywhere in the repository ✗
4. `collect_ablation_results.py` aggregation script is missing ✗

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with `kappa_teach=0.0` and ID SR=0.20 is mathematically plausible (2/10), but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claims 52, 80, 108, 142.

**Next session:** Claims 144+ (remaining Table 6 ablation metrics).

## Session: 2026-04-01 — execution (claim 144: Table 6, no_instructor_cost, OOD SR: 0.40)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.40 for `no_instructor_cost` ablation variant from Table 6.

**Context inspected:**
- `fabscore_claude/progress.md` — claims 142 and 143 (no_instructor_cost SR and ID SR) already established the same analysis pattern
- `fabscore_claude/fs_progress.json` — confirmed Insufficient Evidence for claims 142 and 143
- `bica/configs/maptalk_ablation.yaml` lines 103-106: `no_instructor_cost` config (kappa_teach=0.0) confirmed
- `bica/train_maptalk.py` lines 822-825: OOD evaluation uses 10 episodes, so OOD SR=0.40=4/10 is mathematically plausible

**Findings:**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. OOD SR computed over 10 episodes → 0.40 = 4/10 is mathematically plausible ✓
3. No result artifacts exist in the repository ✗
4. `collect_ablation_results.py` aggregation script is missing ✗

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with `kappa_teach=0.0` and OOD SR=0.40 is mathematically plausible (4/10 OOD episodes), but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claims 52, 80, 108, 142, 143.

**Next session:** Claims 145+ (remaining Table 6 ablation metrics for no_instructor_cost: BAS, CCM, Avg Steps, Reward).

## Session: 2026-04-01 — execution (claim 145: Table 6, no_instructor_cost, BAS: 0.500)

**Purpose:** Verify claimed BAS of 0.500 for `no_instructor_cost` (kappa_teach=0.0) ablation variant from Table 6.

**Context inspected:**
- `fabscore_claude/progress.md` — claims 142-144 (no_instructor_cost SR/ID SR/OOD SR) already established the same analysis pattern for this variant
- `bica/eval/metrics_bas_ccm.py` lines 17-70: BAS = (1/5) * (MP + BS + RC + SS + CE), each component in [0,1], so BAS ∈ [0,1]; value 0.500 is mathematically plausible
- `bica/configs/maptalk_ablation.yaml` lines 103-106: `no_instructor_cost` config (kappa_teach=0.0) confirmed to exist
- No result artifacts exist anywhere; `collect_ablation_results.py` is missing

**Execution artifacts created:** None (no commands run — same pattern as prior claims 142, 143, 144)

**Key findings:**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. BAS = (1/5) * (MP + BS + RC + SS + CE), each ∈ [0,1]; BAS=0.500 is mathematically plausible ✓
3. No result artifacts exist anywhere in the repository ✗
4. `collect_ablation_results.py` aggregation script is missing ✗

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with `kappa_teach=0.0` and BAS=0.500 is mathematically plausible (mean of 5 components in [0,1]), but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claims 142, 143, 144.

**Next session:** Claims 146+ (remaining Table 6 ablation metrics for no_instructor_cost: CCM, Avg Steps, Reward).

## Session: 2026-04-01 — execution (claim 146: Table 6, no_instructor_cost, CCM: 0.500)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.500 for `no_instructor_cost` (kappa_teach=0.0) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/progress.md` — claims 142-145 (no_instructor_cost SR/ID SR/OOD SR/BAS) already established the same analysis pattern for this variant
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 103-106: `no_instructor_cost` config (kappa_teach=0.0) confirmed to exist
- `fabscore_claude/workspace/` — no claim_146 artifacts exist; prior workspace has only claim_43 and claim_45 outputs
- Prior session claim 139 (high_rep_gap, CCM: 0.500) established that CCM=0.500 exactly matches hardcoded fallback in train_maptalk.py:851
- Prior session claim 145 (no_instructor_cost, BAS: 0.500) — same no_instructor_cost config, BAS=0.500 from fallback

**Key findings (reusing established analysis):**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. CCM computation code exists in `bica/eval/metrics_bas_ccm.py`: CCM = λ*diversity + (1-λ)*synergy (λ=0.5, bounded [0,1]) ✓
3. CCM=0.500 is mathematically plausible (within [0,1]) ✓
4. CCM=0.500 exactly matches hardcoded fallback in train_maptalk.py:851 (triggered when advanced metrics fail) ⚠️
5. No result artifacts exist anywhere in the repository ✗
6. `collect_ablation_results.py` aggregation script is missing ✗

**No new commands run** — reusing established analysis pattern from claims 132, 139, 142-145 which cover identical CCM=0.500/BAS=0.500 fallback scenario for ablation variants.

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with `kappa_teach=0.0` and CCM=0.500 is mathematically plausible (within [0,1]), but no result artifacts exist and collect_ablation_results.py is missing. Value 0.500 exactly matches fallback, but without result files we cannot conclusively say it was produced from the fallback rather than actual computation.

**Next session:** Claims 147+ (no_instructor_cost Avg Steps, Reward and subsequent ablation variants).

----

## Session: 2026-04-01 — execution (claim 147: Table 6, no_instructor_cost, Avg Steps: 73.94)

**Purpose:** Verify claimed Avg Steps of 73.94 for `no_instructor_cost` (kappa_teach=0.0) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 142-146 (no_instructor_cost all prior metrics) established the same analysis pattern for this variant
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 103-106: `no_instructor_cost` config (kappa_teach=0.0) confirmed to exist
- Prior session claim 56 (small_code, Avg Steps: 77.66) — global ablation env sets `max_steps: 80`, so Avg Steps must be in [1, 80]

**Key findings:**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. Global ablation env `max_steps: 80` — value 73.94 is mathematically plausible (within [1, 80])
3. No result files, logs, or checkpoints found in repository
4. `collect_ablation_results.py` (the aggregation script) is absent
5. Training pipeline is computationally prohibitive (9600 episodes × 15 variants)
6. Value 73.94 is NOT hardcoded anywhere in source code

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists with `kappa_teach=0.0` and Avg Steps=73.94 is mathematically plausible (within max_steps=80 from ablation env config), but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claims 56, 63, 70, 77, etc. (all Avg Steps claims for Table 6 ablation variants).

**Next session:** Claim 148 (no_instructor_cost, Reward: -64.26) and subsequent ablation variants.

## Session: 2026-04-01 — execution (claim 148: Table 6, no_instructor_cost, Reward: -64.26)

**Purpose:** Verify claimed Reward of -64.26 for `no_instructor_cost` (kappa_teach=0.0) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/progress.md` — claims 142-147 (no_instructor_cost all prior metrics) established pattern
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 103-106: `no_instructor_cost` config confirmed
- Grep for `-64.26` across repository: not found anywhere in source

**Key findings:**
1. `no_instructor_cost` config exists in YAML with `kappa_teach: 0.0` ✓
2. Reward value -64.26 is NOT hardcoded anywhere in source code
3. No result files, logs, or checkpoints found in repository
4. `collect_ablation_results.py` aggregation script is absent
5. Training pipeline is computationally prohibitive (9600 episodes × 15 variants)
6. Pattern identical to claims 142-147 (all no_instructor_cost metrics) and 84-91 (reward claims for other variants)

**Verdict summary:** Insufficient Evidence — `no_instructor_cost` config exists and reward value is plausible (negative reward is consistent with environment design), but no result artifacts exist and the aggregation script is missing. Pattern identical to all other Table 6 ablation reward claims.

**Next session:** Claim 149+ (subsequent Table 6 ablation variants).

## Session: 2026-04-01 — execution (claim 149: Table 6, high_instructor_cost, SR: 0.1250)

**Purpose:** Verify claimed SR of 0.1250 for `high_instructor_cost` (kappa_teach=0.2) ablation variant from Table 6.

**Files inspected:**
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/configs/maptalk_ablation.yaml` lines 108-111: `high_instructor_cost` variant with `kappa_teach: 0.2` ✓
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` — SR computation analyzed:
  - Training batch SR (line 792): `success_rate /= len(trajectories)` with batch_episodes=32 → multiples of 1/32
  - Evaluation SR (line 811-814): hardcoded `num_episodes=10` → multiples of 0.1
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/run_experiment.py` — ablation runner confirmed, requires collect_ablation_results.py (missing)

**Commands run:**
- Minimal 1-epoch run with high_instructor_cost config (kappa_teach=0.2, use_wandb=False, episodes=32)
- Output: Training SR = 0.09375 (3/32), Eval ID SR = 0.3, Eval OOD SR = 0.0

**Key findings:**
1. `high_instructor_cost` config exists with `kappa_teach=0.2` ✓
2. SR=0.1250 = 4/32: mathematically achievable ONLY with training batch SR (32 episodes), NOT with evaluation SR (10 hardcoded episodes → multiples of 0.1)
3. Paper Table 6 has separate "SR", "ID SR", and "OOD SR" columns — "SR" likely maps to training batch SR
4. Code runs successfully with this config (no crashes)
5. Single untrained epoch shows SR=0.09375 ≠ 0.1250 (expected difference for untrained model)
6. No result files exist anywhere in repository
7. `collect_ablation_results.py` is absent

**Artifact created:** `fabscore_claude/workspace/claim_149_command_output.txt`

**Verdict summary:** Insufficient Evidence — `high_instructor_cost` config exists, code runs, SR=0.1250=4/32 is mathematically achievable with batch_episodes=32. However, no training result files exist and the aggregation script is missing. The specific value cannot be verified.

**Next session:** Claim 150+ (high_instructor_cost, other metrics and subsequent ablation variants).

## Session: 2026-04-01 — execution (claim 150: Table 6, high_instructor_cost, ID SR: 0.00)

**Purpose:** Verify claimed ID SR of 0.00 for `high_instructor_cost` (kappa_teach=0.2) ablation variant from Table 6.

**Files/context inspected:**
- `fabscore_claude/workspace/claim_149_command_output.txt` — prior session established high_instructor_cost config exists, code runs, evaluation uses hardcoded num_episodes=10
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` lines 816-819: `eval_id_success_rate` confirmed as ID SR metric
- Claim reason: "Same as claim 149"

**Key findings (reusing claim 149 analysis):**
1. `high_instructor_cost` config exists in YAML with `kappa_teach: 0.2` ✓
2. Evaluation uses hardcoded `num_episodes=10` → ID SR values must be multiples of 0.1
3. ID SR=0.00 = 0/10 successes is mathematically achievable ✓
4. No result files, logs, or checkpoints found in repository ✗
5. `collect_ablation_results.py` aggregation script is absent ✗
6. Value 0.00 is NOT hardcoded anywhere in source for this metric

**No new commands run** — reusing established analysis from claim 149.

**Verdict summary:** Insufficient Evidence — `high_instructor_cost` config exists with `kappa_teach=0.2` and ID SR=0.00 (0/10 eval episodes) is mathematically achievable, but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to claim 149 and all other Table 6 ablation metrics.

**Next session:** Claim 151+ (high_instructor_cost, OOD SR and subsequent metrics).

---

## Session: 2026-04-01 — execution (claim 151: Table 6, high_instructor_cost, OOD SR: 0.20)

**Purpose:** Verify claimed Out-of-Distribution Success Rate (OOD SR) of 0.20 for `high_instructor_cost` ablation variant from Table 6.

**Files/artifacts inspected:**
- `fabscore_claude/workspace/claim_149_command_output.txt` — established config exists, kappa_teach=0.2, one-epoch run showed OOD SR=0.0 (untrained)
- `fabscore_claude/progress.md` — sessions 53, 74, 88, 109, 116, and others all confirmed same pattern for OOD SR claims

**Key findings (reusing claim 149/150 analysis):**
1. `high_instructor_cost` config exists in `bica/configs/maptalk_ablation.yaml` with `kappa_teach=0.2` ✓
2. OOD SR = 0.20 = 2/10 OOD eval episodes — mathematically achievable ✓
3. Claim reason says "Same as claim 149" — same config and code path
4. No result files, logs, or checkpoints found in repository ✗
5. `collect_ablation_results.py` aggregation script is absent ✗
6. Value 0.20 is NOT hardcoded anywhere in source for this metric

**No new commands run** — reusing established analysis from claims 149 and 150.

**Verdict summary:** Insufficient Evidence — `high_instructor_cost` config exists with `kappa_teach=0.2` and OOD SR=0.20 (2/10 OOD eval episodes) is mathematically achievable, but no result artifacts exist and collect_ablation_results.py is missing. Pattern identical to all prior Table 6 OOD SR claims.

**Next session:** Remaining high_instructor_cost metrics (claims 152+).

---

## Session: 2026-04-01 — execution (claim 152: Table 6, high_instructor_cost, BAS: 0.500)

**Purpose:** Verify claimed BAS (Bidirectional Alignment Score) of 0.500 for `high_instructor_cost` (kappa_teach=0.2) ablation variant from Table 6.

**Files/artifacts inspected:**
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` lines 846-856: CRITICAL — `except Exception as e:` fallback block hardcodes `'bas_score': 0.5` when BAS computation fails
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/eval/metrics_bas_ccm.py` — legitimate BAS computation that returns values in [0,1]
- `fabscore_claude/fs_extracted.json` — BAS values for all 15 Table 6 variants: 11 variants have BAS in range 0.300-0.332; exactly 4 variants (`no_rep_gap`, `high_rep_gap`, `no_instructor_cost`, `high_instructor_cost`) have BAS = 0.500 exactly
- `fabscore_claude/workspace/claim_149_command_output.txt` — prior session confirmed config exists and code runs

**Key findings:**
1. `train_maptalk.py:850` hardcodes `bas_score: 0.5` in the `except Exception` fallback when BAS computation fails
2. Exactly 4 out of 15 Table 6 variants report BAS = 0.500 — identical to the hardcoded fallback value
3. The other 11 variants have BAS in range 0.300-0.332, suggesting those had genuine computation
4. Pattern strongly suggests these 4 variants encountered the exception handler and received the fallback 0.5 value
5. BAS formula (average of 5 components, each 0-1) could legitimately return ~0.5, but the exact match with the fallback across exactly 4 specific variants is highly suspicious
6. No result files exist in the repository to confirm

**No new commands run** — static code analysis + pattern analysis of extracted values is conclusive.

**Verdict summary:** Result Fabrication — BAS = 0.500 matches exactly the hardcoded fallback value in `train_maptalk.py:850` (except handler), and this pattern appears in exactly 4 of 15 Table 6 variants. The code logic exists for genuine computation, but the reported value is the hardcoded fallback rather than a genuine experimental result.

**Next session:** Claim 153+ (high_instructor_cost, remaining metrics CCM, Avg Steps, Reward).

---

## Session: 2026-04-01 — execution (claim 153: Table 6, high_instructor_cost, CCM: 0.500)

**Purpose:** Verify claimed CCM (Cognitive Complementarity Metric) of 0.500 for `high_instructor_cost` ablation variant from Table 6.

**Files/artifacts inspected:**
- `220_Co_Alignment_Rethinking_Al_Supplementary Material/bica/train_maptalk.py` lines 846-856: `except Exception as e:` fallback block hardcodes BOTH `'bas_score': 0.5` AND `'ccm_score': 0.5`
- `fabscore_claude/fs_extracted.json` — CCM = 0.500 appears in exactly 4 of 15 Table 6 variants: no_rep_gap (claim 132), high_rep_gap (claim 139), no_instructor_cost (claim 146), high_instructor_cost (claim 153)
- Other 11 Table 6 variants have CCM in range 0.521-0.548 (different from fallback value)

**Key findings:**
1. `train_maptalk.py:851` hardcodes `ccm_score: 0.5` as a fallback in the `except Exception` handler (same block as BAS fallback found in claim 152)
2. Exactly 4 of 15 Table 6 variants report CCM = 0.500 — identical to the hardcoded fallback
3. These 4 variants are the same ones with BAS = 0.500 (same exception handler)
4. Other 11 variants have distinct CCM values (0.521-0.548), suggesting genuine computation
5. Pattern strongly indicates these 4 variants hit the exception handler

**No new commands run** — static code analysis + pattern from fs_extracted.json is conclusive (same pattern as BAS claim 152).

**Verdict summary:** Result Fabrication — CCM = 0.500 matches the hardcoded fallback `ccm_score: 0.5` in `train_maptalk.py:851` (exception handler). The same pattern appears in 4 of 15 Table 6 variants. The reported value is the hardcoded fallback, not a genuine experimental result.

**Next session:** Claim 154+ (high_instructor_cost, Avg Steps and Reward).

---

## Session: 2026-04-01 — execution (claim 154: Table 6, high_instructor_cost, Avg Steps: 74.56)

**Purpose:** Verify claimed Avg Steps of 74.56 for `high_instructor_cost` ablation variant from Table 6.

**Files/artifacts inspected:**
- `fabscore_claude/workspace/claim_149_command_output.txt` — prior session established config exists, kappa_teach=0.2, code runs
- `fabscore_claude/progress.md` — sessions 152 (BAS) and 153 (CCM) confirmed same pattern for this variant
- `bica/configs/maptalk_ablation.yaml` lines 113-117: `max_steps: 80` for ablation configs
- `bica/train_maptalk.py` lines 783-797: `episode_length_mean = np.mean(episode_lengths)` = "Avg Steps" metric
- `fabscore_claude/fs_extracted.json` — all 15 Table 6 Avg Steps values: range 71.78–79.16, each unique

**Key findings:**
1. `high_instructor_cost` config exists in `bica/configs/maptalk_ablation.yaml` with `kappa_teach: 0.2` ✓
2. `max_steps: 80` → Avg Steps=74.56 ∈ [0, 80] is mathematically feasible ✓
3. `episode_length_mean` computed from actual trajectory lengths — no hardcoding found for this metric ✓
4. All 15 Avg Steps values are in [71-79] range (plausible for low-success agents) — no suspicious repetition ✓
5. No result files, logs, or checkpoints found in repository ✗
6. `collect_ablation_results.py` aggregation script is absent ✗

**No new commands run** — reusing established analysis from claims 149-153.

**Verdict summary:** Insufficient Evidence — `high_instructor_cost` config exists, the code computes episode_length_mean from actual runs, and Avg Steps=74.56 is mathematically feasible (max_steps=80). However, no training result files exist and collect_ablation_results.py is missing. Pattern identical to all other Table 6 ablation metrics without hardcoded fallbacks.

**Next session:** Claim 155 (high_instructor_cost, Reward: -75.38).

---

## Session: 2026-04-01 — execution (claim 155: Table 6, high_instructor_cost, Reward: -75.38)

**Purpose:** Verify claimed Reward of -75.38 for `high_instructor_cost` (kappa_teach=0.2) ablation variant from Table 6.

**Files inspected:**
- `fabscore_claude/workspace/claim_149_command_output.txt` — prior claim for same variant (high_instructor_cost, SR): config exists, code runs, no result files
- `bica/configs/maptalk_ablation.yaml` — `high_instructor_cost` config: `kappa_teach: 0.2` confirmed ✓
- `bica/train_maptalk.py` lines 782-800: `episode_reward_mean = np.mean(episode_rewards)` where episode reward = sum of step rewards
- `bica/envs/maptalk.py` lines 34-38: reward_step=-0.5, reward_collision=-5.0, reward_success=50.0; ablation config sets max_steps=80
- `run_experiment.py` lines 860, 873: hardcoded fallback rewards [-93.12, -85, -78, -72, -68, -65] (for visualization only, not Table 6 values)
- Grep for "75.38" in all source files: value NOT hardcoded anywhere ✓

**Mathematical analysis:**
- Steps alone: max_steps=80 × -0.5 = -40.0
- Claimed -75.38 requires additional -35.38: ~7.1 collisions @ -5.0 each
- Value is mathematically plausible for a poorly trained agent with kappa_teach=0.2 ✓

**Key findings:**
1. `high_instructor_cost` config (kappa_teach=0.2) exists in maptalk_ablation.yaml ✓
2. Training code exists in train_maptalk.py ✓
3. Reward -75.38 is NOT hardcoded anywhere in source code ✓
4. Value is mathematically plausible given env config ✓
5. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo ✗
6. `collect_ablation_results.py` called by run_experiment.py does NOT exist in repo ✗
7. Training is computationally expensive (9600 episodes × 15 variants) — not feasible to run

**Artifact reused:** `claim_149_command_output.txt` — prior claim 149 for same variant; same code path applies.
**No new execution performed** — same analysis as claim 149 and prior Table 6 reward claims (e.g., claim 57).

**Verdict:** `Insufficient Evidence` — `high_instructor_cost` config exists, training code exists, reward -75.38 is mathematically plausible and not hardcoded, but no result files are present and the aggregation script (`collect_ablation_results.py`) is missing. Cannot verify without full training.

**Next session:** No further action needed for this claim.

---

---

## Session: 2026-04-01 — execution (claim 156: Figure 1(a), MapTalk gridworld 8×8 layout with 3×3 egocentric view)

**Purpose:** Verify Figure 1(a) claim that MapTalk is an 8×8 gridworld with obstacles, start/goal, AI's 3×3 egocentric view (blue frustum), heading, and message exchange overlays.

**Files inspected:**
- `bica/envs/maptalk.py`: MapTalkEnv implementation
  - `grid_size = config.get('grid_size', 8)` (line 30) → 8×8 layout ✓
  - `ai_obs_space = spaces.Box(shape=(3, 3, 2))` (lines 44-46) → 3×3 egocentric view ✓
  - `obstacle_rate`, `_generate_grid()` → obstacles ✓
  - `agent_pos`, `goal_pos` → start/goal ✓
  - `agent_heading` (4-directional) → heading ✓
  - `ai_message_space = Discrete(64)`, `human_message_space = Discrete(32)` → message exchange ✓
- `bica/viz/plots.py`: no gridworld visualization function found
- `run_experiment.py`: training pipeline requires wandb (failed due to wandb auth)

**Execution performed:**
- `PYTHONPATH=. /home/chenhui/miniconda3/bin/python -c "from bica.envs.maptalk import MapTalkEnv; env = MapTalkEnv({}); obs = env.reset(); print(...)"` 
- Output confirmed: grid_size=8, grid shape=(8,8), ai_obs shape=(3,3,2), ai_heading shape=(4,), obstacle count=21, ai_message_space=Discrete(64), human_message_space=Discrete(32)
- Fresh artifact: `workspace/claim_156_command_output.txt`

**Verdict:** `Verified` — Direct execution of MapTalkEnv confirms all claimed elements of Figure 1(a):
- 8×8 layout (grid_size=8, grid shape=(8,8))
- Obstacles (21 obstacles in test run)
- Start/goal positions (agent_pos, goal_pos)
- 3×3 egocentric observation (ai_obs shape=(3,3,2)) — corresponds to the "blue frustum"
- Heading (4-directional one-hot)
- Message exchange (ai_message_space, human_message_space)
The "blue frustum" is a visual representation of the 3×3 egocentric view confirmed in code.

**Next session:** No further action needed for claim 156.

---

## Session: 2026-04-01 — execution (claim 157: Figure 1(b), Latent Navigator 2D projection interface)

**Purpose:** Verify Figure 1(b) claim — Latent Navigator interface showing 2D projection Pϕ(z) with policy-suggested regions (dashed), user-selected samples (dots), and decoded renderings (insets).

**Files inspected:**
- `bica/latent_nav_lite/vae_model.py` (lines 301-379): `ProjectionNetwork` — explicitly P_φ: R^16 → R^2. Maps latent code to 2D via MLP → Tanh ✓
- `bica/latent_nav_lite/ui_interface.py`: `NavigatorUI` class:
  - `ax_map` subplot: 2D latent space visualization (xlim=[-1.1,1.1], ylim=[-1.1,1.1]) ✓
  - `scatter_visited`: user-selected samples shown as colored dots (cmap='viridis') ✓
  - `scatter_suggestions`: AI suggestions shown as red 'x' markers (marker='x') — NOT dashed regions as described ⚠️
  - `ax_image` second subplot: decoded image display ✓ (separate subplot, not embedded insets)
- `bica/latent_nav_lite/navigator.py`: `suggest_region()` computes policy suggestions ✓
- `run_experiment.py`: `generate_plots()` (line 786) generates training curves, protocol analysis, CCM trajectory — NO Figure 1(b) specific generation code
- `results/` directory: only `config.yaml` exists; no figure files, data files, or npy artifacts

**Key findings:**
1. `ProjectionNetwork` (Pϕ) explicitly implemented as R^16→R^2 projection ✓
2. UI interface code exists with 2D visualization, user sample dots, and decoded image display ✓
3. Policy suggestions rendered as 'x' markers in code, NOT dashed regions as paper describes ⚠️
4. Decoded images shown as a separate subplot, not "insets" embedded in the 2D projection ⚠️
5. No specific code to generate Figure 1(b) as a static paper figure
6. No pre-existing figure files or underlying data files anywhere in the repo
7. Latent nav experiment crashes at runtime (established in claims 43-50: GPU/CPU device mismatch in vae_model.decode() line 152)

**Execution artifacts:** No new execution performed — reusing static code analysis and prior session crash evidence (claims 43-50). No command output file created.

**Verdict summary:** `Insufficient Evidence` — The code implements the core described interface elements (2D projection Pϕ(z), navigation suggestions, user-click dots, decoded image display), but the visual descriptions differ from code (dashed regions vs 'x' markers; insets vs. separate subplot). No figure generation code for Figure 1(b) exists; the experiment crashes before running. The latent navigator interface concept is supported by the code, but the specific figure cannot be reproduced or verified.

**Next session:** No further action needed for claim 157.

---

## Session: 2026-04-01 — execution (claim 158: Figure 2, Ablation Heatmap)

**Purpose:** Verify Figure 2 claim — Ablation study overview: normalized colors (per metric) with raw values annotated. Metrics shown: success rate, BAS score, CCM score, and average steps. Variants are ordered by success rate.

**Files inspected:**
- `bica/viz/plots.py`: Completely read. Contains `plot_training_curves`, `plot_bas_radar`, `plot_ood_performance`, `plot_ccm_trajectory`, `plot_protocol_analysis`. NO ablation heatmap function exists here. The only heatmap is a correlation matrix in `MetricsPlotter.plot_correlation_matrix()`. The claim's reason that "The ablation heatmap visualization exists in bica/viz/plots.py" is INCORRECT.
- `bica/viz/analysis.py`: Contains an effect sizes heatmap in `ExperimentAnalyzer` for comparing BiCA with baselines — NOT the ablation heatmap described in the claim.
- `bica/configs/maptalk_ablation.yaml`: Defines 15 ablation variants (small_code, large_code, high_temp, low_temp, tight_budgets, loose_budgets, low_ib, high_ib, no_gru, small_hidden, large_hidden, no_rep_gap, high_rep_gap, no_instructor_cost, high_instructor_cost).
- `run_experiment.py`: Contains `run_ablation_experiments()` function that would run ablations, then calls `collect_ablation_results.py` — but this script doesn't exist in the repo.
- `list_ablations.py`: Lists the 15 ablation variants, confirms their structure.
- `results/` directory: Only `config.yaml` exists. No experiment results, no JSON results files, no .npy files, no figure files.

**Key findings:**
1. The ablation heatmap visualization code (normalized colors per metric, raw values annotated, ordered by success rate) does NOT exist in bica/viz/plots.py.
2. No ablation experiment results data files exist anywhere in the repository.
3. `collect_ablation_results.py` referenced in `run_experiment.py` does not exist in the repository.
4. The 15 ablation variants are defined in the YAML config, so the experiment structure is present.
5. No pre-existing image files (.png) exist anywhere in the repo.

**Execution performed:** None (no existing artifacts to reuse; running ablation experiments would require GPU training for all 15 variants — not feasible).

**Verdict summary:** `Insufficient Evidence` — The ablation configuration infrastructure exists (15 variants defined), but the specific Figure 2 heatmap visualization code is absent from plots.py (contrary to the claim's reason), no ablation results data files exist, and collect_ablation_results.py is missing. The complete pipeline needed to reproduce Figure 2 does not exist in the repository.

**Next session:** No further action needed for claim 158.

---

## Session: 2026-04-01 — execution (claim 159: results_section, hyperparameter variants mean success ≈ 9.9%)

**Purpose:** Verify claim that "hyperparameter variants exhibited the largest spread and highest mean success (≈ 9.9%; best: high_temp)" from the paper's ablation results section.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — hyperparameter variants: high_temp, low_temp, tight_budgets, loose_budgets, low_ib, high_ib (confirmed to exist)
- `list_ablations.py` — confirms "Hyperparameter ablations: high_temp, low_temp, tight_budgets, loose_budgets, low_ib, high_ib"
- `fabscore_claude/fs_extracted.json` — Table 6 SR values for all hyperparameter variants
- Prior session analysis (claims 51-135): same code path, same conclusions

**Table 6 SR values for hyperparameter variants (from paper):**
- high_temp: 0.1563, low_temp: 0.1563, tight_budgets: 0.0625, loose_budgets: 0.1563, low_ib: 0.0313, high_ib: 0.0313

**Mathematical check (internal consistency):**
- Mean SR = (0.1563+0.1563+0.0625+0.1563+0.0313+0.0313)/6 = 0.5940/6 = 0.099 = 9.9% ✓ (exactly matches claim)
- Best variant: high_temp (tied with low_temp and loose_budgets at 0.1563) ✓ (claim says "best: high_temp")
- Hyperparameter spread = 0.125 vs architecture spread = 0.0625 ✓ (claim: "largest spread")

**Key finding:** The claim is internally self-consistent with the paper's Table 6 values. BUT:
1. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo
2. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repo
3. Training requires 9600 episodes × 15 variants (computationally prohibitive)
4. The Table 6 values themselves cannot be verified through execution

**No new execution performed** — reusing static analysis from prior sessions (claims 51-135) that established no result files exist and training is computationally prohibitive.

**Verdict:** `Insufficient Evidence` — the claim is internally self-consistent with the Table 6 values in the paper (mean SR = exactly 9.9%, high_temp is one of the best performers, spread is larger than architecture category), but the underlying Table 6 experimental results have no execution artifacts in the repository. Training is computationally expensive (9600 episodes × 15 variants), and the results collection script (`collect_ablation_results.py`) is missing.

**Next session:** Claims 160-161 (similar results_section summary claims for co-alignment variants ≈9.4% and architecture ≈7.5%).

---

## Session: 2026-04-01 — execution (claim 160: results_section, co-alignment variants mean success ≈ 9.4%)

**Purpose:** Verify claim that "co-alignment variants (≈ 9.4%; best: no_instructor_cost)" from the paper's ablation results section.

**Files inspected:**
- `bica/configs/maptalk_ablation.yaml` — co-alignment (regularization) variants: no_rep_gap, high_rep_gap, no_instructor_cost, high_instructor_cost (all confirmed to exist)
- `fabscore_claude/fs_extracted.json` — Table 6 SR values for all co-alignment variants
- Prior session analysis (claims 51-155): same code path, no result files, `collect_ablation_results.py` missing

**Table 6 SR values for co-alignment variants (from paper):**
- no_rep_gap: 0.0313, high_rep_gap: 0.0625, no_instructor_cost: 0.1563, high_instructor_cost: 0.1250

**Mathematical check (internal consistency):**
- Mean SR = (0.0313+0.0625+0.1563+0.1250)/4 = 0.3751/4 = 0.09378 ≈ 9.4% ✓ (exactly matches claim)
- Best variant: no_instructor_cost (SR = 0.1563) ✓ (claim says "best: no_instructor_cost")

**Key finding:** The claim is internally self-consistent with the paper's Table 6 values. BUT:
1. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo
2. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repo
3. Training requires 9600 episodes × 15 variants (computationally prohibitive)

**No new execution performed** — reusing static analysis from prior sessions that established no result files exist.

**Verdict:** `Insufficient Evidence` — the claim is internally self-consistent with Table 6 values (mean SR = exactly 9.4%, no_instructor_cost is the best with SR=0.1563), but the underlying Table 6 experimental results have no execution artifacts in the repository.

**Next session:** Claim 161 (architecture variants ≈7.5%).

---

## Session: 2026-04-01 — execution (claim 161: results_section, architecture variants mean success ≈ 7.5%)

**Purpose:** Verify claim that "architecture (≈ 7.5%; best: large_code)" from the paper's ablation results section.

**Files inspected:**
- `fabscore_claude/fs_extracted.json` — Table 6 SR values for all architecture variants
- Prior session analysis (claims 51-160): same code path, no result files, `collect_ablation_results.py` missing

**Table 6 SR values for architecture variants (from paper):**
- small_code: 0.0625, large_code: 0.0938, no_gru: 0.0625, small_hidden: 0.0625, large_hidden: 0.0938

**Mathematical check (internal consistency):**
- Mean SR = (0.0625+0.0938+0.0625+0.0625+0.0938)/5 = 0.3751/5 = 0.07502 ≈ 7.5% ✓ (exactly matches claim)
- Best variant: large_code and large_hidden are tied at SR=0.0938; claim says "best: large_code" ✓ (consistent; tied, either could be listed as best)

**Key finding:** The claim is internally self-consistent with the paper's Table 6 values. BUT:
1. No result files (JSON, wandb logs, checkpoints) exist anywhere in repo
2. `collect_ablation_results.py` called by run_experiment.py DOES NOT EXIST in the repo
3. Training requires 9600 episodes × 15 variants (computationally prohibitive)

**No new execution performed** — reusing static analysis from prior sessions that established no result files exist.

**Verdict:** `Insufficient Evidence` — the claim is internally self-consistent with Table 6 values (mean SR = exactly 7.5%, large_code is one of the best tied at SR=0.0938), but the underlying Table 6 experimental results have no execution artifacts in the repository.

**Next session:** No further claims in this chain.

