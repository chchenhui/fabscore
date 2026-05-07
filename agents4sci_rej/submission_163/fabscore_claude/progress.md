# Progress Log

## Session 1 — 2026-04-24

**Purpose:** extraction

**Paper inspected:** `163_Beyond_Game_Theory_Optimal.pdf` — "Beyond Game Theory Optimal: Profit-Maximizing Poker Agents for No-Limit Hold'em"

**Summary:** The paper benchmarks CFR, MCCFR, DeepCFR, NFSP, and a Random baseline against a GTO proxy in heads-up and multiway (k=3–6) NLHE poker. Results are reported in Table 2 (heads-up, 500 iters) and Table 3 (multiway k=3–6). One figure (Figure 1) shows Beta equity distributions. No numerical performance claims in the results text body were found that were not already captured in the tables.

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` — created; contains tables (150 entries from Table 2 and Table 3), figures (1 entry for Figure 1), and results_section (empty, all numerical claims are duplicates of table entries).

**Next session should:**
- Proceed with analysis/scoring of the extracted results against paper claims.

---

## Session 2 — 2026-04-24

**Purpose:** analysis

**Files inspected:**
- `poker_benchmark/cfr.py` — CFR and MCCFRExternal implementation
- `poker_benchmark/deep_cfr.py` — DeepCFR implementation
- `poker_benchmark/nfsp.py` — NFSP implementation
- `poker_benchmark/multiway_scoreboard.py` — Table 3 evaluation script
- `poker_benchmark/NO_1.py` — Copy of multiway_scoreboard.py with reduced params (10 iters)
- `poker_benchmark/run_baselines.py` — Demo script
- `poker_benchmark/eval.py` — Stub returning hardcoded 10.0
- `163_Beyond_Game_Theory_Optimal.pdf` (via exploration agent)

**Key findings:**
1. **CFR** (`cfr.py:42`): `traverse()` uses `self.rng.uniform(-1, 1)` random pseudo-advantages with comment "Dummy traversal to illustrate structure". Not real CFR.
2. **MCCFRExternal** (`cfr.py:70`): `class MCCFRExternal(CFRTabular): pass` — literally identical to the broken CFR stub.
3. **DeepCFR** (`deep_cfr.py:79`): `collect_traversals()` uses `random.uniform(0,1)` as placeholder targets; `train_step_mse()` has explicit `pass` — no gradient updates.
4. **NFSP** (`nfsp.py`): Has no `train()` method. `train_nfsp()` in multiway_scoreboard.py checks `hasattr(n,'train')` which is False — NFSP is never trained.
5. **No Table 2 evaluation script** exists. `multiway_scoreboard.py` only covers k=3..6.
6. **No pre-computed result files** (no CSV/JSON/NPY with paper metrics).
7. **Figure 1**: No code generates Beta distribution plots; no figure files in repo.

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` — created; 150 claims classified as `obvious_hallucination/experiment_fabrication`, 1 claim (Figure 1) as `no_code_files`.

**Classification summary:**
- `obvious_hallucination` (experiment_fabrication): 150 claims (all table claims, indices 1–150)
- `no_code_files`: 1 claim (Figure 1, index 151)

**Recommended next step:**
- No execution needed — static analysis is conclusive. All algorithm implementations are explicitly stubs/placeholders. The paper's claimed numerical results cannot originate from this repository's code.

