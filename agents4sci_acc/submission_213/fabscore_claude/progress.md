# Progress Log

## Session 1 — 2026-04-03
**Purpose:** extraction

**Paper inspected:** 213_Sustainable_Investment_Dec.pdf
- "Sustainable Investment Decision-Making on Office Buildings using Reinforcement Learning and Large Language Models"
- 25-page PDF with main text (11 pages) + appendices

**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- **Tables:** 84 entries extracted from Tables 2, 3, 4, 5, and 6. Tables 1 and 7 were skipped (setup parameters and compute resources, not performance metrics).
  - Table 2: US case RL vs Conventional (12 metrics × 3 columns)
  - Table 3: UK case RL vs Conventional (12 metrics × 3 columns)
  - Table 4: Sensitivity analysis (5 scenarios × 2 regions)
  - Table 5: Robustness tests (3 conditions × 2 strategies)
  - Table 6: LLM evaluation scores (4 metrics)
- **Figures:** 5 figures (Figures 1–5) recorded with their captions
- **Results section claims:** 2 unique numerical values from body text not already in tables
  - US case: "Societal cost NPV decreases by $12.00M" (note: may be inconsistent with Table 2 which shows –$3.50M; possibly a different scope or paper error)
  - UK case: "Societal cost NPV decreases by £7.19M" (Table 3 shows –£3.00M; similarly possibly inconsistent)

**Next session should:**
- Proceed to analysis/scoring phase using fs_extracted.json
- Note the possible discrepancy between the $12.00M/$7.19M values in Section 3.1 body text vs. the –$3.50M/–£3.00M values in Tables 2/3 for Societal cost NPV difference

---

## Session 2 — 2026-04-03
**Purpose:** analysis

**Files/context inspected:**
- `213_Sustainable_Investment_Dec.pdf` — full paper (all 25 pages)
- `Supplementary_Material_213/train.py` — main training script
- `Supplementary_Material_213/dqn_agent.py` — DQN agent with QNetwork
- `Supplementary_Material_213/rl_env.py` — RL environment
- `Supplementary_Material_213/sensitivity.py` — sensitivity analysis script
- `Supplementary_Material_213/robustness.py` — robustness testing script
- `Supplementary_Material_213/llm_eval.py` — LLM evaluation script
- `Supplementary_Material_213/results/llm_eval_scores.csv` — LLM scores artifact
- `Supplementary_Material_213/llm_eval_outputs.csv` / `llm_eval_reference.csv`
- `Supplementary_Material_213/data_sources.csv`
- `Supplementary_Material_213/results/compute_runtimes_times.json`
- `Supplementary_Material_213/results/README.txt`

**Key finding — CRITICAL BUG in greedy evaluation:**
All three evaluation scripts (train.py, sensitivity.py, robustness.py) share a broken greedy action selection:
```python
a = np.argmax(agent.q[_target].forward.__self__.net[-1].weight.detach().numpy() @ s + ...)
```
This accesses only the LAST layer (`net[-1]` = `nn.Linear(64, action_dim)`), whose weight has shape `(3, 64)`. The state vector `s` has shape `(14,)`. The matmul `(3,64) @ (14,)` raises a `ValueError` (dimension mismatch) at runtime. The `np.errstate(all='ignore')` context suppresses float errors only, not shape errors. This means the greedy policy evaluation CRASHES and no trajectory/sensitivity/robustness CSVs can ever be written by this code.

**Supporting evidence for crash (absent artifacts):**
- No `models/dqn_US.pt` or `models/dqn_UK.pt`
- No `results/US_trajectory.csv` or `results/UK_trajectory.csv`
- No `results/sensitivity_{US,UK}.csv`
- No `results/robustness_{US,UK}.csv`
- `results/compute_runtimes_times.json` shows all training timings as `null`
- `results/README.txt` confirms training was not executed

**LLM eval finding:**
- `results/llm_eval_scores.csv` exists and shows 1.000 (100%) for all metrics on all 3 questions
- Paper Table 6 claims 92%, 94%, 89%, 90%
- The metrics in llm_eval.py (exact_match, token_f1, etc.) differ from Table 6 metrics (Explanation Accuracy, Relevance, Actionability, Numerical Consistency)
- `llm_eval_outputs.csv` contains pre-populated exact matches to `llm_eval_reference.csv`, trivially guaranteeing 100% on text-matching metrics
- No code path generates the GPT-5-based 92/94/89/90% scores

**Internal inconsistency (results text vs tables):**
- Body text (Section 3.1): "Societal cost NPV decreases by $12.00M" — Table 2 shows –$3.50M
- Body text (Section 3.1): "Societal cost NPV decreases by £7.19M" — Table 3 shows –£3.00M

**JSON files created/updated:**
- `fabscore_claude/fs_analysis.json` (created)

**Classification summary (91 total claims):**
- `obvious_hallucination` / `experiment_fabrication`: 84 claims (indices 1–80, 88–91) — broken greedy eval in train.py/sensitivity.py/robustness.py; no model or trajectory artifacts
- `insufficient_evidence`: 4 claims (indices 81–84) — LLM eval scores exist but use different metrics and show 100% vs paper's 92/94/89/90%
- `no_code_files`: 3 claims (indices 85–87) — no scripts or data for Pareto frontier (Fig 1), discount factor sensitivity (Fig 2), or process diagram (Fig 3)

**Recommended next step:**
No further sessions needed for analysis. The static evidence is decisive: the greedy evaluation code is broken in all three scripts, and no execution artifacts exist. Any execution attempt would confirm the dimension mismatch crash.

