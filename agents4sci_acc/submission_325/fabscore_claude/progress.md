# Progress Log

## Session 1 — 2026-04-03
**Purpose:** extraction
**Paper inspected:** 325_Scalable_Oversight_in_Mult.pdf ("Scalable Oversight in Multi-Agent Systems: Provable Alignment via Delegated Debate and Hierarchical Verification")
**JSON files created/updated:**
- `fabscore_claude/fs_extracted.json` (created)

**Summary of extraction:**
- Tables: 39 entries extracted from Table 1 (WebArena results), Table 2 (AgentBench highlights), and Table 4 (Error taxonomy on WebArena). Table 3 (hyperparameters) was excluded as it contains setup values, not experimental results.
- Figures: 2 entries — Figure 1 (HDO architecture diagram) and Figure 2 (Pareto frontier of oversight risk vs. cost).
- Results section: 3 textual claims — (1) 28% hallucination reduction vs. No Oversight, (2) oversight accuracy at approximately half the tokens of human oversight, (3) HDO detects 8/10 collusion attempts vs. flat debate's 3/10.

**Next session should:** verify extracted values against paper, check for any missed subsection text claims in Section 6.3 ablations, and perform scoring/analysis if required.

## Session 2 — 2026-04-03
**Purpose:** analysis (static code audit)

**Files inspected:**
- `325_Scalable_Oversight_in_Mult.pdf` (full paper, 12 pages)
- `325_Scalable_Oversight_in_Mult_Supplementary Material/outputs/hdo_evaluation_report.txt`
- `325_Scalable_Oversight_in_Mult_Supplementary Material/outputs/hdo_demo_results.json`
- `325_Scalable_Oversight_in_Mult_Supplementary Material/submission_metadata.json`
- `325_Scalable_Oversight_in_Mult_Supplementary Material/code/nsrag/hdo/evaluation.py` (first 100 lines)
- Repository structure via Explore agent

**Key findings:**
1. The repository is a **simulation prototype** — all 6 demo episodes have `total_cost=0.0`, `delegation_depth=0`, and `collusion_detections_count=0`. No actual LLM calls are made.
2. The paper claims experiments on **WebArena (50 tasks)** and **AgentBench** benchmarks, but neither environment nor task data is present in the repository.
3. The `submission_metadata.json` explicitly acknowledges discrepancies: "oversight_accuracy: 76.7% (claimed 95%, limited by mock verifiers)" and "token_efficiency: Would achieve with real verifier integration".
4. The `hdo_evaluation_report.txt` reports oversight_accuracy=76.7% (vs claimed 95%), collusion_detection_rate=0.000 (vs claimed 80%), token_usage=0 (vs claimed 7,500).
5. The paper checklist says "Experimental reproducibility: [No]" and "Open access to data/code: [No]".
6. Figure 1's architectural description is well-supported by the code (hdo_system.py, verifiers.py, routing.py, aggregation.py).
7. Figure 2 (Pareto frontier) has no underlying data, plotting script, or data file anywhere in the repository.

**JSON files created:**
- `fabscore_claude/fs_analysis.json` (created)

**Classification summary (44 total claims):**
- `static_verifiable`: 1 (claim 40 — Figure 1 architecture directly implemented in code)
- `no_code_files`: 1 (claim 41 — Figure 2 Pareto frontier, no data or plot code)
- `insufficient_evidence`: 42 (claims 1–39, 42–44 — all empirical/numerical claims: no WebArena/AgentBench experiments in repo, only mock simulations with 0 token usage; execution artifacts exist but are from unrelated simulations; paper explicitly declares no reproducibility)

**Next step:** None required; analysis is complete. If execution is attempted in a future session, the first step should be investigating whether WebArena/AgentBench environments can be set up and whether the HDO code can be connected to real verifiers rather than mock ones.
