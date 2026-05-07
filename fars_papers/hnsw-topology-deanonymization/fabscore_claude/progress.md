# Progress Log

## Session: 2026-04-24
- **Purpose**: extraction
- **Paper inspected**: `paper.pdf` — "Auditing HNSW Index Leakage: Recovering Embedding Geometry from Graph Topology" (Analemma / FARS)
- **Files created/updated**:
  - `fabscore_claude/fs_extracted.json` — created; contains tables (51 entries from Tables 1–3), figures (1 entry: Figure 1), and results_section (7 entries of derived/relative metrics not already captured in tables)
- **Summary of extraction**:
  - Table 1: Main attack results (Recall@10 and Δ) for Adjacency-Only, Unweighted Geodesic, and Degree-Penalized across SIFT10K (128-d) and MSMARCO-10K (768-d)
  - Table 2: Hyperparameter sensitivity on SIFT10K — alpha sweep (α = 0.0–4.0) and landmark sweep (L = 64–1024), with Recall@10, Spearman, and Time columns
  - Table 3: Sanity check comparing HNSW vs. ER Random graph vs. Chance Level on Avg Degree, Adj-Only R@10, Deg-Pen R@10
  - Figure 1: Pipeline overview diagram
  - Results section: relative improvements (28%, 52–67%, 94%, 93%, 20%), standard deviation (<0.003), and fold-change comparisons (1.3×, 1.1×, 269×) not present in tables
- **Next session**: No further extraction needed for this paper. Could do analysis/scoring (fs_analysis.json) if required by the pipeline.

## Session: 2026-04-24 (Analysis)
- **Purpose**: analysis
- **Files/context inspected**:
  - `paper.pdf` (via prior extraction)
  - `exp/EXPERIMENT_RESULTS/sift10k_adjacency_only/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/sift10k_unweighted_geodesic/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/sift10k_degree_penalized/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/msmarco10k_adjacency_only/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/msmarco10k_unweighted_geodesic/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/msmarco10k_degree_penalized/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/hyperparameter_ablation/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/sanity_checks/RESULTS.json`
  - `exp/EXPERIMENT_RESULTS/effectiveness_evaluation_result.json`
  - Repository structure (all Python scripts and pipeline code)
- **JSON files created/updated**:
  - `fabscore_claude/fs_analysis.json` — created; all 59 claims classified as `static_verifiable`
- **Summary of classifications**:
  - All 59 claims classified as `static_verifiable`. Every Table 1, Table 2, Table 3, and Results Section numerical claim matches exactly (within rounding) the corresponding stored RESULTS.json files in `exp/EXPERIMENT_RESULTS/`. Figure 1 is a qualitative pipeline diagram whose described methodology is fully implemented in the code.
  - Key matches verified:
    - Table 1: All 10 entries match `sift10k_*/RESULTS.json` and `msmarco10k_*/RESULTS.json`
    - Table 2: All 33 entries match `hyperparameter_ablation/RESULTS.json`
    - Table 3: All 8 entries match `sanity_checks/RESULTS.json` and cross-referenced result files
    - Results section: All 7 derived/relative claims verified by arithmetic on stored values
  - No hallucinations, fabrications, or missing code paths detected.
- **Recommended next step**: No further analysis needed. All claims are fully supported by static artifacts.
