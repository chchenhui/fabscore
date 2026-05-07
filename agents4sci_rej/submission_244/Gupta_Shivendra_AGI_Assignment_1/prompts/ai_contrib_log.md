AI Contribution Log (Provenance & Roles)

Project tag: entropy_pruning_transformer
Purpose: Transparent record of prompts, tools, decisions, and division of labor for Agents4Science 2025.
Anonymity: No personal names, affiliations, or emails appear in this log.

Role scale referenced in disclosure

A = Human-generated (≥95% human)

B = Mostly human, assisted by AI (>50% human)

C = Mostly AI, assisted by human (>50% AI)

D = AI-generated (≥95% AI)

Final declarations used in the paper’s AI Involvement Checklist

Hypothesis development: B

Experimental design & implementation (simulation + runner): B

Analysis/interpretation: B

Writing (paper drafts/edits): C

Master prompts and task prompts (kept in prompts/prompt.py)

SYSTEM_PROMPT: guidance for determinism, artifact logging, ethics, and anonymity.

TASK_METHOD: method section scaffolding (entropy gate, FLOPs proxy, calibration note).

TASK_CHECKLIST: to complete the AI & Paper checklists faithfully.

TASK_README: instructions to write a concise, runnable README.

RESEARCH_WORKFLOW_PROMPT: end-to-end Steps 1–10 workflow used to structure the project.

All prompt texts are included verbatim in prompts/prompt.py.

Timeline (UTC, ISO 8601)
2025-09-08T10:05Z — Research framing & scope

Prompt(s): RESEARCH_WORKFLOW_PROMPT (Steps 1–2)

Agent: AI assistant

Tools: None (planning only)

Outputs: Initial research outline, problem statement, draft contributions, dataset strategy (synthetic + SST-2 pilot).

Human actions: Accepted scope; trimmed claims to avoid overreach.

Attribution: B

2025-09-08T14:20Z — Method and notation

Prompt(s): TASK_METHOD

Agent: AI assistant

Tools: None

Outputs: Encoder → Attention-1 → Entropy gate (top-k) → Attention-2 → Pool → Classifier write-up; FLOPs proxy; calibration blurb; notation block.

Human actions: Revised symbols; added keep-rate variants and placement guidance; ensured consistency with code.

Attribution: B

2025-09-09T09:10Z — NumPy simulation and runner

Prompt(s): RESEARCH_WORKFLOW_PROMPT (Steps 4–5)

Agent: AI assistant (code scaffold), Human (integration/fixes)

Tools: Python 3.10, NumPy, Matplotlib

Outputs: implementation.py (dataset, preprocessor, attention, models, trainer, evaluator), initial experiment_runner.py.

Human actions: Seed handling, figure saving, masking correctness; validated shapes; added JSON/NPY artifacts.

Attribution: B

2025-09-10T12:40Z — Figures & plots

Prompt(s): RESEARCH_WORKFLOW_PROMPT (Step 6)

Agent: AI assistant

Tools: Matplotlib

Outputs: Loss/accuracy/AUC curves; bar charts; ROC; ablation plots.

Human actions: Verified axes, titles, DPI; ensured PNG+PDF export; cross-checked against numbers.

Attribution: B

2025-09-11T16:05Z — SST-2 pilot notebook

Prompt(s): RESEARCH_WORKFLOW_PROMPT (Step 4: optional real-data pilot)

Agent: AI assistant (scaffold), Human (execution)

Tools: PyTorch, Transformers, Datasets, scikit-learn

Outputs: experiment_sst2.ipynb; single-epoch pilot results JSON (results_sst2.json).

Human actions: Ran on GPU runtime; recorded baseline vs. pruned metrics; confirmed ~40% FLOPs proxy reduction; noted accuracy drop.

Attribution: B

2025-09-12T09:30Z — Reproducibility & metrics recomputation

Prompt(s): TASK_README

Agent: AI assistant

Tools: None (drafting text), Python (manual verification)

Outputs: README.md draft; metrics.py to recompute acc/AUC from val_true.npy & val_score_proposed.npy.

Human actions: Validated recomputed metrics; aligned paper macros with saved JSON.

Attribution: B

2025-09-13T11:15Z — Packaging & CLI improvements

Prompt(s): RESEARCH_WORKFLOW_PROMPT (Step 5) + user request for configurable runner

Agent: AI assistant (patch text), Human (implementation)

Tools: Python, argparse

Outputs: Updated experiment_runner.py with --out and --seed; timestamped default output path; requirements.txt (NumPy/Matplotlib; optional Torch stack for SST-2).

Human actions: Tested clean run; verified artifacts and plots under results/<timestamp>/.

Attribution: B

2025-09-14T07:50Z — Paper template compliance

Prompt(s): TASK_CHECKLIST

Agent: AI assistant

Tools: LaTeX (Agents4Science 2025 style)

Outputs: Guidance to migrate to agents4science_2025.sty; integrate AI Involvement & Paper Checklists; add Responsible-AI/Broader Impact and Reproducibility statements.

Human actions: Migrated template; ensured anonymity; inserted statements; removed identifiers.

Attribution: C

2025-09-15T08:40Z — Final pass & consistency checks

Prompt(s): TASK_CHECKLIST, TASK_METHOD

Agent: AI assistant

Tools: Manual LaTeX compile; JSON diff checks

Outputs: Abstract calibrated to match numbers; alignment of FLOPs/latency proxy text; placement notes for SST-2 & Simulation-vs-Real comparison.

Human actions: Final compile; verified page budget; figure paths; anonymous PDF metadata.

Attribution: C

Artifacts (who produced/edited)

Code (simulation): AI drafted; human integrated/validated (implementation.py, experiment_runner.py) — B

Metrics recomputation (metrics.py): AI drafted; human validated — B

Plots & figures: AI scripted; human checked exports & labels — B

SST-2 notebook: AI scaffolded; human executed & logged — B

Paper text (method/limitations/ethics/reproducibility): AI drafted; human edited — C

Templates & checklists integration: AI guided; human applied — C

Tools & environments

Simulation: Python ≥3.9, NumPy ≥1.24, Matplotlib ≥3.7 (CPU)

SST-2 pilot: PyTorch ≥2.3, Transformers ≥4.42, Datasets ≥2.19, scikit-learn ≥1.3 (GPU or CPU; GPU recommended)

LaTeX: Agents4Science 2025 style (agents4science_2025.sty), pdflatex (US Letter)

Random seeds: default 2025 (runner), plus per-module fixed seeds; see code comments

Determinism & provenance notes

Deterministic RNG for synthetic pipeline (NumPy default_rng); seeds surfaced in runner.

Saved artifacts: JSON histories/finals, NPY for ROC arrays, and all figures in PNG/PDF.

Recomputation: metrics.py re-derives acc/AUC from saved arrays for checksum alignment.

SST-2: non-determinism from Torch/cuDNN possible; notebook records single-epoch pilot; recommend 3-seed runs for stability (not included here).

Known limitations (logged)

SST-2 single-epoch pilot (ρ=0.75) shows accuracy drop (0.914 → 0.827) though FLOPs proxy reduces by ~40%; framed explicitly as compute–accuracy trade-off.

Minor metric differences can occur if splits/thresholding vary—metrics.py standardizes recomputation.

No multi-seed CIs in this submission; suggested for future work.

Division of labor summary (for checklist copy-over)

Hypothesis: Mostly human; AI proposed variants and refined scope (B).

Experimental design & implementation: Mostly human; AI provided scaffolds and patches (B).

Analysis: Mostly human; AI assisted with structure and clarity (B).

Writing: Mostly AI drafting with human editing for accuracy/compliance (C).

End of AI Contribution Log.