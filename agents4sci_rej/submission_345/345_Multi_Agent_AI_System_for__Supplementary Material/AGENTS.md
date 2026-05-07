# AGENTS.md — System-level prompts and execution protocol

This repository uses agents aligned to the PI’s requirements. Agents must adhere to these roles and use prompts from `docs/specs/SPEC.md`. Do not invent ad-hoc prompts.

## Roles

- Orchestrator (GPT-5 Deep Thinking): Owns planning, gating (G1–G5), and delegations.
- Execution Agent: Runs code, ETL, baselines, evaluation, backtesting.
- Writing Agent: Updates Overleaf-style markdown sections.
- Reviewer Agent (Perplexity Sonar): Audits methodology, flags violations of gates and stats protocol.

## Sections ownership (async updates)
- reports/overleaf/INTRODUCTION.md — Writing Agent; Reviewer comments inline.
- reports/overleaf/METHODS.md — Execution → Writing; Reviewer mandatory.
- reports/overleaf/RESULTS.md — Execution → Writing; Reviewer mandatory.

## Ground rules
1) Never break userspace (reproducibility): log seeds, config snapshots, usage and provenance artifacts.
2) Only use prompt templates from SPEC. If you need a new template, propose it by appending to SPEC under a new versioned block.
3) Claims are gated: architecture claims only after G1–G5. Use acceptance gates defined in `MASSIVE_OVERHAUL_PLAN.md`.
4) Supplemental traces go under `supplemental/*` and must link to run IDs in `results/*`.

## Execution protocol
1) Read SPEC → select the narrowest applicable template.
2) Record run intent to `results/run_provenance.json` via AuditLogger.
3) Execute task; dump intermediate artifacts to `supplemental/intermediate/<run_id>/`.
4) Reviewer Agent audits; if blocked, raise gate failure with remediation checklist.
5) Writing Agent updates Overleaf sections.

## Minimal prompts routing
- Planning → SPEC.PLANNING
- ETL/Data checks → SPEC.ETL
- Baselines/Forecast → SPEC.BASELINE, SPEC.ANALOGS, SPEC.PATIENT_FLOW
- Evaluation/Stats → SPEC.EVAL, SPEC.POWER, SPEC.COVERAGE
- Paper writing → SPEC.WRITING.INTRO/METHODS/RESULTS
- Review → SPEC.REVIEW


