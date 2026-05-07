# Vibe Programming · HumanEval Pipeline (Baseline & Spec-Driven)

A reproducible pipeline for evaluating LLM code generation on **HumanEval** with two modes:

- **Baseline:** generate code directly from original HumanEval prompts.
- **Spec-Driven (Vibe Programming):** first convert prompts to natural language requests, then run a specification-guided generation pipeline.

It supports parallel runs and logs all generations for re-evaluation.

---

## What’s Included

**Data**
- `HumanEval.jsonl` — Original HumanEval tasks (prompts / solutions / tests).
- `HumanEval_with_nl.jsonl` — HumanEval with added natural-language requests (`nl_request`) produced by `humaneval_convert.py`.

**Pipelines & Utils**
- `humaneval.py` — Utilities for HumanEval format.
- `humaneval_convert.py` — Convert code prompts → natural language (`nl_request`) for spec pipeline.
- `spec_result_generation.py` — Generate **spec-driven** solutions (defaults configured for Claude Sonnet; see below).
- `spec_result_evaluation.py` — Evaluate spec-driven generations.
- `evaluation_baseline_pipeline.py` — Run **baseline** generation + evaluation for (e.g.) GPT‑4o.
- `baseline_result_evaluation.py` — Evaluate existing baseline generations in `all_baseline_*.jsonl`.

**Results (examples already in repo)**
- `all_baseline_claude_3_5_sonnet_20241022.jsonl` — baseline generations (Claude 3.5 Sonnet 2024-10-22).
- `all_baseline_claude_3_7_sonnet_20250219.jsonl` — baseline generations (Claude 3.7 Sonnet 2025-02-19).
- `all_spec_claude_3_7_sonnet_20250219.jsonl` — spec-driven generations (Claude 3.7 Sonnet 2025-02-19).
- `all_spec_gpt_4o.jsonl` — spec-driven generations (GPT-4o).
- `results_summary_*.txt` — one-line summary: `Total / Passed / Failed`.
- `failed_*.jsonl` — failing cases for quick inspection.

---
> **Heads-up (easy to re-run):** Two artifacts are not committed — **GPT-4o baseline** and **Claude 3.5 spec**.  
> These files were originally saved but later misplaced. If you wish to verify, you can easily regenerate them:
>
> **GPT-4o baseline**:
> ```bash
> python evaluation_baseline_pipeline.py
> ```
> **Claude 3.5 spec**
> ```bash
> python spec_result_generation.py
> python spec_result_evaluation.py
> ```

---

## Requirements

- Python **3.8+**
- Python packages: `openai`  *(native Anthropic SDK not required if you’re using an OpenAI‑compatible endpoint via `base_url`)*

```bash
pip install -U openai
# optional, if you prefer .env management
pip install -U python-dotenv
```

**Environment variables**
- `OPENAI_API_KEY` — used for both OpenAI and OpenAI‑compatible endpoints (e.g., Claude via `base_url`).
- (Optional) `OPENAI_BASE_URL` — if you prefer to set the endpoint globally; otherwise the scripts have it inline.

---

## Quickstart

### A) Baseline Pipeline (e.g., GPT‑4o)

**Option 1 (one-shot, recommended):**
```bash
python evaluation_baseline_pipeline.py
```
- Uses GPT‑4o by default.
- Produces `all_baseline_<model>.jsonl`, `results_summary_baseline_<model>.txt`, and `failed_baseline_<model>.jsonl`.

**Option 2 (evaluate an existing result file):**
```bash
python baseline_result_evaluation.py
```
- Reads `all_baseline_*.jsonl` files and writes summaries + failures.

---

### B) Spec-Driven (Vibe Programming) Pipeline

1) **Convert prompts → natural language (`nl_request`)**  
   (Skip if `HumanEval_with_nl.jsonl` already exists.)
```bash
python humaneval_convert.py
```

2) **Generate spec-driven solutions**  
   `spec_result_generation.py` already sets **Claude 3.5 Sonnet** model name and **base_url**.  
   **Just set your API key** (e.g., `OPENAI_API_KEY`) and run:
```bash
python spec_result_generation.py
```
This writes `all_spec_<model>.jsonl` (e.g., `all_spec_claude_3_5_sonnet_20241022.jsonl`).

3) **Evaluate**
```bash
python spec_result_evaluation.py
```
This writes `results_summary_spec_<model>.txt` and `failed_spec_<model>.jsonl`.

---

## File Naming Conventions

- **Generations:**  
  `all_baseline_<model>.jsonl` · `all_spec_<model>.jsonl`
- **Summaries:**  
  `results_summary_baseline_<model>.txt` · `results_summary_spec_<model>.txt`
- **Failures (per-case debug):**  
  `failed_baseline_<model>.jsonl` · `failed_spec_<model>.jsonl`

Each `all_*.jsonl` line contains the full prompt, model settings, the model output, and bookkeeping fields used by the evaluators.

---

## Reproducibility Statement

- **Determinism:** All generations use `temperature = 0`. We avoid sampling to reduce variance.
- **Model & Endpoint Pinning:** We record the exact **model IDs** (e.g., `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `gpt-4o`) and, when applicable, the **`base_url`** used for OpenAI‑compatible providers.
- **Artifacts:** We publish raw generations as `all_*.jsonl`, summaries as `results_summary_*.txt`, and failure cases as `failed_*.jsonl`. Anyone can re‑evaluate the same generations with `*_result_evaluation.py` scripts to reproduce Pass@1 outcomes.
- **Environment:** Python 3.8+; `openai` Python package. 

---




## License

For research and educational use. See individual files for details.
