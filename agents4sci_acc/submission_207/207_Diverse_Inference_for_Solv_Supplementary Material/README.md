# Supplementary Materials — Diverse Inference for Solving ARC at a Human Level

**Anonymous submission to Agents4Science 2025.** This archive contains code, notebooks, and (optionally) intermediate outputs supporting the paper.

---

## Directory Layout

```
.
├── code/
│   ├── inference_engine.py          # unified LLM wrappers
│   ├── best_of_n.py                 # Best-of-N sampling driver
│   ├── arc_zero_shot.py                 # Zero-shot baseline
│   ├── arc_predict_submission.py        # Utility to build ARC submission.json (uses arclib if installed)
│   ├── arc_prompts.py                   # Build OptiLLM input JSON from task_prompts/*.txt
│   ├── arc_query_examples.py            # Export ARC JSON tasks to readable prompt .txt files
│   ├── optillm_runner.py                # Optillm package wrapper
│   └── optillm/                         # Modular methods used in the paper
│       ├── mcts.py, moa.py, plansearch.py, rto.py, rstar.py, ...
│       └── (and other helpers: cot_decoding.py, litellm_wrapper.py, etc.)
├── docs/
│   ├── REPRODUCIBILITY.md               # Reproducibility Statement
│   └── LICENSE_AND_ANONYMITY.md         # Anonymity checklist
├── intermediate_outputs/                # (Optional) put logs/candidates/verifier traces here
│   └── README.md
├── notebooks/
│   ├── run_barc.ipynb
│   ├── run_marc.ipynb
│   └── visualize_arc_problems.ipynb
└── requirements.txt
```

## Getting Started

1. **Python**: 3.10 or newer is recommended.
2. **Install**:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # or use conda
   pip install -r requirements.txt
   ```
   *Some packages are optional and platform-specific (e.g., `flash-attn`, `xformers`).*
3. **Provider credentials (if using closed models)**: set the following if you plan to call hosted models.
   ```bash
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   export GEMINI_API_KEY=...
   ```

## Data: ARC Benchmark

The ARC JSON files are public. Download the **ARC public training/evaluation tasks** and point scripts to the local path.

> Example sources (mirror any official source you prefer):
> ```
> https://github.com/fchollet/ARC
> https://github.com/aborruso/arc-dataset
> ```

## Reproducing Key Pipelines (best-effort)

Because our approach aggregates **diverse models and methods** with a **verifier**, there are two practical paths for reproduction:

### A. Open-Model–Only Reproduction (no paid API keys)

This replicates the *structure* of the evaluation with open checkpoints. It will not exactly match closed-model numbers but preserves the method comparisons and the aggregation logic.
- Use the modules in `code/optillm/` (e.g., `mcts.py`, `moa.py`, `plansearch.py`, `self_consistency.py`).
- Run BARC/MARC notebooks in `notebooks/` to reconstruct induction/transduction and test-time training.
- Generate `submission.json` with:
  ```bash
  python code/arc_predict_submission.py     --data_file path/to/arc_public_evaluation.json     --experiment_folder ./runs/open_model_eval     --solution_file path/to/arc_public_solutions.json  # optional: for local eval only
  ```

### B. Full Diverse-Inference Reproduction (requires model API access)

Set credentials as above, then use `code/arc_inference_engine.py` to call the specific providers (OpenAI/Gemini/etc.).
We recommend writing slate scripts that enqueue multiple methods and then **aggregate via the verifier** (see `docs/REPRODUCIBILITY.md`).

**Determinism**: set these environment variables or call the standard seeds in your launcher:
```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:16:8  # if using CUDA determinism
```
And in Python:
```python
import random, numpy as np, torch
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
```

## Intermediate Outputs

Place any logs (e.g., JSONL traces from `optillm/conversation_logger.py`), candidate programs, verifier traces, and per-method `submission.json` files in `intermediate_outputs/` before uploading your final zip. These artifacts enable the committee to **replay** our verification and aggregation.

## Anonymity Notes

- This package contains **no author names** and no institutional identifiers.

