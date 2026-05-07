# Reproducibility Statement

**Scope.** Our results are computed on the **400 ARC public evaluation tasks** by aggregating 16 diverse models and methods with an **automatic verifier** (code execution on training pairs) and counting a task as solved if *any* method returns a verified solution. The same aggregation logic is used throughout.

**Data & Splits.** We use the standard ARC public training (400) and public evaluation (400) tasks and do not alter splits. No private tasks are used during development.

**Code & Artifacts.** We release the method drivers in `code/` and the modular implementations in `code/optillm/`, as well as runnable notebooks for BARC and MARC. The package accepts ARC JSONs and writes `submission.json` files. We request reviewers to inspect (or drop in) **intermediate outputs**—JSONL traces, candidate programs, and per-method verifier logs—under `intermediate_outputs/`.

**Verifier.** For each candidate program/output we **execute on training examples** and only keep solutions that exactly match the provided outputs. This near-perfect verifier minimizes false positives and enables consistent aggregation across methods.

**Environment.** We provide a `requirements.txt` with pinned minimum versions. Experiments were run on Linux with Python ≥3.10 and PyTorch ≥2.2; open-model baselines use `transformers/peft/bitsandbytes`. Some experiments call closed providers (OpenAI/Anthropic/Gemini); reproducibility for these follows path **A (open-model)** or **B (full)** in the top-level `README.md`.

**Compute.** The diverse-inference runs are parallel but modest in wall-clock time (each method typically ≤10 minutes on 1×A100/RTX4090, depending on sampling budget). We include method-level timing in our logs where available.

**Determinism.** We set seeds for `random`, `numpy`, and `torch` and recommend enabling CUDA deterministic flags. Stochasticity from remote providers (temperature, sampling) is reported with seeds/configs in logs when applicable.

**Closed-Model Access.** Some numbers in the paper require access to commercial APIs (e.g., o1/o3). Where access is unavailable, our open-model path replicates the *relative* improvements and the aggregation pipeline; absolute numbers may differ.

**Ethics & Licensing.** We use a public benchmark, release only anonymized artifacts, and avoid shipping any proprietary data. All added code is under a permissive license for review purposes.

These steps are intended to provide **sufficient information to reproduce our comparisons and to audit the verifier-driven aggregation** even without commercial API access.
