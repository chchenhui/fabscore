# Entropy-Guided Token Pruning — Reproducibility Package

This repo contains:
- A **NumPy-only synthetic pipeline** (deterministic, lightweight) that generates sequences, runs baselines vs. entropy-gated pruning, and saves **JSON logs + figures**.
- An **SST-2 pilot** (PyTorch/HF) notebook that applies the pruning idea to DistilBERT for one-epoch sanity checks.

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
