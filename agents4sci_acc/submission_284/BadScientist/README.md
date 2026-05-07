# Minimal Paper Generator (Azure + Semantic Scholar)

** ALL EXPERIMENT RESULTS ARE IN https://anonymous.4open.science/r/BadScientist **

Single, self-contained pipeline:
- Input: a broad `seed_idea` (JSON) + a LaTeX conference template directory
- Output: a finished `paper.pdf` with figures generated from code/data returned by the LLM

Features
- Azure OpenAI only
- Semantic Scholar Graph API for references (optional)
- LLM generates: multiple detailed ideas, per-idea simplified experiment plans, code+data for plots, and full paper sections
- Auto-review: after PDF compile, the pipeline runs an LLM review and saves review.json

Configuration
To configure the project, copy `local_review_config.py.example` to `local_review_config.py` and fill in the required settings:
- Azure OpenAI API key, endpoint, and API version
- Optional: Semantic Scholar API key


Dependencies
Install Python deps:
```bash
conda create -n bad_scientist python=3.12
conda activate bad_scientist
pip install -r requirements.txt
```

Install LaTeX (pdfLaTeX)
- macOS (Homebrew):
```bash
brew install texlive
```
- Linux (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install texlive-full
```
The pipeline uses pdfLaTeX and BibTeX; ensure `pdflatex` and `bibtex` are on PATH.

Quick start
1) Prepare a template directory (contains template.tex). Prefer using your real conference template, e.g.:
  `examples/latex/`.
  Seed file example:
```json
{
  "seed": "ResNet improvements for image recognition"
}
```
1) Run to test:
```bash
python launch.py \
  --template-dir examples/latex \
  --seed-path examples/seed_idea.json \
  --num-ideas 3 \
  --out results
```
or with reviews
```bash
python launch.py \
  --template-dir examples/latex \
  --seed-path examples/seed_idea.json \
  --num-ideas 3 \
  --out results \
  --enable-review
```

2) Run experiments:
```bash
python run_experiments.py
```

Artifacts will be under `results/<timestamp>/`.
