# FabScore: Fine-Grained Evaluation of Fabrications in Automated AI Research

This repository contains the official implementation and data of FabScore.

## Installation
Commands for ```uv``` tool.
```shell
1. uv init
2. uv venv
3. # add requirements to pyproject.toml
4. uv add requests
5. uv lock
# update packages: uv sync
```
Install ```fabscore``` as a package:
```shell
uv pip install -e .
```
Before running the following steps, ensure you have activated the virtual environment:
```shell
source .venv/bin/activate
```
You can also skip manual activation and run commands through `uv run`, which is the recommended style.

## Usage

The evaluation pipeline consists of **4 modular steps**. You can run them all at once using the main orchestrator, or individually for more control.

### Full Pipeline Run (Recommended)
Run the entire 4-step process automatically:
```shell
uv run python main.py --task_path <path_to_task_directory> --paper_filename <paper_filename_or_relative_path> --judge_type claude
```
Key optional arguments:
- `--judge_type` — Agent to use: `claude`, or `codex` (default: `claude`)
- `--model_name` — Model name override (e.g. `claude-sonnet-4-6`)
- `--extraction_only` — Stop after extraction
- `--analysis_only` — Stop after extraction + static analysis
- `--execution_only` — Stop after extraction + static analysis + execution, and skip final summarization writeout

Required arguments:
- `--task_path` — Task root directory
- `--paper_filename` — Paper filename or relative path inside the task directory, for example `paper.pdf`, `results/paper.md`, or `data_augmentation_grokking.pdf`
