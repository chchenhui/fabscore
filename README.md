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
