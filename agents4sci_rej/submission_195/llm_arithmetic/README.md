# LLM Arithmetic Research Framework

## Project Summary

This project implements a comprehensive research framework for studying arithmetic reasoning capabilities in Large Language Models (LLMs).

## Architecture Overview

```
llm_arithmetic/
├── src/llm_arithmetic/           # Core research framework
│   ├── evaluation/               # Phase 1: Multi-model evaluation
│   ├── models/                   # Phase 2: Fine-tuning infrastructure  
│   └── interpretability/         # Phase 3: Mechanistic analysis
├── experiments/                  # Phase-specific experiment scripts
├── data/                         # Datasets and benchmarks
├── results/                      # Analysis outputs
├── scripts/                      # Utility scripts
└── papers/                      # paper files, figures, and the script to create the figures
```

## Multi-Model Evaluation Infrastructure ✅
**Objective**: Comprehensive evaluation of LLM arithmetic capabilities

**Key Components**:
- `evaluation/base_evaluator.py` - Core evaluation framework
- `evaluation/model_evaluator.py` - Multi-provider API support (OpenAI, Anthropic, Together AI, HuggingFace)
- `evaluation/benchmark_generator.py` - MATH 401+ benchmark generation
- `experiments/phase1/run_evaluation.py` - Evaluation experiment script

**Features**:
- Support for 4 major LLM providers
- Operation-specific accuracy analysis (addition, subtraction, multiplication, division)
- Difficulty-based performance measurement
- Confidence calibration and uncertainty quantification
- Automated benchmark generation with 400+ diverse problems

### Get started


```bash
pip install -e .

python experiments/phase1/run_evaluation.py \
  --models "openai/gpt-4" "anthropic/claude-3-5-sonnet-20241022" \
  --benchmark-size 100
```

To reproduce the full results in the paper, please refer to the ```research_progress.md```. The raw outputs can be found in ```results/phase1/math401```.