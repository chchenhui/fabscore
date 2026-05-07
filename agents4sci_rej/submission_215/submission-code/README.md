# Agentic Auto-Survey: Conference Submission Code

## Overview

This directory contains the essential code and results for the "Agentic Auto-Survey" research project comparing different approaches to automated literature survey generation.

## Directory Structure

```
submission-code/
├── src/                    # Core Python source code (26 files)
├── results/                # Generated surveys and comparisons
│   ├── multi-agent/       # Multi-agent system results
│   ├── autosurvey/        # AutoSurvey baseline results
│   └── COMPARISON_SUMMARY.md  # Performance comparison
├── evaluation/            # Evaluation metrics and scores
├── templates/             # System prompts and templates
├── scripts/               # Additional utility scripts
├── docs/                  # Documentation and manuscript
├── claude-code-setup/     # Claude Code configuration for reproduction
│   └── agents/           # Subagent prompts for multi-agent system
├── baseline/             # Baseline systems for comparison
│   └── auto-survey/      # Modified AutoSurvey baseline system
└── data/                  # Data directory (placeholder)
```

## Key Components

### Source Code (`src/`)
- **enhanced_evaluator.py** - Advanced survey evaluation system
- **batch_enhanced_eval.py** - Batch evaluation processing
- **batch_survey_generator.py** - Multi-agent survey generation
- **advanced_clustering.py** - Paper clustering algorithms
- **search_*.py** - Paper search and retrieval scripts
- **cluster_*.py** - Domain-specific clustering implementations

### Results (`results/`)
- **Multi-Agent System**: High-quality surveys handling 100-1,334 papers
- **AutoSurvey Baseline**: Comparison baseline results
- **COMPARISON_SUMMARY.md**: Comprehensive performance analysis

### Evaluation (`evaluation/`)
- Comprehensive evaluation scores and metrics
- Multi-dimensional assessment results
- System performance comparisons

## Key Findings

| Method | Average Score | Papers Processed | Grade |
|--------|---------------|------------------|-------|
| **Multi-Agent System** | 8.17 | 100-1,334 | A- |
| **AutoSurvey** | 4.77 | 75-100 | C |

## Reproducing Results

### 1. Environment Setup
- Copy `.env.example` to `.env` and configure API keys
- Install required Python dependencies

### 2. Claude Code Setup (For Multi-Agent System)
**Important**: To reproduce the multi-agent results, you need to set up Claude Code subagents:

1. Copy the agents from `claude-code-setup/agents/` to your `.claude/agents/` directory:
   ```bash
   cp -r claude-code-setup/agents/ .claude/
   ```

2. The subagent prompts include:
   - `academic-survey-writer.md` - Main survey writing agent
   - `enhanced-survey-evaluator.md` - Survey evaluation agent  
   - `paper-search-specialist.md` - Paper search and retrieval agent
   - `survey-quality-evaluator.md` - Quality assessment agent
   - `topic-mining-clustering.md` - Paper clustering agent

### 3. Running the System
- **Multi-Agent Survey Generation**: Use `batch_survey_generator.py` 
- **Survey Evaluation**: Use `enhanced_evaluator.py` for comprehensive assessment
- **Results Processing**: Use evaluation scripts to extract and analyze scores

### 4. Running AutoSurvey Baseline (Optional)
To reproduce the baseline comparison:
- Navigate to `baseline/auto-survey/`
- Use `run_local_survey_robust.py` for improved reliability
- See `MODIFICATIONS.md` for details on our enhancements

### 5. Expected Outputs
The system should generate surveys in the same format as found in `results/multi-agent/` with similar quality scores (8.17/10 average).

## Templates

The `templates/` directory contains:
- Agent prompts for multi-agent system
- Survey generation SOPs
- Evaluation criteria and rubrics

## Contact

For questions about this submission code, please refer to the manuscript draft in `docs/manuscript_draft.tex`.

---

*Generated for conference submission - excludes single-agent baseline per requirements*