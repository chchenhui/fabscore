# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent research repository focused on making novel scientific discoveries through TCGA (The Cancer Genome Atlas) pan-cancer analysis using AI modeling. The goal is to produce a high-quality scientific paper that meets NeurIPS submission standards and can pass both LLM-based and human reviewers.

## Key Development Commands

### Environment Management
- **R analysis**: `conda activate r`
- **Python analysis**: `conda activate llm` or `conda activate scanpy` 
- **Biomedical tools**: `conda activate biomni`

Note: Conda environments are mentioned in env_context.md but may need to be set up if not available.

### LaTeX Paper Compilation
- Navigate to `Agents4Science_Template/` directory
- Use standard LaTeX commands: `pdflatex agents4science_2025.tex`
- Template follows Agents4Science 2025 conference format

## Repository Architecture

### Core Directory Structure
- **`data/`**: TCGA datasets
  - `RNAseq_data/`: Cancer type-specific RNA-seq data files (*.rds format)
  - `clinical_data/`: ALL_Cancer_clinical.rds with clinical metadata
- **`code/`**: Analysis scripts and pipelines (currently empty, to be populated)
- **`manuscript/`**: LaTeX manuscript files (currently empty)
- **`figures/`**: Generated plots and scientific visualizations
- **`results/`**: Processed outputs, analysis results, organized in sub-folders
- **`log/`**: Experiment logs and reasoning steps for reproducibility
- **`reference/`**: Research papers and literature (includes Biomni reference)

### Template and Requirements
- **`Agents4Science_Template/`**: LaTeX template with .sty and .tex files
- **`conference_requirement/`**: Conference submission guidelines
- **`NeurlPS_requirement/`**: Reviewer and ethics guidelines

### Multi-Agent System
The repository uses specialized agents defined in `.claude/agents/`:
- **research-principal-investigator**: Strategic guidance, novel idea generation and evaluation
- **bioinformatics-tcga-analyst**: TCGA data analysis and machine learning modeling
- **neurips-reviewer**: Simulates NeurIPS peer review process
- **latex-paper-writer**: Manuscript drafting and editing
- **scientific-visualization**: High-quality figure generation

## Data Analysis Workflow

### TCGA Data Structure
- RNA-seq data organized by cancer type (ACC, BLCA, BRCA, etc.)
- Clinical data consolidated in single file
- Data format: R data files (.rds) requiring R/Bioconductor for analysis

### Research Focus Areas
- Novel AI methodologies applied to pan-cancer analysis
- Emphasis on methods not previously applied to TCGA data
- Multi-omics integration opportunities
- Cancer subtype classification and biomarker discovery

## Key Principles

### Reproducibility Requirements
- Document all analysis steps in `log/` directory
- Avoid black-box reasoning approaches
- Maintain clear audit trail of methods and decisions
- Store all code in version-controlled scripts

### Quality Standards
- Follow NeurIPS submission standards strictly
- Prioritize scientific novelty and ethical compliance
- Ensure statistical rigor in all analyses
- Create publication-quality visualizations

### Claude Configuration
- Conda commands are permitted via `.claude/settings.local.json`
- Five specialized agents available for different aspects of research
- MCP servers available for literature search and biomedical knowledge

## Development Notes

- Python and R environments available via conda
- No existing code files - project in initial setup phase
- LaTeX template configured for Agents4Science 2025 conference
- Emphasis on novel methodology application to TCGA datasets