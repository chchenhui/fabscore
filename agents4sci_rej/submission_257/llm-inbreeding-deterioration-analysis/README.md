# llm-inbreeding-deterioration-analysis

An investigation into the potential deterioration of language model capabilities when training new models exclusively on outputs from existing LLMs, examining the parallel to biological inbreeding depression.

This research project was created with [Co-Sci](https://co-sci.org)

## Project Structure

- `paper_drafts/` - Draft versions of research papers
- `grant_proposals/` - Grant proposal documents  
- `paper_template/` - LaTeX template files for academic papers
- `hypothesis.jsonl` - Tracked assumptions and hypotheses
- `paper.jsonl` - Literature review papers and references
- `proposal.jsonl` - Experiment proposals and evaluation plans
- `run.jsonl` - Experiment runs and execution tracking
- `analysis.jsonl` - Experiment analyses and findings
- `experiments/` - Organized experiment folders
  - Each experiment contains: `plan.md`, `result.md`, `run/`, `analysis/`
- `related_work/` - Literature review notes and paper summaries
- `section_notes/` - Working notes for paper sections
  - `01-research-concept-direction.md` - Research concept and hypotheses
  - `02-lit-review.md` - Literature review notes
  - `03-experiment-ideas.md` - Experiment proposals
  - `04-datasets.md` - Dataset documentation
  - `05-experiment-runs.md` - Experiment execution logs
  - `06-experiment-analyses.md` - Analysis and findings
- `data/` - Research data (max 2GB per file, use Git LFS for larger)
- `notes/` - Informal notes, thoughts, and documentation
  - `meeting_notes/` - Meeting notes and collaborative discussions
  - `code/` - Code snippets, scripts, and development notes
- `.github/workflows/` - Automated workflows including Claude AI assistant
- `CLAUDE.md` - AI research assistant instructions

## Getting Started

1. Start by documenting your assumption and hypothesis in `hypothesis.jsonl`
2. Add papers to `paper.jsonl` and conduct literature review in `related_work/`
3. Propose experiments in `proposal.jsonl`
4. Execute experiments and track in `run.jsonl`
5. Analyze results in `analysis.jsonl`
6. Draft your paper in `paper_drafts/`

## Research Methodology

This project follows standard scientific research methodology. See `CLAUDE.md` for detailed guidance.
