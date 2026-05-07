# Standard Operating Procedure: Autonomous Survey Generation

## Overview
This SOP defines the workflow for generating academic literature surveys using Claude Code agents and Task Master.

## Prerequisites
- Claude Code with configured agents
- Task Master initialized
- API keys configured (.env file)
- Python environment with dependencies

## Workflow Steps

### Step 1: Initialize Survey Project
```bash
# Create project directory
mkdir surveys/{KEYWORD_SLUG}
cd surveys/{KEYWORD_SLUG}

# Copy PRD template
cp ../../templates/SURVEY_PRD_TEMPLATE.md PRD.md
# Replace {KEYWORD} with actual keyword in PRD.md

# Initialize Task Master
task-master init
task-master parse-prd PRD.md
```

### Step 2: Execute Paper Search Phase
```bash
# Expand search tasks
task-master expand --id=1

# Launch paper search agent
task-master next
# Agent: paper-search-specialist will execute
# Provide keyword and parameters when prompted
# Wait for completion

# Mark task complete
task-master set-status --id=1 --status=done
```

### Step 3: Execute Clustering Phase
```bash
# Start clustering task
task-master next

# Launch clustering agent
# Agent: topic-mining-clustering will execute
# Provide papers.json path when prompted
# Wait for clustering completion

# Mark task complete
task-master set-status --id=2 --status=done
```

### Step 4: Execute Survey Writing Phase
```bash
# Start writing task
task-master next

# Launch survey writer agent
# Agent: academic-survey-writer will execute
# Provide clusters.json path when prompted
# Wait for survey generation

# Mark task complete
task-master set-status --id=3 --status=done
```

### Step 5: Execute Evaluation Phase
```bash
# Start evaluation task
task-master next

# Launch evaluator agent
# Agent: survey-quality-evaluator will execute
# Provide survey.md path when prompted
# Review quality report

# Mark task complete
task-master set-status --id=4 --status=done
```

### Step 6: Review and Iterate
```bash
# Check overall progress
task-master list

# Review outputs
ls -la output/

# If quality < 7.0, iterate:
task-master add-task --prompt="Improve survey based on evaluation feedback"
```

## Agent Prompts

### For paper-search-specialist:
```
Search for papers on "{KEYWORD}" using both Semantic Scholar and arXiv APIs. 
Target: 100-200 papers
Filters: Last 5 years, minimum 5 citations for older papers
Output: Save to data/papers/papers.json
```

### For topic-mining-clustering:
```
Analyze papers from data/papers/papers.json
Identify 5-10 main research themes
Use embedding-based clustering
Output: Save clusters to data/clusters/clusters.json
```

### For academic-survey-writer:
```
Generate comprehensive survey using:
- Papers: data/papers/papers.json
- Clusters: data/clusters/clusters.json
Style: Academic, 5000-8000 words
Output: Save to output/survey.md
```

### For survey-quality-evaluator:
```
Evaluate survey from output/survey.md against papers in data/papers/papers.json
Assess: Coverage, accuracy, organization, synthesis, readability
Output: Save evaluation to output/evaluation.json
```

## Quality Checkpoints

### After Paper Search:
- [ ] Minimum 50 papers collected
- [ ] Both sources returned results
- [ ] Deduplication successful
- [ ] Papers have abstracts

### After Clustering:
- [ ] 5-10 clusters created
- [ ] All papers assigned
- [ ] Cluster names meaningful
- [ ] Outliers identified

### After Survey Generation:
- [ ] All sections present
- [ ] 5000+ words
- [ ] Citations included
- [ ] Logical flow

### After Evaluation:
- [ ] Overall score ≥ 7.0
- [ ] No major gaps identified
- [ ] Improvements documented
- [ ] Ready for review

## Troubleshooting

### Issue: API Rate Limits
- Solution: Add delays between requests
- Use cached results when available

### Issue: Low Paper Count
- Solution: Broaden search terms
- Remove year restrictions
- Lower citation threshold

### Issue: Poor Clustering
- Solution: Adjust cluster count
- Review outliers
- Improve embeddings

### Issue: Low Quality Score
- Solution: Review evaluation feedback
- Add missing citations
- Improve synthesis sections
- Enhance transitions

## Output Structure
```
surveys/{KEYWORD_SLUG}/
├── PRD.md                    # Product requirements
├── .taskmaster/             # Task management
│   └── tasks/
│       └── tasks.json
├── data/
│   ├── papers/
│   │   └── papers.json     # Retrieved papers
│   └── clusters/
│       └── clusters.json   # Topic clusters
└── output/
    ├── survey.md           # Final survey
    ├── evaluation.json     # Quality assessment
    └── summary.json        # Pipeline summary
```

## Notes
- Each survey runs in its own Claude Code session
- Agents handle the heavy lifting autonomously
- Task Master tracks progress and dependencies
- Human oversight at key checkpoints
- Iterative improvement based on evaluation