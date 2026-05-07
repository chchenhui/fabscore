# Improved Standard Operating Procedure: Autonomous Survey Generation v2.0

## Lessons Learned & Best Practices

### Key Success Factors
1. **Use uv for environment management** - Much faster than pip
2. **B200 GPU optimization** - Use CUDA-enabled PyTorch for embeddings
3. **Real API usage** - Always use actual APIs, not mock data
4. **Agent specialization** - Each agent should focus on one task
5. **Progressive validation** - Check outputs at each stage

## Pre-Setup Requirements

### Environment Setup (CRITICAL)
```bash
# ALWAYS use uv for fast dependency management
cd survey_project
uv venv
source .venv/bin/activate

# For B200 GPU support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install -r requirements.txt
```

### API Configuration
```bash
# Ensure .env file has API keys
cat > .env << EOF
ANTHROPIC_API_KEY=your_key_here
# Semantic Scholar works without key but has lower rate limits
SEMANTIC_SCHOLAR_API_KEY=optional_key_here
EOF
```

## Optimized Workflow

### Phase 1: Paper Search (paper-search-specialist)

#### Inputs
- **Keyword**: Be specific (e.g., "LLM agents" not just "agents")
- **Target count**: 50-100 papers optimal (more causes clustering issues)
- **Time range**: Last 3 years provides good balance

#### Key Instructions for Agent
```markdown
CRITICAL: Use the actual API clients from scripts/core/api_clients.py
- Use BOTH Semantic Scholar AND arXiv for comprehensive coverage
- Generate 20-30 diverse search queries for better coverage
- Apply quality filters AFTER retrieval, not during search
- Save raw results before filtering for transparency
```

#### Quality Checks
- [ ] Minimum 50 papers collected
- [ ] Both sources contributed papers
- [ ] Papers have abstracts (required for clustering)
- [ ] Deduplication successful (check for title variations)
- [ ] Save search report with statistics

#### Common Issues & Solutions
- **Rate limiting**: Add exponential backoff, use cached results
- **Low paper count**: Broaden search terms, include synonyms
- **Missing abstracts**: Filter these out early, they break clustering

### Phase 2: Clustering (topic-mining-clustering)

#### Inputs
- **Papers JSON**: Must have title AND abstract for each paper
- **Target clusters**: 5-10 optimal (7-8 works best)

#### Key Instructions for Agent
```markdown
CRITICAL: Use actual embedding generation with GPU support
- Load embedding model to GPU if available (check torch.cuda.is_available())
- Use title + abstract concatenated for better embeddings
- Let clustering algorithm optimize K (don't force cluster count)
- Name clusters based on actual paper content, not generic terms
```

#### Quality Checks
- [ ] All papers assigned to clusters
- [ ] Cluster sizes relatively balanced (no cluster with >40% of papers)
- [ ] Cluster names are specific and descriptive
- [ ] Outliers identified (<5% expected)
- [ ] Save clustering visualization if possible

#### Common Issues & Solutions
- **Memory issues**: Process embeddings in batches of 32
- **Poor cluster quality**: Try different embedding models
- **Generic cluster names**: Use TF-IDF on abstracts for better names

### Phase 3: Survey Writing (academic-survey-writer)

#### Inputs
- **Papers + Clusters**: Both files required
- **Target length**: 5000-8000 words optimal

#### Key Instructions for Agent
```markdown
CRITICAL: Focus on synthesis, not listing
- Cite at least 50% of papers (major weakness if <30%)
- Each cluster section should synthesize 3-5 key papers minimum
- Use [Author, Year] format consistently
- Include specific findings and numbers from papers
- Create comparison tables where appropriate
```

#### Structure Requirements
1. **Abstract**: 200-250 words, mention paper count and time range
2. **Introduction**: Set context, state contributions, outline structure
3. **Background**: Define key terms, theoretical foundations
4. **Cluster Sections**: One per cluster, with synthesis not lists
5. **Trends**: Temporal analysis, emerging patterns
6. **Future Directions**: Based on identified gaps
7. **Conclusion**: Summarize key findings
8. **References**: All cited papers in consistent format

#### Quality Checks
- [ ] Word count in target range
- [ ] All clusters have dedicated sections
- [ ] >50% paper citation rate (CRITICAL for good score)
- [ ] Each section has multiple citations
- [ ] Logical flow between sections

#### Common Issues & Solutions
- **Low citation rate**: Major issue - redistribute citations across sections
- **List-like sections**: Rewrite to synthesize findings
- **Missing sections**: Use template structure strictly

### Phase 4: Evaluation (survey-quality-evaluator)

#### Inputs
- All three files: survey, papers, clusters

#### Key Instructions for Agent
```markdown
CRITICAL: Be thorough but constructive
- Calculate exact citation percentage (papers cited / total papers)
- Check citation accuracy by matching to source
- Assess synthesis quality (not just presence of citations)
- Provide specific, actionable improvements
- Compare to publication standards for the venue type
```

#### Evaluation Dimensions (with lessons learned)
1. **Coverage (25%)**: 
   - Papers cited: Must be >50% for good score
   - Cluster coverage: All clusters should be discussed
   - Temporal span: Should cover the date range claimed

2. **Accuracy (30%)**:
   - Citation format consistency
   - Factual correctness of claims
   - No hallucinated papers or findings

3. **Organization (15%)**:
   - Logical section ordering
   - Clear transitions
   - Balanced section lengths

4. **Synthesis (20%)**:
   - Compare and contrast papers
   - Identify patterns and trends
   - Not just sequential descriptions

5. **Readability (10%)**:
   - Academic but accessible
   - Define technical terms
   - Vary sentence structure

#### Score Interpretation
- **9-10**: Exceptional, ready for top venues
- **8-9**: Very good, minor revisions for publication
- **7-8**: Good, moderate revisions needed (TARGET RANGE)
- **6-7**: Acceptable, significant improvements needed
- **<6**: Requires major rewrite

## Agent Coordination Best Practices

### Sequential Execution
```bash
# Each agent must complete before next starts
# DO NOT run agents in parallel - outputs are dependencies

1. paper-search-specialist → papers.json
   ↓ (dependency)
2. topic-mining-clustering → clusters.json  
   ↓ (dependency)
3. academic-survey-writer → survey.md
   ↓ (dependency)
4. survey-quality-evaluator → evaluation.json
```

### Inter-Agent Communication
- Each agent should read previous outputs completely
- Validate input files exist and have expected structure
- Log progress and any issues encountered
- Save intermediate results for debugging

### Error Recovery
- If agent fails, check input files first
- Validate JSON structure with jq or python
- Look for missing required fields
- Check file permissions and paths

## Optimization Tips

### For Faster Execution
1. **Cache API responses**: Reuse papers.json if re-running
2. **Batch operations**: Process embeddings in batches
3. **Use GPU**: Ensure CUDA is available for embeddings
4. **Limit scope**: 50-100 papers is sufficient for good survey

### For Better Quality
1. **Diverse searches**: Use many search query variations
2. **Recent papers**: Focus on last 2-3 years for relevance
3. **Citation coverage**: Aim for >60% papers cited
4. **Synthesis focus**: Compare papers, don't list them
5. **Specific examples**: Include numbers and specific findings

### For Debugging
1. **Check logs**: Each agent should log progress
2. **Validate JSON**: Use `python -m json.tool` to check
3. **Test components**: Run each module standalone first
4. **Monitor resources**: Check GPU/CPU/memory usage

## Template Files Structure

```
templates/
├── IMPROVED_SURVEY_SOP.md      # This file
├── SURVEY_PRD_TEMPLATE.md      # Product requirements
├── requirements.txt             # Python dependencies
├── agent_prompts/              # Refined agent instructions
│   ├── search_agent_v2.md     
│   ├── cluster_agent_v2.md    
│   ├── writer_agent_v2.md     
│   └── evaluator_agent_v2.md  
└── quality_checklist.md        # Validation checkpoints
```

## Quick Start Checklist

- [ ] Create project directory
- [ ] Set up uv environment with CUDA support
- [ ] Configure API keys in .env
- [ ] Copy and customize PRD template
- [ ] Initialize Task Master (optional)
- [ ] Run paper-search-specialist (wait for completion)
- [ ] Validate papers.json has >50 papers with abstracts
- [ ] Run topic-mining-clustering (wait for completion)
- [ ] Validate clusters.json has 5-10 named clusters
- [ ] Run academic-survey-writer (wait for completion)
- [ ] Validate survey.md is 5000+ words with citations
- [ ] Run survey-quality-evaluator
- [ ] Review evaluation.json for score >7.0

## Success Metrics

### Minimum Acceptable
- Papers collected: ≥50
- Clusters created: ≥5
- Survey length: ≥5000 words
- Papers cited: ≥30%
- Quality score: ≥6.0

### Target Performance
- Papers collected: 75-100
- Clusters created: 7-8
- Survey length: 6000-7000 words
- Papers cited: ≥60%
- Quality score: ≥7.5

### Excellent Result
- Papers collected: 100-150
- Clusters created: 8-10
- Survey length: 7000-8000 words
- Papers cited: ≥75%
- Quality score: ≥8.5

## Common Pitfalls to Avoid

1. **Don't skip environment setup** - uv + CUDA is critical
2. **Don't use mock data** - Real APIs only
3. **Don't parallelize agents** - They have dependencies
4. **Don't ignore abstracts** - Required for clustering
5. **Don't accept low citation rates** - Major quality issue
6. **Don't rush evaluation** - Thorough assessment improves next iteration
7. **Don't hardcode paths** - Use relative paths from project root
8. **Don't ignore outliers** - They may indicate issues
9. **Don't force cluster counts** - Let algorithm optimize
10. **Don't skip validation** - Check outputs at each stage

## Notes for Future Improvements

- Consider implementing checkpoint/resume functionality
- Add visualization of clusters and paper relationships
- Create automated quality improvement iterations
- Implement multi-language support for global coverage
- Add domain-specific templates for different fields
- Create benchmarking suite for comparing approaches
- Implement collaborative multi-agent review process