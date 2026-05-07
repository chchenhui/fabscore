# Product Requirements Document: {KEYWORD} Literature Survey

## Overview
Generate a comprehensive academic literature survey on "{KEYWORD}" using autonomous AI agents. The system will search for papers, cluster them into topics, generate survey content, and evaluate quality - all through Claude Code's agent orchestration.

## Core Features

### 1. Paper Discovery and Retrieval
- Search for 100-200 papers on "{KEYWORD}" from multiple sources
- Use Semantic Scholar and arXiv APIs
- Apply quality filters (citations, recency, relevance)
- Deduplicate results across sources

### 2. Topic Mining and Clustering
- Generate embeddings for all papers
- Identify 5-10 main research themes
- Assign papers to clusters
- Name clusters descriptively
- Identify relationships between topics

### 3. Survey Content Generation
- Create structured academic survey
- Generate abstract, introduction, topic sections, trends, conclusion
- Synthesize findings rather than listing papers
- Add proper citations throughout
- Target 5000-8000 words

### 4. Quality Evaluation
- Assess coverage, accuracy, organization, synthesis, readability
- Generate improvement suggestions
- Provide overall quality score
- Compare against baseline standards

## Technical Architecture

### Agent Orchestration
1. **paper-search-specialist**: Handles paper retrieval
2. **topic-mining-clustering**: Performs clustering analysis  
3. **academic-survey-writer**: Generates survey content
4. **survey-quality-evaluator**: Evaluates output quality

### Data Flow
1. Search agent → Papers JSON
2. Clustering agent → Clusters JSON  
3. Writing agent → Survey Markdown
4. Evaluation agent → Quality Report

## Development Roadmap

### Phase 1: Paper Collection
- Launch paper-search-specialist agent
- Search for "{KEYWORD}" papers
- Save results to data/papers/
- Validate minimum 50 papers found

### Phase 2: Topic Analysis
- Launch topic-mining-clustering agent
- Process papers from Phase 1
- Generate clusters and relationships
- Save clustering results

### Phase 3: Survey Generation
- Launch academic-survey-writer agent
- Use clusters from Phase 2
- Generate comprehensive survey
- Save to output/survey.md

### Phase 4: Quality Assessment
- Launch survey-quality-evaluator agent
- Evaluate generated survey
- Generate quality report
- Save evaluation results

## Logical Dependency Chain
1. Paper search must complete before clustering
2. Clustering must complete before survey writing
3. Survey writing must complete before evaluation
4. Each phase depends on previous phase output

## Success Criteria
- Minimum 50 papers collected
- 5-10 coherent topic clusters identified
- Survey length 5000-8000 words
- Quality score ≥ 7.0/10
- All sections properly cited

## Output Deliverables
- `papers.json`: Retrieved papers
- `clusters.json`: Topic clusters
- `survey.md`: Final survey document
- `evaluation.json`: Quality assessment
- `summary.json`: Pipeline summary