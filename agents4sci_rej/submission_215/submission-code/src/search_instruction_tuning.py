#!/usr/bin/env python3
"""
Comprehensive Academic Paper Search for Instruction Tuning LLMs
===============================================================

This script conducts a systematic literature search for papers on instruction tuning
and instruction-following capabilities in Large Language Models (LLMs).

Search Strategy:
- Multiple diverse queries targeting different aspects of instruction tuning
- Dual-source search: Semantic Scholar + arXiv
- Quality filtering: abstracts required, temporal filtering (2021-2025)
- Deduplication by title similarity and DOI
- Results ranked by relevance and impact

Author: Agentic Auto-Survey System
Date: September 2025
"""

import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import re

# Add scripts directory to Python path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir / "scripts"))

# Now import our modules
try:
    from core.api_clients import SemanticScholarClient, ArxivClient
    from core.utils import load_config, setup_logging, save_json, get_timestamp, calculate_similarity
    from loguru import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


class InstructionTuningSearch:
    """
    Comprehensive search system for instruction tuning research papers
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the search system"""
        # Setup logging
        setup_logging(config_path)
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize API clients
        self.s2_client = SemanticScholarClient(self.config)
        self.arxiv_client = ArxivClient(self.config)
        
        # Search parameters
        self.year_from = 2021
        self.year_to = 2025
        self.max_papers_per_query = 50  # Conservative per query to ensure quality
        self.target_total_papers = 100  # Target 50-100 high-quality papers
        
        # Quality thresholds
        self.min_citation_threshold = 1  # Lowered for recent papers
        self.title_similarity_threshold = 0.85  # For deduplication
        
        # Storage for results
        self.all_papers = []
        self.search_metadata = {
            'queries_used': [],
            'papers_found': {
                'semantic_scholar': 0,
                'arxiv': 0,
                'total_before_dedup': 0,
                'total_after_dedup': 0
            },
            'execution_time': None,
            'filters_applied': [],
            'duplicates_removed': 0,
            'search_timestamp': datetime.now().isoformat()
        }
        
        logger.info("InstructionTuningSearch initialized")
    
    def generate_search_queries(self) -> List[str]:
        """
        Generate comprehensive search queries for instruction tuning research
        
        Returns:
            List of search queries covering different aspects
        """
        queries = [
            # Core instruction tuning terms
            "instruction tuning",
            "instruction following LLM",
            "instruction following language model",
            "instruction-based fine-tuning",
            "instruction-tuned model",
            "instruction tuning large language model",
            
            # Specific methodologies
            "supervised fine-tuning instruction",
            "SFT instruction following",
            "multi-task instruction tuning",
            "zero-shot instruction following",
            "few-shot instruction following",
            
            # Famous models and papers
            "FLAN model instruction",
            "T5-FLAN instruction tuning",
            "InstructGPT alignment",
            "ChatGPT instruction following",
            "GPT-3 instruction tuning",
            "Alpaca instruction dataset",
            "Alpaca instruction following",
            "Vicuna instruction tuning",
            
            # Methodological approaches
            "self-instruct method",
            "self-instruct bootstrapping",
            "instruction generation synthetic",
            "instruction data generation",
            "instruction dataset creation",
            "human feedback instruction",
            "RLHF instruction following",
            
            # Technical aspects
            "prompt engineering instruction",
            "chain-of-thought instruction",
            "task-oriented fine-tuning",
            "multi-task learning instruction",
            "instruction comprehension",
            "instruction following evaluation",
            "instruction tuning benchmark",
            
            # Broader related concepts
            "teaching language models tasks",
            "natural language instruction AI",
            "conversational AI instruction",
            "dialogue system instruction",
            "task specification natural language",
            "human-AI interaction instruction"
        ]
        
        logger.info(f"Generated {len(queries)} search queries")
        return queries
    
    def search_semantic_scholar(self, query: str) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query
            
        Returns:
            List of paper dictionaries
        """
        try:
            papers = self.s2_client.search(
                query=query,
                limit=self.max_papers_per_query,
                year_from=self.year_from
            )
            logger.info(f"Semantic Scholar found {len(papers)} papers for query: '{query}'")
            return papers
        except Exception as e:
            logger.error(f"Semantic Scholar search failed for '{query}': {e}")
            return []
    
    def search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query
            
        Returns:
            List of paper dictionaries
        """
        try:
            papers = self.arxiv_client.search(
                query=query,
                limit=self.max_papers_per_query,
                year_from=self.year_from
            )
            logger.info(f"arXiv found {len(papers)} papers for query: '{query}'")
            return papers
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")
            return []
    
    def filter_papers_by_relevance(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter papers by relevance to instruction tuning
        
        Args:
            papers: List of papers to filter
            query: Original search query for context
            
        Returns:
            Filtered list of papers
        """
        instruction_keywords = {
            'instruction', 'instruct', 'teaching', 'fine-tuning', 'fine-tune',
            'alignment', 'following', 'task', 'prompt', 'training', 'learning',
            'supervised', 'chatgpt', 'gpt', 'language model', 'llm', 'flan',
            'alpaca', 'vicuna', 'self-instruct', 'human feedback', 'rlhf',
            'chain-of-thought', 'zero-shot', 'few-shot', 'multi-task',
            'conversational', 'dialogue', 'natural language'
        }
        
        filtered_papers = []
        for paper in papers:
            # Must have abstract
            abstract = paper.get('abstract', '')
            if not abstract or len(abstract.strip()) < 50:
                continue
            
            # Check title and abstract for relevance
            title = paper.get('title', '').lower()
            abstract_lower = abstract.lower()
            
            # Count keyword matches
            keyword_matches = 0
            text_to_check = f"{title} {abstract_lower}"
            
            for keyword in instruction_keywords:
                if keyword in text_to_check:
                    keyword_matches += 1
            
            # Require at least 2 keyword matches for relevance
            if keyword_matches >= 2:
                paper['relevance_score'] = min(keyword_matches / 10.0, 1.0)  # Normalize to 0-1
                filtered_papers.append(paper)
        
        logger.info(f"Filtered {len(filtered_papers)} relevant papers from {len(papers)} for query: '{query}'")
        return filtered_papers
    
    def remove_duplicates(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate papers based on title similarity and DOI
        
        Args:
            papers: List of papers to deduplicate
            
        Returns:
            List of unique papers
        """
        unique_papers = []
        seen_dois = set()
        seen_titles = []
        duplicates_removed = 0
        
        for paper in papers:
            # Check DOI first (exact match)
            doi = paper.get('doi')
            if doi and doi in seen_dois:
                duplicates_removed += 1
                continue
            
            # Check title similarity
            title = paper.get('title', '').strip()
            if not title:
                continue
            
            is_duplicate = False
            for existing_title in seen_titles:
                similarity = calculate_similarity(title, existing_title)
                if similarity >= self.title_similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.append(title)
                if doi:
                    seen_dois.add(doi)
        
        self.search_metadata['duplicates_removed'] = duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicates, {len(unique_papers)} unique papers remaining")
        return unique_papers
    
    def apply_quality_filters(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply quality filters to papers
        
        Args:
            papers: List of papers to filter
            
        Returns:
            Filtered papers meeting quality criteria
        """
        filtered_papers = []
        filters_applied = []
        
        for paper in papers:
            # Must have abstract
            if not paper.get('abstract') or len(paper.get('abstract', '').strip()) < 50:
                continue
            
            # Must have title
            if not paper.get('title') or len(paper.get('title', '').strip()) < 10:
                continue
            
            # Must have authors
            if not paper.get('authors') or len(paper.get('authors', [])) == 0:
                continue
            
            # Must have year within range
            year = paper.get('year')
            if not year or year < self.year_from or year > self.year_to:
                continue
            
            # Citation filter (relaxed for recent papers)
            citations = paper.get('citations', 0)
            current_year = datetime.now().year
            paper_age = current_year - year
            
            # More lenient citation requirements for newer papers
            min_citations_needed = max(0, self.min_citation_threshold - max(0, 3 - paper_age))
            if citations < min_citations_needed and year < 2023:  # Only apply to older papers
                continue
            
            filtered_papers.append(paper)
        
        filters_applied = [
            "Must have abstract (>50 chars)",
            "Must have title (>10 chars)", 
            "Must have authors",
            f"Year range: {self.year_from}-{self.year_to}",
            "Citation threshold (age-adjusted)"
        ]
        
        self.search_metadata['filters_applied'] = filters_applied
        logger.info(f"Quality filtering: {len(filtered_papers)} papers passed from {len(papers)}")
        return filtered_papers
    
    def rank_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank papers by relevance and quality
        
        Args:
            papers: List of papers to rank
            
        Returns:
            Ranked list of papers
        """
        for paper in papers:
            # Calculate composite score
            relevance = paper.get('relevance_score', 0.5)
            citations = paper.get('citations', 0)
            year = paper.get('year', 2021)
            
            # Normalize citation count (log scale)
            citation_score = min(1.0, (citations + 1) / 100.0)
            
            # Recency bonus (more weight for recent papers)
            recency_score = (year - 2020) / (2025 - 2020)  # Normalize to 0-1
            
            # Venue quality (simplified - arXiv gets bonus for being preprint)
            venue_score = 0.8 if paper.get('source') == 'arxiv' else 0.6
            
            # Composite score
            composite_score = (
                0.4 * relevance +
                0.3 * citation_score +
                0.2 * venue_score +
                0.1 * recency_score
            )
            
            paper['composite_score'] = composite_score
        
        # Sort by composite score descending
        ranked_papers = sorted(papers, key=lambda x: x.get('composite_score', 0), reverse=True)
        logger.info(f"Ranked {len(ranked_papers)} papers by composite score")
        return ranked_papers
    
    def execute_comprehensive_search(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute comprehensive search across all sources and queries
        
        Returns:
            Tuple of (papers_list, metadata_dict)
        """
        start_time = time.time()
        logger.info("Starting comprehensive instruction tuning paper search")
        
        # Generate search queries
        queries = self.generate_search_queries()
        self.search_metadata['queries_used'] = queries
        
        all_papers = []
        
        # Search each query across both sources
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            
            # Search Semantic Scholar
            s2_papers = self.search_semantic_scholar(query)
            s2_filtered = self.filter_papers_by_relevance(s2_papers, query)
            all_papers.extend(s2_filtered)
            self.search_metadata['papers_found']['semantic_scholar'] += len(s2_filtered)
            
            # Brief pause between API calls
            time.sleep(1)
            
            # Search arXiv
            arxiv_papers = self.search_arxiv(query)
            arxiv_filtered = self.filter_papers_by_relevance(arxiv_papers, query)
            all_papers.extend(arxiv_filtered)
            self.search_metadata['papers_found']['arxiv'] += len(arxiv_filtered)
            
            # Longer pause between queries to be respectful
            time.sleep(2)
            
            # Early stopping if we have enough papers
            if len(all_papers) > self.target_total_papers * 3:  # 3x target for dedup buffer
                logger.info(f"Sufficient papers found ({len(all_papers)}), stopping early")
                break
        
        self.search_metadata['papers_found']['total_before_dedup'] = len(all_papers)
        logger.info(f"Raw search completed: {len(all_papers)} papers found")
        
        # Apply deduplication
        unique_papers = self.remove_duplicates(all_papers)
        
        # Apply quality filters
        quality_papers = self.apply_quality_filters(unique_papers)
        
        # Rank papers
        ranked_papers = self.rank_papers(quality_papers)
        
        # Take top papers up to target
        final_papers = ranked_papers[:self.target_total_papers]
        
        # Update metadata
        self.search_metadata['papers_found']['total_after_dedup'] = len(final_papers)
        self.search_metadata['execution_time'] = f"{time.time() - start_time:.2f} seconds"
        
        logger.info(f"Search completed: {len(final_papers)} high-quality papers selected")
        return final_papers, self.search_metadata
    
    def save_results(self, papers: List[Dict[str, Any]], metadata: Dict[str, Any], 
                    output_dir: str = "/path/to/project/output/instruction_tuning/output"):
        """
        Save search results to files
        
        Args:
            papers: List of papers to save
            metadata: Search metadata
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save papers JSON
        papers_file = output_path / "papers.json"
        final_data = {
            "search_metadata": metadata,
            "papers": papers,
            "summary": {
                "total_papers": len(papers),
                "year_range": f"{self.year_from}-{self.year_to}",
                "sources": ["Semantic Scholar", "arXiv"],
                "search_date": datetime.now().isoformat()
            }
        }
        
        save_json(final_data, papers_file)
        logger.info(f"Saved {len(papers)} papers to {papers_file}")
        
        # Generate and save markdown report
        self.generate_markdown_report(papers, metadata, output_path / "search_report.md")
    
    def generate_markdown_report(self, papers: List[Dict[str, Any]], metadata: Dict[str, Any], output_file: Path):
        """
        Generate comprehensive markdown search report
        
        Args:
            papers: List of papers
            metadata: Search metadata
            output_file: Output file path
        """
        report = []
        
        # Header
        report.append("# Instruction Tuning for Large Language Models - Literature Search Report")
        report.append("")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Search completed in:** {metadata.get('execution_time', 'N/A')}")
        report.append("")
        
        # Search Summary
        report.append("## Search Summary")
        report.append("")
        papers_found = metadata['papers_found']
        report.append(f"- **Total queries executed:** {len(metadata['queries_used'])}")
        report.append(f"- **Semantic Scholar results:** {papers_found['semantic_scholar']}")
        report.append(f"- **arXiv results:** {papers_found['arxiv']}")
        report.append(f"- **Total before deduplication:** {papers_found['total_before_dedup']}")
        report.append(f"- **Duplicates removed:** {metadata['duplicates_removed']}")
        report.append(f"- **Final high-quality papers:** {papers_found['total_after_dedup']}")
        report.append("")
        
        # Search Queries Used
        report.append("## Search Queries")
        report.append("")
        report.append("The following search queries were used to ensure comprehensive coverage:")
        report.append("")
        for i, query in enumerate(metadata['queries_used'], 1):
            report.append(f"{i}. \"{query}\"")
        report.append("")
        
        # Quality Filters
        report.append("## Quality Filters Applied")
        report.append("")
        for filter_desc in metadata.get('filters_applied', []):
            report.append(f"- {filter_desc}")
        report.append("")
        
        # Year Distribution
        year_dist = defaultdict(int)
        source_dist = defaultdict(int)
        
        for paper in papers:
            year = paper.get('year')
            if year:
                year_dist[year] += 1
            source = paper.get('source', 'unknown')
            source_dist[source] += 1
        
        report.append("## Paper Distribution")
        report.append("")
        report.append("### By Year")
        for year in sorted(year_dist.keys(), reverse=True):
            report.append(f"- **{year}:** {year_dist[year]} papers")
        report.append("")
        
        report.append("### By Source")
        for source, count in source_dist.items():
            report.append(f"- **{source.title()}:** {count} papers")
        report.append("")
        
        # Top Papers
        report.append("## Top 20 Most Relevant Papers")
        report.append("")
        
        top_papers = papers[:20]  # Already ranked
        
        for i, paper in enumerate(top_papers, 1):
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', [])
            year = paper.get('year', 'N/A')
            citations = paper.get('citations', 0)
            venue = paper.get('venue', 'N/A')
            score = paper.get('composite_score', 0)
            
            # Format authors
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif len(authors) > 1:
                author_str = ", ".join(authors[:-1]) + f" & {authors[-1]}"
            else:
                author_str = authors[0] if authors else "Unknown"
            
            report.append(f"### {i}. {title}")
            report.append(f"**Authors:** {author_str}  ")
            report.append(f"**Year:** {year} | **Citations:** {citations} | **Venue:** {venue}  ")
            report.append(f"**Relevance Score:** {score:.3f}  ")
            
            abstract = paper.get('abstract', '')
            if abstract:
                # Truncate abstract if too long
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                report.append(f"**Abstract:** {abstract}")
            
            url = paper.get('url')
            if url:
                report.append(f"**URL:** {url}")
            
            report.append("")
        
        # All Papers List
        report.append("## Complete Paper List")
        report.append("")
        report.append("| # | Title | Authors | Year | Citations | Source |")
        report.append("|---|-------|---------|------|-----------|---------|")
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Untitled')[:80]  # Truncate long titles
            authors = paper.get('authors', [])
            year = paper.get('year', 'N/A')
            citations = paper.get('citations', 0)
            source = paper.get('source', 'unknown').title()
            
            # Format authors for table
            if len(authors) > 2:
                author_str = f"{authors[0]} et al."
            else:
                author_str = ", ".join(authors) if authors else "Unknown"
            
            # Escape markdown special characters
            title = title.replace('|', '\\|')
            author_str = author_str.replace('|', '\\|')
            
            report.append(f"| {i} | {title} | {author_str} | {year} | {citations} | {source} |")
        
        report.append("")
        
        # Research Insights
        report.append("## Key Research Areas Identified")
        report.append("")
        
        # Extract common themes from titles
        themes = self.extract_research_themes(papers)
        report.append("Based on the paper titles and abstracts, the following key research areas were identified:")
        report.append("")
        for theme, count in themes.items():
            report.append(f"- **{theme}:** {count} papers")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        report.append("")
        report.append(f"This comprehensive search identified {len(papers)} high-quality papers on instruction tuning for large language models, ")
        report.append(f"spanning from {min(year_dist.keys())} to {max(year_dist.keys())}. The results cover major methodological approaches, ")
        report.append("foundational models, evaluation techniques, and practical applications in the rapidly evolving field of instruction-following AI systems.")
        report.append("")
        report.append("---")
        report.append("*Generated by Agentic Auto-Survey System*")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated comprehensive search report: {output_file}")
    
    def extract_research_themes(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract common research themes from paper titles
        
        Args:
            papers: List of papers
            
        Returns:
            Dictionary of themes and their counts
        """
        themes = defaultdict(int)
        
        theme_keywords = {
            "Self-Instruction & Bootstrapping": ["self-instruct", "bootstrap", "self-supervised", "auto-instruction"],
            "Human Feedback & Alignment": ["human feedback", "rlhf", "alignment", "human preference", "reward model"],
            "Fine-tuning Methods": ["fine-tun", "sft", "supervised", "parameter-efficient", "lora", "adapter"],
            "Chain-of-Thought": ["chain-of-thought", "cot", "reasoning", "step-by-step", "rationale"],
            "Multi-task Learning": ["multi-task", "multitask", "task-general", "cross-task", "universal"],
            "Zero/Few-shot Learning": ["zero-shot", "few-shot", "in-context", "prompt", "demonstration"],
            "Evaluation & Benchmarking": ["evaluation", "benchmark", "metric", "assessment", "performance"],
            "Dataset Creation": ["dataset", "data generation", "synthetic", "collection", "annotation"],
            "Model Architectures": ["transformer", "t5", "gpt", "bert", "architecture", "model"],
            "Conversational AI": ["conversation", "dialogue", "chat", "interactive", "assistant"]
        }
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            text = f"{title} {abstract}"
            
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        themes[theme] += 1
                        break  # Count each theme only once per paper
        
        # Sort by frequency
        return dict(sorted(themes.items(), key=lambda x: x[1], reverse=True))


def main():
    """Main execution function"""
    try:
        # Initialize search system
        searcher = InstructionTuningSearch()
        
        # Execute comprehensive search
        papers, metadata = searcher.execute_comprehensive_search()
        
        # Save results
        searcher.save_results(papers, metadata)
        
        # Print summary
        print("\n" + "="*80)
        print("INSTRUCTION TUNING LITERATURE SEARCH COMPLETED")
        print("="*80)
        print(f"Papers found: {len(papers)}")
        print(f"Execution time: {metadata['execution_time']}")
        print(f"Sources: Semantic Scholar ({metadata['papers_found']['semantic_scholar']}), " + 
              f"arXiv ({metadata['papers_found']['arxiv']})")
        print(f"Output saved to: /path/to/project/output/instruction_tuning/output/")
        print("="*80)
        
        # Print top 5 papers
        print("\nTop 5 Papers by Relevance:")
        for i, paper in enumerate(papers[:5], 1):
            print(f"\n{i}. {paper.get('title', 'Untitled')}")
            authors = paper.get('authors', [])
            if authors:
                author_str = authors[0] + " et al." if len(authors) > 1 else authors[0]
                print(f"   Authors: {author_str}")
            print(f"   Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citations', 0)} | Score: {paper.get('composite_score', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"\nERROR: Search failed - {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)