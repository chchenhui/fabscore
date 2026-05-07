#!/usr/bin/env python3
"""
Comprehensive Synthetic Data Paper Search for Machine Learning and LLMs

This script conducts a systematic literature search across multiple academic databases
to find high-quality papers related to synthetic data generation, evaluation, and applications
in machine learning and large language models.

Key Features:
- 25-30 diverse search queries covering all aspects of synthetic data
- Multi-source search (Semantic Scholar, arXiv)
- Quality filters for high-impact, recent papers
- Comprehensive deduplication and ranking
- Focus on privacy-preserving techniques, GANs, diffusion models, LLM training
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import hashlib
from collections import defaultdict

# Add the core modules to path
sys.path.append(str(Path(__file__).parent))
from core.api_clients import SemanticScholarClient, ArxivClient
from core.paper_processor import PaperProcessor
from core.utils import load_config, save_json, get_timestamp
from loguru import logger


class ComprehensiveSyntheticDataSearcher:
    """Advanced paper searcher for synthetic data across ML and LLM domains"""
    
    def __init__(self, output_file: str):
        """Initialize the searcher with specific output file"""
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = load_config()
        
        # Initialize API clients with enhanced configuration
        self.semantic_scholar = SemanticScholarClient(self.config)
        self.arxiv = ArxivClient(self.config)
        self.processor = PaperProcessor(self.config)
        
        # Enhanced search parameters
        self.year_from = 2020
        self.year_to = 2024
        self.target_papers = 200  # Target for comprehensive collection
        self.min_citations = 3    # Lower for recent synthetic data papers
        self.papers_per_query = 8  # Moderate per query to ensure coverage
        
        # Results storage and metadata
        self.all_papers = []
        self.query_results = {}
        self.search_metadata = {
            'search_keyword': 'Synthetic Data',
            'search_date': datetime.now().isoformat(),
            'queries_used': [],
            'sources_searched': ['semantic_scholar', 'arxiv'],
            'year_range': f"{self.year_from}-{self.year_to}",
            'papers_found_raw': {},
            'papers_found_filtered': 0,
            'duplicates_removed': 0,
            'execution_time': None,
            'quality_filters': [
                'has_abstract', 'has_title', 'english_only',
                f'year_{self.year_from}_to_{self.year_to}',
                f'min_citations_{self.min_citations}_adjusted_by_age',
                'relevance_threshold'
            ]
        }

    def generate_comprehensive_queries(self) -> List[str]:
        """Generate 25-30 comprehensive search queries covering all synthetic data aspects"""
        
        queries = [
            # === Core Synthetic Data Generation ===
            "synthetic data generation",
            "artificial data generation machine learning",
            "synthetic dataset creation",
            "data synthesis methods",
            "synthetic training data",
            
            # === Privacy-Preserving Synthetic Data ===
            "differential privacy synthetic data",
            "privacy preserving data generation",
            "federated synthetic data",
            "anonymization synthetic datasets",
            "privacy utility trade-off synthetic data",
            
            # === Generative Models for Data Synthesis ===
            "GAN synthetic data generation",
            "diffusion models synthetic data",
            "VAE synthetic data",
            "generative adversarial networks data augmentation",
            "stable diffusion synthetic datasets",
            "flow-based models synthetic data",
            
            # === LLM and NLP Synthetic Data ===
            "synthetic data large language models",
            "LLM training synthetic data",
            "synthetic text generation training",
            "data augmentation language models",
            "synthetic instruction tuning",
            "self-instruct synthetic examples",
            "GPT synthetic training data",
            
            # === Quality and Evaluation ===
            "synthetic data quality evaluation",
            "fidelity utility privacy synthetic data",
            "model collapse synthetic training",
            "evaluating synthetic datasets",
            "synthetic vs real data performance",
            "synthetic data benchmark evaluation",
            
            # === Domain-Specific Applications ===
            "medical synthetic data generation",
            "healthcare privacy synthetic datasets",
            "financial synthetic data",
            "tabular synthetic data generation",
            "time series synthetic data",
            "image synthetic data augmentation",
            "computer vision synthetic datasets",
            
            # === Advanced Techniques ===
            "conditional synthetic data generation",
            "few-shot synthetic data",
            "transfer learning synthetic data",
            "multi-modal synthetic data",
            "controllable synthetic data generation",
            "adversarial training synthetic data"
        ]
        
        logger.info(f"Generated {len(queries)} comprehensive search queries")
        return queries

    def search_semantic_scholar_comprehensive(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Enhanced Semantic Scholar search with comprehensive coverage"""
        logger.info(f"Searching Semantic Scholar with {len(queries)} queries...")
        
        all_papers = []
        successful_queries = 0
        
        for i, query in enumerate(queries):
            logger.info(f"  Query {i+1}/{len(queries)}: '{query}'")
            
            try:
                papers = self.semantic_scholar.search(
                    query=query,
                    limit=self.papers_per_query,
                    year_from=self.year_from
                )
                
                # Filter for relevance and quality
                relevant_papers = [p for p in papers if self._is_highly_relevant(p, query)]
                all_papers.extend(relevant_papers)
                
                # Track query performance
                self.query_results[f"s2_{query}"] = {
                    'total_found': len(papers),
                    'relevant_found': len(relevant_papers),
                    'source': 'semantic_scholar'
                }
                
                successful_queries += 1
                logger.info(f"    Found {len(relevant_papers)} relevant papers")
                
                # Rate limiting
                time.sleep(1.2)
                
            except Exception as e:
                logger.warning(f"    Query failed: {e}")
                continue
        
        logger.info(f"Semantic Scholar: {len(all_papers)} papers from {successful_queries} successful queries")
        self.search_metadata['papers_found_raw']['semantic_scholar'] = len(all_papers)
        return all_papers

    def search_arxiv_comprehensive(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Enhanced arXiv search with domain-specific focus"""
        logger.info(f"Searching arXiv with {len(queries)} queries...")
        
        all_papers = []
        successful_queries = 0
        
        for i, query in enumerate(queries):
            logger.info(f"  Query {i+1}/{len(queries)}: '{query}'")
            
            try:
                papers = self.arxiv.search(
                    query=query,
                    limit=self.papers_per_query,
                    year_from=self.year_from
                )
                
                # Filter for relevance
                relevant_papers = [p for p in papers if self._is_highly_relevant(p, query)]
                all_papers.extend(relevant_papers)
                
                # Track query performance
                self.query_results[f"arxiv_{query}"] = {
                    'total_found': len(papers),
                    'relevant_found': len(relevant_papers),
                    'source': 'arxiv'
                }
                
                successful_queries += 1
                logger.info(f"    Found {len(relevant_papers)} relevant papers")
                
                # arXiv rate limiting (more conservative)
                time.sleep(2.5)
                
            except Exception as e:
                logger.warning(f"    Query failed: {e}")
                continue
        
        logger.info(f"arXiv: {len(all_papers)} papers from {successful_queries} successful queries")
        self.search_metadata['papers_found_raw']['arxiv'] = len(all_papers)
        return all_papers

    def _is_highly_relevant(self, paper: Dict[str, Any], query: str) -> bool:
        """Enhanced relevance checking for synthetic data papers"""
        
        # Essential field requirements
        if not paper.get('abstract') or not paper.get('title'):
            return False
        
        # Year filtering
        year = paper.get('year')
        if not year or year < self.year_from or year > self.year_to:
            return False
        
        # Combine title and abstract for analysis
        text_content = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        
        # Core synthetic data indicators (must have at least one)
        core_indicators = [
            'synthetic', 'artificial', 'generated', 'simulation', 'augmentation',
            'fake', 'pseudo', 'surrogate', 'proxy'
        ]
        
        has_core = any(indicator in text_content for indicator in core_indicators)
        if not has_core:
            return False
        
        # Domain relevance indicators
        domain_indicators = [
            # ML/AI terms
            'machine learning', 'deep learning', 'neural', 'model', 'training',
            'dataset', 'data', 'algorithm', 'learning',
            
            # Generative models
            'gan', 'generative', 'diffusion', 'vae', 'autoencoder', 'transformer',
            'gpt', 'bert', 'llm', 'language model',
            
            # Privacy and evaluation
            'privacy', 'differential privacy', 'federated', 'evaluation',
            'quality', 'fidelity', 'utility', 'benchmark',
            
            # Applications
            'medical', 'healthcare', 'financial', 'tabular', 'image', 'text',
            'time series', 'computer vision', 'nlp'
        ]
        
        domain_matches = sum(1 for indicator in domain_indicators if indicator in text_content)
        
        # Query-specific relevance boost
        query_terms = query.lower().split()
        query_matches = sum(1 for term in query_terms if term in text_content)
        
        # Scoring: need good domain coverage plus query relevance
        relevance_score = domain_matches + (query_matches * 2)  # Boost query matches
        
        return relevance_score >= 3  # Threshold for relevance

    def apply_quality_filters(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply comprehensive quality filters for synthetic data papers"""
        logger.info(f"Applying quality filters to {len(papers)} papers...")
        
        filtered_papers = []
        current_year = datetime.now().year
        
        for paper in papers:
            # Must have comprehensive abstract (critical for synthetic data papers)
            abstract = paper.get('abstract', '')
            if not abstract or len(abstract.split()) < 20:
                continue
            
            # Must have meaningful title
            title = paper.get('title', '')
            if not title or len(title.split()) < 3:
                continue
            
            # Year and citation filtering with age adjustment
            year = paper.get('year', 0)
            if not year or year < self.year_from or year > self.year_to:
                continue
            
            citations = paper.get('citations', 0)
            # Adjust citation threshold based on paper age
            age = current_year - year
            min_citations_adjusted = max(0, self.min_citations - age)
            
            if citations < min_citations_adjusted:
                continue
            
            # Enhanced English detection
            if not self._is_quality_english_text(title + ' ' + abstract):
                continue
            
            # Content quality indicators for synthetic data
            if not self._has_quality_content_indicators(title, abstract):
                continue
            
            filtered_papers.append(paper)
        
        logger.info(f"Quality filtering: {len(papers)} -> {len(filtered_papers)} papers")
        return filtered_papers

    def _is_quality_english_text(self, text: str, threshold: float = 0.85) -> bool:
        """Enhanced English text detection"""
        if not text:
            return False
        
        # ASCII character ratio
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        ascii_ratio = ascii_chars / total_chars if total_chars > 0 else 0
        
        # English word indicators
        english_indicators = [
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'this', 'that', 'we', 'our', 'data', 'method', 'model', 'results'
        ]
        
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        return ascii_ratio > threshold and english_word_count >= 3

    def _has_quality_content_indicators(self, title: str, abstract: str) -> bool:
        """Check for quality content indicators specific to synthetic data research"""
        
        content = (title + ' ' + abstract).lower()
        
        # Quality research indicators
        quality_indicators = [
            # Methodological terms
            'method', 'approach', 'framework', 'algorithm', 'technique',
            'model', 'architecture', 'system',
            
            # Experimental terms
            'evaluation', 'experiment', 'performance', 'results', 'analysis',
            'comparison', 'benchmark', 'metric',
            
            # Research terms
            'propose', 'novel', 'new', 'improve', 'enhance', 'show',
            'demonstrate', 'achieve', 'outperform'
        ]
        
        quality_count = sum(1 for indicator in quality_indicators if indicator in content)
        return quality_count >= 2

    def calculate_advanced_relevance_score(self, paper: Dict[str, Any]) -> float:
        """Calculate sophisticated relevance score for synthetic data papers"""
        
        score = 0.0
        text_content = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        
        # High-impact synthetic data terms
        high_impact_terms = [
            ('differential privacy', 0.4),
            ('gan', 0.3), ('generative adversarial', 0.3),
            ('diffusion model', 0.35), ('stable diffusion', 0.3),
            ('synthetic data quality', 0.35),
            ('model collapse', 0.4),
            ('privacy utility', 0.35),
            ('federated synthetic', 0.3)
        ]
        
        for term, weight in high_impact_terms:
            if term in text_content:
                score += weight
        
        # Domain-specific terms
        domain_terms = [
            ('medical synthetic', 0.25), ('healthcare synthetic', 0.25),
            ('financial synthetic', 0.2), ('tabular synthetic', 0.2),
            ('llm training', 0.3), ('instruction tuning', 0.25),
            ('data augmentation', 0.2), ('few-shot', 0.15)
        ]
        
        for term, weight in domain_terms:
            if term in text_content:
                score += weight
        
        # Citation impact (normalized by age and synthetic data context)
        citations = paper.get('citations', 0)
        year = paper.get('year', 2024)
        age = max(1, 2024 - year + 1)
        
        # Synthetic data papers often have different citation patterns
        citation_score = min(0.3, citations / (age * 15))  # Adjusted for domain
        score += citation_score
        
        # Recency bonus for fast-moving field
        if year >= 2023:
            score += 0.15
        elif year >= 2022:
            score += 0.1
        elif year >= 2021:
            score += 0.05
        
        # High-impact venue bonus
        venue = paper.get('venue', '').lower()
        top_venues = [
            'neurips', 'icml', 'iclr', 'nature', 'science',
            'acl', 'emnlp', 'aaai', 'ijcai', 'kdd', 'www',
            'usenix security', 'ccs', 'sp', 'oakland'
        ]
        
        if any(v in venue for v in top_venues):
            score += 0.15
        elif 'arxiv' in venue:
            score += 0.05
        
        return score

    def execute_comprehensive_search(self) -> List[Dict[str, Any]]:
        """Execute the complete comprehensive search workflow"""
        
        start_time = time.time()
        logger.info("=== Starting Comprehensive Synthetic Data Paper Search ===")
        logger.info(f"Target: {self.target_papers} high-quality papers")
        logger.info(f"Year range: {self.year_from}-{self.year_to}")
        logger.info(f"Output file: {self.output_file}")
        
        # Generate comprehensive queries
        queries = self.generate_comprehensive_queries()
        self.search_metadata['queries_used'] = queries
        logger.info(f"Generated {len(queries)} search queries")
        
        # Execute multi-source search
        logger.info("\n1. Searching Semantic Scholar...")
        s2_papers = self.search_semantic_scholar_comprehensive(queries)
        
        logger.info("\n2. Searching arXiv...")
        arxiv_papers = self.search_arxiv_comprehensive(queries)
        
        # Combine and process results
        logger.info("\n3. Processing and filtering results...")
        all_papers = s2_papers + arxiv_papers
        logger.info(f"Combined raw papers: {len(all_papers)}")
        
        # Deduplicate using the processor
        unique_papers = self.processor.deduplicate_papers(all_papers)
        duplicates_removed = len(all_papers) - len(unique_papers)
        self.search_metadata['duplicates_removed'] = duplicates_removed
        logger.info(f"After deduplication: {len(unique_papers)} papers")
        
        # Apply quality filters
        filtered_papers = self.apply_quality_filters(unique_papers)
        self.search_metadata['papers_found_filtered'] = len(filtered_papers)
        
        # Calculate relevance scores and rank
        logger.info("4. Ranking papers by relevance...")
        for paper in filtered_papers:
            paper['relevance_score'] = self.calculate_advanced_relevance_score(paper)
        
        # Sort by relevance score
        ranked_papers = sorted(filtered_papers, key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        # Take top papers up to target
        final_papers = ranked_papers[:self.target_papers]
        
        # Finalize metadata
        end_time = time.time()
        self.search_metadata['execution_time'] = f"{end_time - start_time:.2f} seconds"
        self.search_metadata['total_raw'] = len(all_papers)
        self.search_metadata['total_filtered'] = len(final_papers)
        self.search_metadata['sources'] = {
            'semantic_scholar': self.search_metadata['papers_found_raw'].get('semantic_scholar', 0),
            'arxiv': self.search_metadata['papers_found_raw'].get('arxiv', 0)
        }
        
        logger.info(f"\n=== Search Complete ===")
        logger.info(f"Final collection: {len(final_papers)} high-quality papers")
        logger.info(f"Execution time: {self.search_metadata['execution_time']}")
        logger.info(f"Quality coverage: {len(final_papers)}/{self.target_papers} target papers")
        
        return final_papers

    def save_comprehensive_results(self, papers: List[Dict[str, Any]]) -> None:
        """Save comprehensive results with full metadata and statistics"""
        
        logger.info("Saving comprehensive results...")
        
        # Calculate detailed statistics
        stats = self.processor.get_statistics(papers)
        
        # Add query performance analysis
        query_performance = {
            'total_queries': len(self.search_metadata['queries_used']),
            'successful_queries': len([q for q in self.query_results.values() if q['relevant_found'] > 0]),
            'avg_papers_per_query': sum(q['relevant_found'] for q in self.query_results.values()) / len(self.query_results) if self.query_results else 0,
            'top_performing_queries': sorted(
                [(k.replace('s2_', '').replace('arxiv_', ''), v['relevant_found']) 
                 for k, v in self.query_results.items()],
                key=lambda x: x[1], reverse=True)[:10]
        }
        
        # Prepare final result structure
        result_data = {
            'metadata': {
                'keyword': self.search_metadata['search_keyword'],
                'search_date': self.search_metadata['search_date'],
                'total_raw': self.search_metadata['total_raw'],
                'total_filtered': self.search_metadata['total_filtered'],
                'duplicates_removed': self.search_metadata['duplicates_removed'],
                'execution_time': self.search_metadata['execution_time'],
                'year_range': self.search_metadata['year_range'],
                'sources': self.search_metadata['sources'],
                'quality_filters': self.search_metadata['quality_filters'],
                'statistics': stats,
                'query_performance': query_performance
            },
            'papers': papers
        }
        
        # Save main results file
        save_json(result_data, str(self.output_file))
        
        # Generate and save search report
        self.generate_comprehensive_report(papers, stats, query_performance)
        
        logger.info(f"Results saved successfully:")
        logger.info(f"  - Papers: {self.output_file}")
        logger.info(f"  - Report: {self.output_file.with_suffix('.md')}")

    def generate_comprehensive_report(self, papers: List[Dict[str, Any]], 
                                    stats: Dict[str, Any], 
                                    query_performance: Dict[str, Any]) -> None:
        """Generate comprehensive search and analysis report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = self.output_file.with_suffix('.md')
        
        # Calculate additional analysis
        year_dist = stats.get('years', {})
        venue_dist = stats.get('venues', {})
        top_venues = stats.get('top_venues', [])
        
        # Top papers by citations and relevance
        top_cited = sorted(papers, key=lambda p: p.get('citations', 0), reverse=True)[:10]
        top_relevant = sorted(papers, key=lambda p: p.get('relevance_score', 0), reverse=True)[:10]
        
        # Recent papers (2023-2024)
        recent_papers = [p for p in papers if p.get('year', 0) >= 2023]
        recent_sorted = sorted(recent_papers, key=lambda p: p.get('relevance_score', 0), reverse=True)[:15]
        
        report_content = f"""# Synthetic Data Research - Comprehensive Literature Search Report

**Generated**: {timestamp}  
**Search Keywords**: Synthetic Data Generation, Privacy-Preserving Data, GANs, Diffusion Models, LLM Training  
**Year Range**: {self.search_metadata['year_range']}  
**Total Papers Collected**: {len(papers)}

## Executive Summary

This comprehensive search targeted high-quality papers on synthetic data generation and applications in machine learning and large language models. The search covered:

- **Core synthetic data generation techniques**
- **Privacy-preserving synthetic data methods** 
- **Generative models (GANs, diffusion models, VAEs)**
- **Synthetic data for LLM training and instruction tuning**
- **Quality evaluation and model collapse issues**
- **Domain-specific applications (medical, financial, etc.)**

## Search Strategy & Coverage

### Query Strategy
We employed **{len(self.search_metadata['queries_used'])} diverse search queries** covering:

#### Core Areas Covered:
- Synthetic data generation fundamentals
- Privacy-preserving techniques (differential privacy, federated learning)
- Generative model architectures (GANs, diffusion, VAE, flow-based)
- LLM and NLP applications (instruction tuning, self-instruct, data augmentation)
- Quality evaluation and fidelity assessment
- Domain-specific applications (healthcare, finance, computer vision)
- Advanced techniques (conditional generation, few-shot, multi-modal)

#### Multi-Source Search Results:
- **Semantic Scholar**: {self.search_metadata['sources'].get('semantic_scholar', 0)} papers
- **arXiv**: {self.search_metadata['sources'].get('arxiv', 0)} papers
- **Total Raw Papers**: {self.search_metadata['total_raw']}
- **After Deduplication**: {self.search_metadata['total_raw'] - self.search_metadata['duplicates_removed']} papers
- **High-Quality Filtered**: {len(papers)} papers

### Query Performance Analysis

**Most Productive Queries** (by relevant papers found):
"""
        
        # Add top performing queries
        for i, (query, count) in enumerate(query_performance['top_performing_queries'], 1):
            report_content += f"\n{i}. `{query}` - {count} relevant papers"
        
        report_content += f"""

**Search Efficiency**:
- Average papers per query: {query_performance['avg_papers_per_query']:.1f}
- Successful queries: {query_performance['successful_queries']}/{query_performance['total_queries']} ({100*query_performance['successful_queries']/query_performance['total_queries']:.1f}%)
- Execution time: {self.search_metadata['execution_time']}

## Paper Collection Statistics

### Temporal Distribution
"""
        
        # Year distribution
        for year in sorted(year_dist.keys(), reverse=True):
            count = year_dist[year]
            percentage = (count / len(papers)) * 100
            report_content += f"- **{year}**: {count} papers ({percentage:.1f}%)\n"
        
        report_content += f"""
### Top Publication Venues
"""
        
        # Venue distribution
        for venue, count in top_venues[:10]:
            percentage = (count / len(papers)) * 100
            report_content += f"- **{venue}**: {count} papers ({percentage:.1f}%)\n"
        
        report_content += f"""
### Citation Impact Analysis
- **Total citations**: {stats.get('total_citations', 0):,}
- **Average citations per paper**: {stats.get('avg_citations', 0):.1f}
- **Papers with 100+ citations**: {len([p for p in papers if p.get('citations', 0) >= 100])}
- **Papers with 50+ citations**: {len([p for p in papers if p.get('citations', 0) >= 50])}

### Content Quality Metrics
- **Papers with abstracts**: {len(papers)} (100% - required)
- **Papers with DOI**: {stats.get('papers_with_doi', 0)} ({100*stats.get('papers_with_doi', 0)/len(papers):.1f}%)
- **Papers with URLs**: {stats.get('papers_with_url', 0)} ({100*stats.get('papers_with_url', 0)/len(papers):.1f}%)
- **Average relevance score**: {sum(p.get('relevance_score', 0) for p in papers)/len(papers):.2f}

## High-Impact Paper Highlights

### Most Cited Papers
"""
        
        # Top cited papers
        for i, paper in enumerate(top_cited, 1):
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            year = paper.get('year', 'Unknown')
            citations = paper.get('citations', 0)
            
            author_str = self._format_authors(authors)
            report_content += f"\n{i}. **{title}**  \n   *{author_str} ({year})* - {citations:,} citations\n"
        
        report_content += f"""
### Highest Relevance Papers
"""
        
        # Top relevant papers
        for i, paper in enumerate(top_relevant, 1):
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            year = paper.get('year', 'Unknown')
            score = paper.get('relevance_score', 0)
            
            author_str = self._format_authors(authors)
            report_content += f"\n{i}. **{title}**  \n   *{author_str} ({year})* - Relevance Score: {score:.2f}\n"
        
        report_content += f"""
### Recent High-Impact Work (2023-2024)
"""
        
        # Recent papers
        for i, paper in enumerate(recent_sorted, 1):
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            year = paper.get('year', 'Unknown')
            score = paper.get('relevance_score', 0)
            citations = paper.get('citations', 0)
            
            author_str = self._format_authors(authors)
            report_content += f"\n{i}. **{title}**  \n   *{author_str} ({year})* - Score: {score:.2f}, Citations: {citations}\n"
        
        report_content += f"""

## Quality Assurance

### Applied Filters
All papers in this collection meet these criteria:
"""
        
        for filter_desc in self.search_metadata['quality_filters']:
            report_content += f"- {filter_desc.replace('_', ' ').title()}\n"
        
        report_content += f"""
### Deduplication Process
- **Duplicate removal method**: Title similarity (90% threshold) + DOI matching
- **Duplicates removed**: {self.search_metadata['duplicates_removed']} papers
- **Deduplication efficiency**: {100*self.search_metadata['duplicates_removed']/self.search_metadata['total_raw']:.1f}% duplicate rate

### Content Validation
- **Abstract quality**: Minimum 20 words, substantive content
- **English language**: 85% ASCII character threshold + English word indicators
- **Research quality**: Methodological and experimental terminology required
- **Domain relevance**: Multi-term matching for synthetic data context

## Search Query Catalog

### All {len(self.search_metadata['queries_used'])} Queries Used:
"""
        
        for i, query in enumerate(self.search_metadata['queries_used'], 1):
            report_content += f"\n{i}. `{query}`"
        
        report_content += f"""

---

## Technical Details

**Search Configuration**:
- Target papers: {self.target_papers}
- Papers per query: {self.papers_per_query}
- Minimum citations: {self.min_citations} (age-adjusted)
- Relevance threshold: Dynamic scoring
- API rate limits: Semantic Scholar (1.2s), arXiv (2.5s)

**Data Sources**:
- Semantic Scholar Graph API v1
- arXiv API with `arxiv` Python client
- Cached responses with 24h TTL

**Output Format**:
- JSON structure with comprehensive metadata
- Individual paper records with relevance scores
- Full search provenance and statistics

---

*Report generated by Agentic Auto-Survey System*  
*Comprehensive Synthetic Data Literature Search v2.0*
"""
        
        # Save the report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

    def _format_authors(self, authors: List[str], max_authors: int = 3) -> str:
        """Format author list for display"""
        if not authors:
            return "Unknown Authors"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        elif len(authors) <= max_authors:
            return ", ".join(authors[:-1]) + f" & {authors[-1]}"
        else:
            return f"{authors[0]} et al."


def main():
    """Main execution function"""
    
    # Set output file path as requested
    output_file = "/path/to/project/output-multi-agent-v2/synthetic_data_papers.json"
    
    logger.info("Starting Comprehensive Synthetic Data Paper Search")
    logger.info(f"Output file: {output_file}")
    
    # Initialize searcher
    searcher = ComprehensiveSyntheticDataSearcher(output_file)
    
    # Execute comprehensive search
    papers = searcher.execute_comprehensive_search()
    
    # Save results with comprehensive metadata
    searcher.save_comprehensive_results(papers)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA PAPER SEARCH COMPLETED")
    print(f"{'='*60}")
    print(f"Papers collected: {len(papers)}")
    print(f"Year range: {searcher.year_from}-{searcher.year_to}")
    print(f"Search queries used: {len(searcher.search_metadata['queries_used'])}")
    print(f"Sources: Semantic Scholar, arXiv")
    print(f"Execution time: {searcher.search_metadata['execution_time']}")
    print(f"\nOutput files:")
    print(f"  📄 Papers JSON: {output_file}")
    print(f"  📊 Search Report: {Path(output_file).with_suffix('.md')}")
    print(f"\nCollection covers:")
    print(f"  ✓ Synthetic data generation techniques")
    print(f"  ✓ Privacy-preserving methods")
    print(f"  ✓ GANs & diffusion models")
    print(f"  ✓ LLM training applications")
    print(f"  ✓ Quality evaluation metrics")
    print(f"  ✓ Domain-specific applications")


if __name__ == "__main__":
    main()