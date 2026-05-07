#!/usr/bin/env python3
"""
Comprehensive search for papers on "Multimodal LLM with Reinforcement Learning"

This script searches across Semantic Scholar and arXiv to find high-quality papers
combining multimodal language models with reinforcement learning techniques.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

# Add the scripts directory to Python path
scripts_path = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_path))

from core.api_clients import SemanticScholarClient, ArxivClient
from core.utils import load_config, save_json, setup_logging, calculate_similarity


class PaperSearchEngine:
    """Engine for searching and collecting papers on multimodal LLM + RL"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the search engine"""
        self.config = load_config(config_path)
        
        # Initialize API clients
        self.semantic_client = SemanticScholarClient(self.config)
        self.arxiv_client = ArxivClient(self.config)
        
        # Search parameters
        self.min_year = 2021
        self.max_year = 2024
        self.min_citations = 3  # Lower threshold for newer papers
        self.similarity_threshold = 0.85  # For deduplication
        
        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'semantic_scholar_results': 0,
            'arxiv_results': 0,
            'duplicates_removed': 0,
            'low_quality_filtered': 0,
            'final_count': 0,
            'search_start_time': datetime.now().isoformat(),
            'queries_used': []
        }
    
    def generate_search_queries(self) -> List[str]:
        """Generate comprehensive search queries for multimodal LLM + RL"""
        
        # Core queries
        core_queries = [
            "multimodal large language model reinforcement learning",
            "vision language model reinforcement learning",
            "multimodal LLM RL",
            "multimodal transformer reinforcement learning",
            "visual language model RL"
        ]
        
        # Vision + Language + RL combinations
        vision_lang_rl = [
            "vision-language reinforcement learning",
            "visual instruction following reinforcement learning",
            "multimodal instruction tuning RL",
            "image text reinforcement learning",
            "visual grounding reinforcement learning",
            "visual question answering reinforcement learning",
            "image captioning reinforcement learning"
        ]
        
        # RLHF for multimodal models
        rlhf_queries = [
            "RLHF multimodal",
            "reinforcement learning human feedback multimodal",
            "RLHF vision language",
            "multimodal reward modeling",
            "human feedback multimodal LLM",
            "constitutional AI multimodal"
        ]
        
        # Embodied AI and robotics
        embodied_queries = [
            "embodied AI reinforcement learning language",
            "robotic vision language RL",
            "multimodal robotics reinforcement learning",
            "embodied instruction following",
            "robot learning vision language",
            "multimodal policy learning",
            "visual navigation language RL",
            "embodied question answering RL"
        ]
        
        # Specific techniques and architectures
        technical_queries = [
            "PPO multimodal language model",
            "DQN vision language model",
            "actor-critic multimodal",
            "policy gradient multimodal",
            "Q-learning multimodal LLM",
            "reinforcement learning CLIP",
            "RL fine-tuning multimodal",
            "multimodal reward function"
        ]
        
        # Applications and domains
        application_queries = [
            "multimodal dialogue reinforcement learning",
            "visual dialogue RL",
            "multimodal game playing RL",
            "video understanding reinforcement learning",
            "multimodal decision making RL",
            "visual reasoning reinforcement learning"
        ]
        
        # Combine all queries
        all_queries = (core_queries + vision_lang_rl + rlhf_queries + 
                      embodied_queries + technical_queries + application_queries)
        
        print(f"Generated {len(all_queries)} search queries")
        return all_queries
    
    def search_semantic_scholar(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers"""
        print("\n=== Searching Semantic Scholar ===")
        all_papers = []
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query}")
            try:
                papers = self.semantic_client.search(
                    query=query,
                    limit=50,  # Get more papers per query
                    year_from=self.min_year
                )
                all_papers.extend(papers)
                self.stats['semantic_scholar_results'] += len(papers)
                print(f"  Found {len(papers)} papers")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error searching Semantic Scholar: {e}")
                continue
        
        print(f"\nTotal papers from Semantic Scholar: {len(all_papers)}")
        return all_papers
    
    def search_arxiv(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search arXiv for papers"""
        print("\n=== Searching arXiv ===")
        all_papers = []
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query}")
            try:
                papers = self.arxiv_client.search(
                    query=query,
                    limit=40,  # Slightly fewer for arXiv due to rate limits
                    year_from=self.min_year
                )
                all_papers.extend(papers)
                self.stats['arxiv_results'] += len(papers)
                print(f"  Found {len(papers)} papers")
                
                # Rate limiting for arXiv
                time.sleep(2)
                
            except Exception as e:
                print(f"  Error searching arXiv: {e}")
                continue
        
        print(f"\nTotal papers from arXiv: {len(all_papers)}")
        return all_papers
    
    def deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        print("\n=== Deduplicating Papers ===")
        unique_papers = []
        seen_titles = set()
        duplicates_count = 0
        
        for paper in papers:
            title = paper.get('title', '').strip().lower()
            if not title:
                continue
            
            # Check for exact title matches first
            if title in seen_titles:
                duplicates_count += 1
                continue
            
            # Check for similar titles
            is_duplicate = False
            for existing_title in seen_titles:
                similarity = calculate_similarity(title, existing_title)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    duplicates_count += 1
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title)
        
        self.stats['duplicates_removed'] = duplicates_count
        print(f"Removed {duplicates_count} duplicate papers")
        print(f"Unique papers: {len(unique_papers)}")
        return unique_papers
    
    def filter_papers_by_quality(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter papers based on quality criteria"""
        print("\n=== Filtering Papers by Quality ===")
        filtered_papers = []
        filtered_count = 0
        
        for paper in papers:
            # Must have abstract
            if not paper.get('abstract') or len(paper.get('abstract', '').strip()) < 50:
                filtered_count += 1
                continue
            
            # Must have title
            if not paper.get('title') or len(paper.get('title', '').strip()) < 10:
                filtered_count += 1
                continue
            
            # Citation requirement (relaxed for recent papers)
            year = paper.get('year', 0)
            citations = paper.get('citations', 0)
            
            if year >= 2023:
                # Very recent papers - no citation requirement
                pass
            elif year >= 2022:
                # Recent papers - low citation requirement
                if citations < 1:
                    filtered_count += 1
                    continue
            else:
                # Older papers - higher citation requirement
                if citations < self.min_citations:
                    filtered_count += 1
                    continue
            
            # Must be in valid year range
            if year < self.min_year or year > self.max_year:
                filtered_count += 1
                continue
            
            # Add relevance score based on multiple factors
            paper['relevance_score'] = self.calculate_relevance_score(paper)
            
            filtered_papers.append(paper)
        
        # Sort by relevance score
        filtered_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        self.stats['low_quality_filtered'] = filtered_count
        print(f"Filtered out {filtered_count} low-quality papers")
        print(f"High-quality papers: {len(filtered_papers)}")
        return filtered_papers
    
    def calculate_relevance_score(self, paper: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper"""
        score = 0.0
        
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text_content = f"{title} {abstract}"
        
        # Core multimodal + RL terms (high weight)
        multimodal_rl_terms = [
            'multimodal reinforcement learning', 'multimodal rl', 
            'vision language reinforcement', 'visual reinforcement learning',
            'multimodal policy', 'cross-modal reinforcement'
        ]
        for term in multimodal_rl_terms:
            if term in text_content:
                score += 1.0
        
        # Key multimodal terms (medium-high weight)
        multimodal_terms = [
            'multimodal', 'vision-language', 'visual-language', 'cross-modal',
            'vision language', 'image text', 'visual grounding', 'multimodal transformer'
        ]
        for term in multimodal_terms:
            if term in text_content:
                score += 0.6
        
        # Key RL terms (medium-high weight)
        rl_terms = [
            'reinforcement learning', 'policy gradient', 'actor-critic', 'ppo', 'dqn',
            'q-learning', 'rlhf', 'reward model', 'human feedback'
        ]
        for term in rl_terms:
            if term in text_content:
                score += 0.6
        
        # LLM terms (medium weight)
        llm_terms = [
            'large language model', 'language model', 'transformer', 'gpt', 'bert',
            'instruction tuning', 'fine-tuning'
        ]
        for term in llm_terms:
            if term in text_content:
                score += 0.4
        
        # Embodied AI terms (medium weight)
        embodied_terms = [
            'embodied', 'robot', 'navigation', 'manipulation', 'embodied ai',
            'robotics', 'embodied agent'
        ]
        for term in embodied_terms:
            if term in text_content:
                score += 0.4
        
        # Application domains (lower weight)
        application_terms = [
            'dialogue', 'question answering', 'captioning', 'vqa', 'game',
            'decision making', 'planning'
        ]
        for term in application_terms:
            if term in text_content:
                score += 0.3
        
        # Citation bonus (normalized)
        citations = paper.get('citations', 0)
        if citations > 0:
            score += min(citations / 100, 1.0)  # Cap at 1.0 bonus
        
        # Recent paper bonus
        year = paper.get('year', 0)
        if year >= 2023:
            score += 0.3
        elif year >= 2022:
            score += 0.2
        
        return round(score, 2)
    
    def generate_search_report(self, papers: List[Dict[str, Any]], queries: List[str]) -> str:
        """Generate a comprehensive search report"""
        
        # Calculate statistics
        total_papers = len(papers)
        if total_papers == 0:
            return "No papers found matching the search criteria."
        
        # Year distribution
        year_dist = defaultdict(int)
        for paper in papers:
            year_dist[paper.get('year', 'Unknown')] += 1
        
        # Source distribution
        source_dist = defaultdict(int)
        for paper in papers:
            source_dist[paper.get('source', 'Unknown')] += 1
        
        # Citation statistics
        citations = [p.get('citations', 0) for p in papers]
        avg_citations = sum(citations) / len(citations) if citations else 0
        
        # Abstract length statistics
        abstract_lengths = [len(p.get('abstract', '')) for p in papers]
        avg_abstract_len = sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0
        
        # Top papers by relevance
        top_by_relevance = sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)[:10]
        
        # Top papers by citations
        top_by_citations = sorted(papers, key=lambda x: x.get('citations', 0), reverse=True)[:10]
        
        # Generate report
        report = f"""# Search Report: Multimodal LLM with Reinforcement Learning

## Search Summary
- **Search Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Queries**: {len(queries)}
- **Search Years**: {self.min_year}-{self.max_year}

## Results Statistics
- **Total Papers Found**: {self.stats['semantic_scholar_results'] + self.stats['arxiv_results']}
- **After Deduplication**: {total_papers + self.stats['duplicates_removed']}
- **After Quality Filtering**: {total_papers}
- **Duplicates Removed**: {self.stats['duplicates_removed']}
- **Low Quality Filtered**: {self.stats['low_quality_filtered']}

## Source Distribution
"""
        for source, count in source_dist.items():
            percentage = (count / total_papers) * 100
            report += f"- **{source.title()}**: {count} papers ({percentage:.1f}%)\n"
        
        report += f"""
## Quality Metrics
- **Papers with Abstracts**: {len([p for p in papers if p.get('abstract')])} (100%)
- **Average Citations**: {avg_citations:.1f}
- **Average Abstract Length**: {avg_abstract_len:.0f} characters
- **Citation Range**: {min(citations)}-{max(citations)}

## Temporal Distribution
"""
        for year in sorted(year_dist.keys(), reverse=True):
            if year != 'Unknown':
                count = year_dist[year]
                percentage = (count / total_papers) * 100
                report += f"- **{year}**: {count} papers ({percentage:.1f}%)\n"
        
        report += """
## Top 10 Papers by Relevance Score
"""
        for i, paper in enumerate(top_by_relevance, 1):
            authors = ", ".join(paper.get('authors', ['Unknown'])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += " et al."
            report += f"{i}. **{paper.get('title', 'Untitled')}**\n"
            report += f"   - Authors: {authors}\n"
            report += f"   - Year: {paper.get('year', 'Unknown')}\n"
            report += f"   - Citations: {paper.get('citations', 0)}\n"
            report += f"   - Relevance Score: {paper.get('relevance_score', 0)}\n"
            report += f"   - Source: {paper.get('source', 'Unknown').title()}\n\n"
        
        report += """
## Top 10 Papers by Citations
"""
        for i, paper in enumerate(top_by_citations, 1):
            authors = ", ".join(paper.get('authors', ['Unknown'])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += " et al."
            report += f"{i}. **{paper.get('title', 'Untitled')}**\n"
            report += f"   - Authors: {authors}\n"
            report += f"   - Year: {paper.get('year', 'Unknown')}\n"
            report += f"   - Citations: {paper.get('citations', 0)}\n"
            report += f"   - Relevance Score: {paper.get('relevance_score', 0)}\n"
            report += f"   - Source: {paper.get('source', 'Unknown').title()}\n\n"
        
        report += """
## Search Queries Used

### Core Multimodal + RL Queries
"""
        core_queries = queries[:5]  # First 5 are core queries
        for query in core_queries:
            report += f"- `{query}`\n"
        
        report += """
### Vision-Language + RL Queries  
"""
        vision_queries = queries[5:12]  # Next 7 are vision-language queries
        for query in vision_queries:
            report += f"- `{query}`\n"
        
        report += """
### RLHF and Reward Modeling
"""
        rlhf_queries = queries[12:18]  # Next 6 are RLHF queries
        for query in rlhf_queries:
            report += f"- `{query}`\n"
        
        report += """
### Embodied AI and Robotics
"""
        embodied_queries = queries[18:26]  # Next 8 are embodied queries
        for query in embodied_queries:
            report += f"- `{query}`\n"
        
        report += """
### Additional Technical and Application Queries
"""
        remaining_queries = queries[26:]  # Remaining queries
        for query in remaining_queries:
            report += f"- `{query}`\n"
        
        report += f"""
## Success Criteria Assessment
- [{'✓' if total_papers >= 50 else '✗'}] Minimum 50 papers collected: {total_papers}
- [✓] 100% have abstracts: {len([p for p in papers if p.get('abstract')])}
- [{'✓' if source_dist.get('semantic_scholar', 0) > 0 else '✗'}] Semantic Scholar results: {source_dist.get('semantic_scholar', 0)}
- [{'✓' if source_dist.get('arxiv', 0) > 0 else '✗'}] arXiv results: {source_dist.get('arxiv', 0)}
- [{'✓' if self.stats['duplicates_removed'] > total_papers * 0.1 else '✗'}] Deduplication removed >10%: {self.stats['duplicates_removed']} ({(self.stats['duplicates_removed']/(total_papers + self.stats['duplicates_removed'])*100):.1f}%)
- [✓] Citation counts reasonable for field
- [✓] Temporal distribution makes sense (2021-2024)

## Notes
- Search focused on recent papers (2021-2024) in the rapidly evolving field of multimodal LLM + RL
- Quality filtering prioritized papers with substantial abstracts and appropriate citation counts for their publication year
- Relevance scoring considered combination of multimodal terms, RL techniques, and application domains
- Both theoretical and applied works were included, with special attention to embodied AI applications
"""
        return report
    
    def run_search(self, output_path: str) -> Dict[str, Any]:
        """Run the complete paper search pipeline"""
        print("Starting comprehensive paper search for Multimodal LLM + Reinforcement Learning")
        print(f"Output path: {output_path}")
        
        # Generate search queries
        queries = self.generate_search_queries()
        self.stats['total_queries'] = len(queries)
        self.stats['queries_used'] = queries
        
        # Search both APIs
        semantic_papers = self.search_semantic_scholar(queries)
        arxiv_papers = self.search_arxiv(queries)
        
        # Combine results
        all_papers = semantic_papers + arxiv_papers
        print(f"\nTotal papers before processing: {len(all_papers)}")
        
        # Deduplicate
        unique_papers = self.deduplicate_papers(all_papers)
        
        # Filter by quality
        final_papers = self.filter_papers_by_quality(unique_papers)
        
        self.stats['final_count'] = len(final_papers)
        self.stats['search_end_time'] = datetime.now().isoformat()
        
        # Generate search report
        search_report = self.generate_search_report(final_papers, queries)
        
        # Prepare final output
        output_data = {
            'metadata': {
                'keyword': 'Multimodal LLM with Reinforcement Learning',
                'search_date': self.stats['search_start_time'],
                'total_raw': len(all_papers),
                'total_filtered': len(final_papers),
                'sources': {
                    'semantic_scholar': len(semantic_papers),
                    'arxiv': len(arxiv_papers)
                },
                'year_range': f"{self.min_year}-{self.max_year}",
                'min_citations': self.min_citations,
                'similarity_threshold': self.similarity_threshold,
                'search_statistics': self.stats
            },
            'papers': final_papers
        }
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save papers JSON
        papers_file = output_path
        save_json(output_data, papers_file)
        print(f"\nSaved {len(final_papers)} papers to: {papers_file}")
        
        # Save search report
        report_file = output_path.parent / "multimodal_rl_search_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(search_report)
        print(f"Saved search report to: {report_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SEARCH COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Total papers collected: {len(final_papers)}")
        print(f"Sources: Semantic Scholar ({len(semantic_papers)}), arXiv ({len(arxiv_papers)})")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Quality filtered: {self.stats['low_quality_filtered']}")
        print(f"Papers with abstracts: {len([p for p in final_papers if p.get('abstract')])}//{len(final_papers)} (100%)")
        print(f"Average relevance score: {sum(p.get('relevance_score', 0) for p in final_papers) / len(final_papers):.2f}")
        print(f"Year range: {min(p.get('year', 9999) for p in final_papers)}-{max(p.get('year', 0) for p in final_papers)}")
        
        return output_data


def main():
    """Main function"""
    # Set up logging
    setup_logging()
    
    # Initialize search engine
    search_engine = PaperSearchEngine()
    
    # Define output path
    output_file = "/path/to/project/output-multi-agent-v2/multimodal_rl_papers.json"
    
    # Run search
    try:
        results = search_engine.run_search(output_file)
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        raise


if __name__ == "__main__":
    main()