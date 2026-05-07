#!/usr/bin/env python3
"""
Process the intermediate RLHF alignment paper search results and create final output.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from difflib import SequenceMatcher

# Add the scripts directory to Python path
scripts_path = Path(__file__).parent / "scripts"
import sys
sys.path.insert(0, str(scripts_path))

from core.utils import load_config, save_json, setup_logging, calculate_similarity

def load_intermediate_results(filepath: str) -> Dict[str, Any]:
    """Load the intermediate results from the search."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_relevance_score(paper: Dict[str, Any], query_terms: Set[str]) -> float:
    """
    Calculate relevance score based on RLHF and alignment term presence in title and abstract.
    """
    title = paper.get('title', '').lower()
    abstract = paper.get('abstract', '').lower()
    
    # Combine title and abstract with higher weight on title
    combined_text = title + " " + abstract
    
    # RLHF and alignment specific terms with high weights
    rlhf_terms = {
        'rlhf': 3.0,
        'reinforcement learning from human feedback': 3.0, 
        'human feedback': 2.5,
        'preference modeling': 2.5,
        'constitutional ai': 2.5,
        'alignment': 2.0,
        'ppo': 2.0,
        'dpo': 2.0,
        'direct preference optimization': 2.5,
        'red teaming': 2.0,
        'ai safety': 2.0,
        'reward modeling': 2.0,
        'policy optimization': 1.5,
        'instructgpt': 2.0,
        'chatgpt': 1.5,
        'anthropic': 1.5,
        'openai': 1.5
    }
    
    # Calculate weighted score
    score = 0.0
    total_weight = 0.0
    
    # Check for RLHF-specific high-value terms
    for term, weight in rlhf_terms.items():
        if term in combined_text:
            score += weight
            total_weight += weight
    
    # Check for general query terms
    for term in query_terms:
        term_lower = term.lower()
        if term_lower in title:
            score += 2.0
            total_weight += 2.0
        elif term_lower in abstract:
            score += 1.0
            total_weight += 1.0
    
    # Normalize score
    if total_weight > 0:
        score = min(1.0, score / max(total_weight, 5.0))
    
    return score

def is_major_ai_lab(authors: List[str], venue: str) -> bool:
    """
    Check if paper is from major AI labs (OpenAI, Anthropic, DeepMind, Meta).
    """
    if not authors:
        return False
        
    # Convert authors to single string for easier searching
    authors_text = ' '.join(authors).lower()
    venue_text = venue.lower() if venue else ''
    
    major_labs = [
        'openai', 'anthropic', 'deepmind', 'google deepmind',
        'meta ai', 'facebook ai', 'microsoft research',
        'google research', 'google brain', 'nvidia research',
        'stanford', 'berkeley', 'mit', 'cmu', 'university of california'
    ]
    
    for lab in major_labs:
        if lab in authors_text or lab in venue_text:
            return True
            
    return False

def filter_papers_by_criteria(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply RLHF-specific filtering criteria to papers.
    """
    if not papers:
        return papers
    
    filtered_papers = []
    current_year = datetime.now().year
    
    for paper in papers:
        # Must have abstract - this is critical for RLHF papers
        if not paper.get('abstract'):
            continue
            
        # Must have title
        if not paper.get('title'):
            continue
        
        # Year filtering (2020-2024 focus)
        year = paper.get('year', 0)
        if year < 2020 or year > current_year:
            continue
        
        # For RLHF field, even papers with few citations can be valuable if recent
        citations = paper.get('citations', 0)
        if year < current_year - 1:  # Papers older than 1 year
            # Lower citation threshold for RLHF due to field novelty
            if citations < 3:
                continue
        
        # Boost papers from major AI labs
        authors = paper.get('authors', [])
        venue = paper.get('venue', '')
        if is_major_ai_lab(authors, venue):
            paper['is_major_lab'] = True
        else:
            paper['is_major_lab'] = False
            
        filtered_papers.append(paper)
    
    return filtered_papers

def deduplicate_papers(papers: List[Dict[str, Any]], threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Remove duplicate papers based on title similarity.
    """
    if not papers:
        return papers
    
    unique_papers = []
    seen_titles = []
    
    for paper in papers:
        title = paper.get('title', '').lower().strip()
        if not title:
            continue
            
        # Check similarity with existing titles
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title, seen_title).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_papers.append(paper)
            seen_titles.append(title)
    
    return unique_papers

def generate_statistics(papers: List[Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive statistics about the search results.
    """
    if not papers:
        return {}
    
    # Year distribution
    year_dist = {}
    citation_counts = []
    venue_dist = {}
    major_lab_count = 0
    
    for paper in papers:
        year = paper.get('year')
        if year:
            year_dist[str(year)] = year_dist.get(str(year), 0) + 1
        
        citations = paper.get('citations', 0)
        citation_counts.append(citations)
        
        venue = paper.get('venue')
        if venue:
            venue_dist[venue] = venue_dist.get(venue, 0) + 1
            
        if paper.get('is_major_lab'):
            major_lab_count += 1
    
    # Calculate citation statistics
    avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0
    max_citations = max(citation_counts) if citation_counts else 0
    
    # Top papers by citations
    top_by_citations = sorted(papers, 
                             key=lambda x: x.get('citations', 0), 
                             reverse=True)[:10]
    
    # Top papers by relevance
    top_by_relevance = sorted(papers,
                             key=lambda x: x.get('relevance_score', 0),
                             reverse=True)[:10]
    
    # Abstract statistics
    abstract_lengths = []
    papers_with_abstracts = 0
    
    for paper in papers:
        abstract = paper.get('abstract', '')
        if abstract:
            papers_with_abstracts += 1
            abstract_lengths.append(len(abstract.split()))
    
    avg_abstract_length = sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0
    
    return {
        'total_papers': len(papers),
        'year_distribution': dict(sorted(year_dist.items())),
        'citation_stats': {
            'average': round(avg_citations, 1),
            'maximum': max_citations,
            'total': sum(citation_counts)
        },
        'venue_distribution': dict(sorted(venue_dist.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]),
        'major_lab_papers': major_lab_count,
        'abstract_stats': {
            'papers_with_abstracts': papers_with_abstracts,
            'coverage_percentage': round(100 * papers_with_abstracts / len(papers), 1),
            'average_length_words': round(avg_abstract_length, 1)
        },
        'top_by_citations': [
            {
                'title': p.get('title', '')[:100] + ('...' if len(p.get('title', '')) > 100 else ''),
                'citations': p.get('citations', 0),
                'year': p.get('year'),
                'authors': p.get('authors', [])[:3]  # First 3 authors
            }
            for p in top_by_citations
        ],
        'top_by_relevance': [
            {
                'title': p.get('title', '')[:100] + ('...' if len(p.get('title', '')) > 100 else ''),
                'relevance_score': round(p.get('relevance_score', 0), 3),
                'year': p.get('year'),
                'authors': p.get('authors', [])[:3]  # First 3 authors
            }
            for p in top_by_relevance
        ]
    }

def generate_search_report(result: Dict[str, Any], output_path: str) -> None:
    """
    Generate a comprehensive search report in Markdown format.
    """
    metadata = result['metadata']
    papers = result['papers']
    stats = result['statistics']
    
    report = f"""# RLHF Alignment Paper Search Report

## Search Overview
- **Search Topic**: {metadata['keyword']}
- **Search Date**: {metadata['search_date'][:10]}
- **Total Queries**: {len(metadata.get('queries_used', []))}
- **Raw Papers Found**: {metadata['total_raw']}
- **Final Filtered Papers**: {metadata['total_filtered']}
- **Success Rate**: {round(100 * metadata['total_filtered'] / max(metadata['total_raw'], 1), 1)}%

## Data Sources
- **Semantic Scholar**: {metadata['sources']['semantic_scholar']} papers

## Quality Metrics
- **Papers with Abstracts**: {stats['abstract_stats']['papers_with_abstracts']} ({stats['abstract_stats']['coverage_percentage']}%)
- **Average Abstract Length**: {stats['abstract_stats']['average_length_words']} words
- **Papers from Major AI Labs**: {stats['major_lab_papers']}
- **Date Range**: 2020-2024
- **Average Citations**: {stats['citation_stats']['average']}

## Year Distribution
"""
    
    for year, count in stats['year_distribution'].items():
        percentage = round(100 * count / len(papers), 1)
        report += f"- **{year}**: {count} papers ({percentage}%)\n"
    
    report += f"""
## Top Papers by Citations
"""
    
    for i, paper in enumerate(stats['top_by_citations'][:5], 1):
        authors_str = ', '.join(paper['authors']) if paper['authors'] else 'Unknown'
        report += f"{i}. **{paper['title']}**\n   - Authors: {authors_str}\n   - Year: {paper['year']}\n   - Citations: {paper['citations']}\n\n"
    
    report += f"""
## Top Papers by Relevance Score
"""
    
    for i, paper in enumerate(stats['top_by_relevance'][:5], 1):
        authors_str = ', '.join(paper['authors']) if paper['authors'] else 'Unknown'
        report += f"{i}. **{paper['title']}**\n   - Authors: {authors_str}\n   - Year: {paper['year']}\n   - Relevance Score: {paper['relevance_score']}\n\n"
    
    report += f"""
## Top Venues
"""
    
    for venue, count in list(stats['venue_distribution'].items())[:10]:
        percentage = round(100 * count / len(papers), 1)
        report += f"- **{venue}**: {count} papers ({percentage}%)\n"
    
    report += f"""

## Filter Criteria Applied
- **Year Range**: 2020-2024
- **Minimum Citations**: 3 (for papers > 1 year old)
- **Abstract Required**: True
- **Deduplication Threshold**: 0.9 (title similarity)

## Summary
This search successfully identified {metadata['total_filtered']} high-quality papers on RLHF alignment from {metadata['total_raw']} initial results from Semantic Scholar. The papers span from 2020 to 2024, with {stats['abstract_stats']['coverage_percentage']}% having complete abstracts. {stats['major_lab_papers']} papers are from major AI research labs, indicating good coverage of cutting-edge research in this rapidly evolving field.
"""
    
    # Save report
    report_path = Path(output_path).parent / "rlhf_alignment_search_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Search report saved to: {report_path}")

def main():
    """Main processing function"""
    
    # Setup
    setup_logging()
    output_dir = Path("/path/to/project/output-multi-agent-v2")
    
    # Load intermediate results
    intermediate_path = output_dir / "rlhf_alignment_papers_intermediate.json"
    print(f"Loading intermediate results from: {intermediate_path}")
    
    data = load_intermediate_results(str(intermediate_path))
    all_papers = data['semantic_papers']
    
    print(f"Loaded {len(all_papers)} papers from Semantic Scholar")
    
    # Collect all query terms for relevance scoring
    query_terms = set(['rlhf', 'reinforcement', 'learning', 'human', 'feedback', 
                      'preference', 'modeling', 'alignment', 'constitutional', 'ai',
                      'ppo', 'dpo', 'optimization', 'red', 'teaming', 'safety'])
    
    # Deduplicate
    print("Removing duplicates...")
    unique_papers = deduplicate_papers(all_papers)
    print(f"Papers after deduplication: {len(unique_papers)}")
    
    # Apply quality filters
    print("Applying quality filters...")
    filtered_papers = filter_papers_by_criteria(unique_papers)
    print(f"Papers after quality filtering: {len(filtered_papers)}")
    
    # Calculate relevance scores
    print("Calculating relevance scores...")
    for paper in filtered_papers:
        score = calculate_relevance_score(paper, query_terms)
        paper['relevance_score'] = score
    
    # Sort by relevance score
    filtered_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Generate statistics
    print("Generating statistics...")
    stats = generate_statistics(filtered_papers)
    
    # Prepare final dataset
    result = {
        'metadata': {
            'keyword': 'RLHF Alignment',
            'search_date': datetime.now().isoformat(),
            'total_raw': len(all_papers),
            'total_filtered': len(filtered_papers),
            'sources': {
                'semantic_scholar': len(all_papers),
                'arxiv': 0  # Not collected yet in intermediate
            },
            'filter_criteria': {
                'year_range': '2020-2024',
                'min_citations': 3,
                'require_abstract': True,
                'dedup_threshold': 0.9
            }
        },
        'papers': filtered_papers,
        'statistics': stats
    }
    
    # Save final results
    output_path = output_dir / "rlhf_alignment_papers.json"
    save_json(result, str(output_path))
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Papers saved to: {output_path}")
    print(f"Total papers found: {result['metadata']['total_filtered']}")
    print(f"Success rate: {round(100 * result['metadata']['total_filtered'] / max(result['metadata']['total_raw'], 1), 1)}%")
    
    # Generate search report
    generate_search_report(result, str(output_path))
    
    return result

if __name__ == "__main__":
    result = main()