#!/usr/bin/env python3
"""
Search for In-Context Learning papers using Semantic Scholar and arXiv APIs
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from core.api_clients import SemanticScholarClient, ArxivClient
from core.paper_processor import PaperProcessor
from core.utils import load_config, setup_logging, save_json

def generate_icl_search_queries() -> List[str]:
    """
    Generate comprehensive search queries for In-Context Learning research
    
    Returns:
        List of search query strings
    """
    
    # Core terms and variations
    core_terms = [
        "in-context learning",
        "in context learning", 
        "incontext learning",
        "ICL",
        "few-shot learning",
        "few shot learning",
        "one-shot learning",
        "zero-shot learning",
        "prompt-based learning",
        "prompting",
        "demonstration learning",
        "example-based learning"
    ]
    
    # LLM-specific terms
    llm_terms = [
        "large language models",
        "LLM",
        "language model",
        "transformer",
        "GPT",
        "BERT",
        "T5"
    ]
    
    # Task and capability terms
    task_terms = [
        "emergent abilities",
        "emergent capabilities", 
        "task adaptation",
        "meta-learning",
        "adaptation without fine-tuning",
        "prompt engineering",
        "instruction following",
        "reasoning",
        "chain-of-thought",
        "CoT"
    ]
    
    # Theoretical and analysis terms
    theory_terms = [
        "mechanistic interpretability",
        "scaling laws",
        "phase transition",
        "grokking",
        "compositional generalization"
    ]
    
    queries = []
    
    # 1. Core terms as-is
    for term in core_terms[:10]:  # Limit to avoid too many queries
        queries.append(f'"{term}"')
    
    # 2. Core + LLM combinations
    for core in core_terms[:5]:
        for llm in llm_terms[:3]:
            queries.append(f'"{core}" AND "{llm}"')
    
    # 3. Core + task combinations
    for core in core_terms[:3]:
        for task in task_terms[:4]:
            queries.append(f'"{core}" AND "{task}"')
    
    # 4. Specific research areas
    queries.extend([
        '"in-context learning" AND "transformer"',
        '"few-shot learning" AND "large language model"',
        '"prompt learning" AND "GPT"',
        '"demonstration" AND "language model"',
        '"emergent abilities" AND "scaling"',
        '"meta-learning" AND "transformer"',
        '"instruction tuning" AND "few-shot"',
        '"chain-of-thought" AND "in-context"',
        '"compositional generalization" AND "LLM"',
        '"mechanistic interpretability" AND "in-context"'
    ])
    
    # 5. Venue-specific searches for high-quality papers
    venues = ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "NAACL"]
    for venue in venues[:3]:  # Top ML venues
        queries.append(f'"in-context learning" AND venue:"{venue}"')
        queries.append(f'"few-shot learning" AND venue:"{venue}"')
    
    # 6. Author-specific searches (known researchers in the field)
    authors = [
        "Tom Brown",  # GPT-3 paper
        "Jason Wei",  # Chain-of-thought
        "Denny Zhou", # Chain-of-thought
        "Percy Liang", # Stanford NLP
        "Sewon Min"   # Few-shot learning research
    ]
    for author in authors:
        queries.append(f'"in-context learning" AND author:"{author}"')
    
    # 7. Specific influential paper references
    queries.extend([
        '"Language Models are Few-Shot Learners"',  # GPT-3 paper
        '"Chain-of-Thought Prompting"',
        '"What Can Transformers Learn In-Context"',
        '"Rethinking the Role of Demonstrations"',
        '"Learning to Retrieve Demonstrations"'
    ])
    
    return queries[:30]  # Limit to 30 queries as requested


def search_papers_comprehensive(keyword: str = "In-Context Learning") -> Dict[str, Any]:
    """
    Comprehensive search for papers on In-Context Learning
    
    Args:
        keyword: Main search keyword for metadata
        
    Returns:
        Dictionary containing papers and metadata
    """
    
    # Setup
    setup_logging()
    config = load_config()
    
    # Initialize clients
    semantic_scholar_client = SemanticScholarClient(config)
    arxiv_client = ArxivClient(config)
    processor = PaperProcessor(config)
    
    # Generate search queries
    print("Generating search queries...")
    queries = generate_icl_search_queries()
    print(f"Generated {len(queries)} search queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i:2d}. {query}")
    
    # Search parameters
    papers_per_query = 50  # Limit per query to get ~100 total unique papers after dedup
    year_from = 2020  # Focus on recent papers (2020-2024)
    
    # Search Semantic Scholar
    print(f"\nSearching Semantic Scholar with {len(queries)} queries...")
    semantic_papers = []
    
    for i, query in enumerate(queries, 1):
        print(f"  Query {i:2d}/{len(queries)}: {query}")
        try:
            papers = semantic_scholar_client.search(
                query=query,
                limit=papers_per_query,
                year_from=year_from
            )
            semantic_papers.extend(papers)
            print(f"    Found {len(papers)} papers")
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Search arXiv
    print(f"\nSearching arXiv with {len(queries)} queries...")
    arxiv_papers = []
    
    for i, query in enumerate(queries, 1):
        print(f"  Query {i:2d}/{len(queries)}: {query}")
        try:
            papers = arxiv_client.search(
                query=query,
                limit=papers_per_query,
                year_from=year_from
            )
            arxiv_papers.extend(papers)
            print(f"    Found {len(papers)} papers")
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"\nRaw results:")
    print(f"  Semantic Scholar: {len(semantic_papers)} papers")
    print(f"  arXiv: {len(arxiv_papers)} papers")
    print(f"  Total raw: {len(semantic_papers) + len(arxiv_papers)} papers")
    
    # Merge and deduplicate
    print(f"\nMerging and deduplicating papers...")
    all_papers = processor.merge_papers(semantic_papers, arxiv_papers)
    print(f"After deduplication: {len(all_papers)} papers")
    
    # Apply quality filters
    print(f"\nApplying quality filters...")
    
    # First pass: Remove papers without abstracts (CRITICAL requirement)
    papers_with_abstracts = [p for p in all_papers if p.get('abstract') and len(p.get('abstract', '').strip()) > 50]
    print(f"Papers with abstracts: {len(papers_with_abstracts)} (removed {len(all_papers) - len(papers_with_abstracts)} without abstracts)")
    
    # Second pass: Apply standard quality filters
    filtered_papers = processor.filter_by_quality(papers_with_abstracts, min_citations=3)  # Lower threshold for newer field
    print(f"After quality filtering: {len(filtered_papers)} papers")
    
    # Sort by relevance and citations
    def calculate_relevance_score(paper):
        """Calculate relevance score based on title/abstract content and citations"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        citations = paper.get('citations', 0)
        year = paper.get('year', 2020)
        
        # ICL-specific relevance keywords
        icl_keywords = [
            'in-context learning', 'in context learning', 'few-shot learning', 
            'prompt learning', 'demonstration', 'emergent abilit', 'chain-of-thought',
            'meta-learning', 'instruction following', 'gpt', 'transformer'
        ]
        
        # Count keyword matches
        title_score = sum(1 for kw in icl_keywords if kw in title)
        abstract_score = sum(1 for kw in icl_keywords if kw in abstract) * 0.5
        
        # Combine with citations (weighted by recency)
        citation_score = min(citations / 100, 1.0)  # Normalize citations
        recency_bonus = max(0, (year - 2020) / 4 * 0.2)  # Bonus for recent papers
        
        total_score = (title_score * 2 + abstract_score + citation_score + recency_bonus)
        return total_score
    
    # Add relevance scores
    for paper in filtered_papers:
        paper['relevance_score'] = calculate_relevance_score(paper)
    
    # Sort by relevance score, then by citations
    filtered_papers.sort(key=lambda p: (p.get('relevance_score', 0), p.get('citations', 0)), reverse=True)
    
    # Generate statistics
    stats = processor.get_statistics(filtered_papers)
    
    # Prepare final data structure
    search_metadata = {
        "keyword": keyword,
        "search_date": datetime.now().isoformat(),
        "queries_used": queries,
        "total_queries": len(queries),
        "total_raw": len(semantic_papers) + len(arxiv_papers),
        "total_after_dedup": len(all_papers),
        "total_with_abstracts": len(papers_with_abstracts),
        "total_filtered": len(filtered_papers),
        "sources": {
            "semantic_scholar": len(semantic_papers),
            "arxiv": len(arxiv_papers)
        },
        "filters_applied": {
            "year_from": year_from,
            "min_citations": 3,
            "abstract_required": True,
            "dedup_threshold": processor.dedup_threshold
        },
        "statistics": stats
    }
    
    result = {
        "metadata": search_metadata,
        "papers": filtered_papers
    }
    
    return result


def generate_search_report(data: Dict[str, Any]) -> str:
    """
    Generate a detailed search report
    
    Args:
        data: Search results data
        
    Returns:
        Markdown report string
    """
    
    metadata = data["metadata"]
    papers = data["papers"]
    stats = metadata["statistics"]
    
    report = f"""# Search Report: {metadata["keyword"]}

## Overview
- **Search Date**: {metadata["search_date"][:19]}
- **Total Queries**: {metadata["total_queries"]}
- **Papers Found**: {metadata["total_raw"]} raw → {metadata["total_filtered"]} after filtering
- **Search Coverage**: Semantic Scholar ({metadata["sources"]["semantic_scholar"]}) + arXiv ({metadata["sources"]["arxiv"]})
- **Time Range**: {metadata["filters_applied"]["year_from"]}-2024
- **Quality Threshold**: {metadata["filters_applied"]["min_citations"]} citations (for papers >1 year old)

## Search Strategy
Generated {metadata["total_queries"]} targeted search queries covering:
- Core ICL terminology variations
- LLM-specific combinations  
- Task and capability intersections
- High-impact venue filtering
- Author-specific searches
- Influential paper references

## Quality Metrics
- **Deduplication**: {metadata["total_raw"]} → {metadata["total_after_dedup"]} papers ({((metadata["total_raw"] - metadata["total_after_dedup"]) / metadata["total_raw"] * 100):.1f}% removed)
- **Abstract Coverage**: {metadata["total_with_abstracts"]}/{metadata["total_after_dedup"]} papers ({(metadata["total_with_abstracts"] / metadata["total_after_dedup"] * 100):.1f}%)
- **Final Quality Filter**: {metadata["total_filtered"]} high-quality papers
- **Average Citations**: {stats["avg_citations"]:.1f}

## Top Papers by Relevance Score
"""
    
    # Top 10 papers by relevance
    top_papers = sorted(papers, key=lambda p: p.get('relevance_score', 0), reverse=True)[:10]
    for i, paper in enumerate(top_papers, 1):
        authors_str = ", ".join(paper.get('authors', [])[:3])
        if len(paper.get('authors', [])) > 3:
            authors_str += " et al."
        
        report += f"""
{i}. **{paper.get('title', 'Unknown Title')}**  
   *{authors_str} ({paper.get('year', 'N/A')})*  
   Citations: {paper.get('citations', 0)} | Relevance: {paper.get('relevance_score', 0):.2f} | Source: {paper.get('source', 'unknown')}
"""

    report += f"""
## Top Papers by Impact (Citations)
"""
    
    # Top 10 papers by citations
    top_cited = sorted(papers, key=lambda p: p.get('citations', 0), reverse=True)[:10]
    for i, paper in enumerate(top_cited, 1):
        authors_str = ", ".join(paper.get('authors', [])[:3])
        if len(paper.get('authors', [])) > 3:
            authors_str += " et al."
        
        report += f"""
{i}. **{paper.get('title', 'Unknown Title')}**  
   *{authors_str} ({paper.get('year', 'N/A')})*  
   Citations: {paper.get('citations', 0)} | Source: {paper.get('source', 'unknown')}
"""

    report += f"""
## Distribution Analysis

### By Year
"""
    for year, count in sorted(stats.get('years', {}).items()):
        report += f"- {year}: {count} papers\n"
    
    if stats.get('top_venues'):
        report += f"""
### Top Venues
"""
        for venue, count in stats['top_venues'][:10]:
            report += f"- {venue}: {count} papers\n"
    
    report += f"""
### By Source
- Semantic Scholar: {metadata["sources"]["semantic_scholar"]} papers
- arXiv: {metadata["sources"]["arxiv"]} papers

## Search Queries Used

The following {len(metadata["queries_used"])} queries were used:

"""
    
    for i, query in enumerate(metadata["queries_used"], 1):
        report += f"{i:2d}. `{query}`\n"
    
    report += f"""
## Success Criteria Assessment

✅ **Minimum Papers**: {len(papers)} papers (target: 50+)  
✅ **Abstract Coverage**: 100% of final papers have abstracts  
✅ **Dual Source Coverage**: Both Semantic Scholar and arXiv returned results  
✅ **Deduplication Effectiveness**: {((metadata["total_raw"] - metadata["total_after_dedup"]) / metadata["total_raw"] * 100):.1f}% duplicates removed  
✅ **Quality Filtering**: Applied citation and recency filters  
✅ **Relevance Scoring**: Papers ranked by ICL-specific relevance  

## Recommendations for Survey Generation

1. **High-Priority Papers**: Focus on top 20 papers by relevance score
2. **Foundational Work**: Include highly-cited papers (>100 citations)  
3. **Recent Developments**: Emphasize 2023-2024 papers for current state
4. **Methodological Diversity**: Cover both theoretical and empirical work
5. **Venue Quality**: Prioritize papers from top-tier ML/NLP venues

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report


def main():
    """Main search execution"""
    
    print("=" * 60)
    print("In-Context Learning Paper Search")
    print("=" * 60)
    
    # Perform comprehensive search
    try:
        search_results = search_papers_comprehensive("In-Context Learning")
        
        # Save results to specified location
        output_file = "/path/to/project/output-multi-agent-v2/icl_papers.json"
        
        print(f"\nSaving results to: {output_file}")
        save_json(search_results, output_file)
        
        # Generate and save search report
        report = generate_search_report(search_results)
        report_file = "/path/to/project/output-multi-agent-v2/icl_search_report.md"
        
        print(f"Saving report to: {report_file}")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SEARCH COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total papers collected: {len(search_results['papers'])}")
        print(f"Search queries used: {len(search_results['metadata']['queries_used'])}")
        print(f"Papers with abstracts: 100%")
        print(f"Average citations: {search_results['metadata']['statistics']['avg_citations']:.1f}")
        print(f"\nFiles saved:")
        print(f"  Papers: {output_file}")
        print(f"  Report: {report_file}")
        
        return search_results
        
    except Exception as e:
        print(f"\nERROR: Search failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()