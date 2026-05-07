#!/usr/bin/env python3
"""
Simple processor for RLHF alignment paper search results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import Counter

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def calculate_simple_relevance(paper: Dict[str, Any]) -> float:
    """Calculate simple relevance score based on key RLHF terms."""
    title = paper.get('title', '').lower()
    abstract = paper.get('abstract', '').lower()
    combined = title + ' ' + abstract
    
    # Key RLHF terms with weights
    key_terms = {
        'rlhf': 3.0,
        'reinforcement learning from human feedback': 3.0,
        'human feedback': 2.5,
        'preference': 2.0,
        'constitutional ai': 2.5,
        'alignment': 2.0,
        'ppo': 2.0,
        'dpo': 2.5,
        'reward model': 2.0,
        'instructgpt': 2.0,
        'red team': 1.5
    }
    
    score = 0.0
    for term, weight in key_terms.items():
        if term in combined:
            score += weight
    
    return min(1.0, score / 10.0)

def filter_and_process(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and process papers with simple criteria."""
    filtered = []
    seen_titles = set()
    
    for paper in papers:
        # Basic filtering
        if not paper.get('title') or not paper.get('abstract'):
            continue
        
        year = paper.get('year', 0)
        if year < 2020 or year > 2024:
            continue
            
        # Simple deduplication by title
        title_key = paper['title'].lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        
        # Calculate relevance
        relevance = calculate_simple_relevance(paper)
        paper['relevance_score'] = relevance
        
        # Check if from major lab
        authors = paper.get('authors', [])
        authors_text = ' '.join(authors).lower()
        major_labs = ['openai', 'anthropic', 'deepmind', 'meta', 'stanford', 'berkeley']
        paper['is_major_lab'] = any(lab in authors_text for lab in major_labs)
        
        filtered.append(paper)
    
    # Sort by relevance score
    filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return filtered

def generate_simple_stats(papers: List[Dict]) -> Dict[str, Any]:
    """Generate simple statistics."""
    if not papers:
        return {}
    
    years = [p.get('year') for p in papers if p.get('year')]
    citations = [p.get('citations', 0) for p in papers]
    major_lab_count = sum(1 for p in papers if p.get('is_major_lab'))
    
    year_counts = Counter(years)
    
    return {
        'total_papers': len(papers),
        'year_distribution': dict(year_counts),
        'avg_citations': round(sum(citations) / len(citations), 1),
        'max_citations': max(citations) if citations else 0,
        'major_lab_papers': major_lab_count,
        'papers_with_abstracts': len(papers),  # All filtered papers have abstracts
        'top_papers': [
            {
                'title': p['title'][:100] + '...' if len(p['title']) > 100 else p['title'],
                'year': p.get('year'),
                'citations': p.get('citations', 0),
                'relevance_score': round(p.get('relevance_score', 0), 3)
            }
            for p in papers[:10]
        ]
    }

def main():
    """Main processing function."""
    print("Processing RLHF alignment papers...")
    
    # Load intermediate results
    input_path = "/path/to/project/output-multi-agent-v2/rlhf_alignment_papers_intermediate.json"
    print(f"Loading data from: {input_path}")
    
    data = load_json(input_path)
    all_papers = data['semantic_papers']
    print(f"Loaded {len(all_papers)} raw papers")
    
    # Process papers
    print("Filtering and processing papers...")
    processed_papers = filter_and_process(all_papers)
    print(f"Processed to {len(processed_papers)} high-quality papers")
    
    # Generate statistics
    print("Generating statistics...")
    stats = generate_simple_stats(processed_papers)
    
    # Prepare final result
    result = {
        'metadata': {
            'keyword': 'RLHF Alignment',
            'search_date': datetime.now().isoformat(),
            'total_raw': len(all_papers),
            'total_filtered': len(processed_papers),
            'sources': {
                'semantic_scholar': len(all_papers),
                'arxiv': 0
            }
        },
        'papers': processed_papers,
        'statistics': stats
    }
    
    # Save results
    output_path = "/path/to/project/output-multi-agent-v2/rlhf_alignment_papers.json"
    save_json(result, output_path)
    print(f"Results saved to: {output_path}")
    
    # Generate simple report
    report = f"""# RLHF Alignment Paper Search Report

## Overview
- **Total Papers**: {stats['total_papers']}
- **Date Range**: 2020-2024
- **Average Citations**: {stats['avg_citations']}
- **Papers from Major Labs**: {stats['major_lab_papers']}

## Year Distribution
"""
    
    for year, count in sorted(stats['year_distribution'].items()):
        report += f"- {year}: {count} papers\n"
    
    report += f"""
## Top Papers by Relevance

"""
    
    for i, paper in enumerate(stats['top_papers'][:5], 1):
        report += f"{i}. **{paper['title']}**\n"
        report += f"   - Year: {paper['year']}, Citations: {paper['citations']}, Relevance: {paper['relevance_score']}\n\n"
    
    # Save report
    report_path = "/path/to/project/output-multi-agent-v2/rlhf_alignment_search_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Final dataset: {stats['total_papers']} papers")
    
    return result

if __name__ == "__main__":
    main()