"""
Result formatting utilities for literature search results.

This module provides formatting and presentation utilities for search results,
including markdown generation, citation formatting, and summary creation.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from ..models.literature_models import Paper, LiteratureResult
from ..models.query_models import KeywordResult
from ..utils.text_utils import highlight_keywords_in_text, truncate_text, format_author_names


class ResultFormatter:
    """
    Format literature search results for various output formats.
    
    Provides methods to format results as markdown, plain text, JSON,
    and other presentation formats suitable for different use cases.
    """
    
    def __init__(self):
        pass
    
    def format_as_markdown(
        self, 
        result: LiteratureResult,
        include_abstracts: bool = True,
        highlight_keywords: bool = True
    ) -> str:
        """
        Format search results as markdown.
        
        Args:
            result: Literature search results
            include_abstracts: Whether to include paper abstracts
            highlight_keywords: Whether to highlight keywords in text
            
        Returns:
            Formatted markdown string
        """
        md_lines = []
        
        # Header
        md_lines.append(f"# Literature Search Results")
        md_lines.append(f"")
        md_lines.append(f"**Query:** {result.original_query}")
        md_lines.append(f"**Results:** {len(result.papers)} papers found")
        md_lines.append(f"**Processing Time:** {result.processing_time:.2f}s")
        
        if result.search_strategies_used:
            md_lines.append(f"**Search Strategies:** {', '.join(result.search_strategies_used)}")
        
        md_lines.append("")
        
        # Keywords section
        if result.extracted_keywords:
            md_lines.append("## Extracted Keywords")
            md_lines.append("")
            
            keywords = result.extracted_keywords
            if hasattr(keywords, 'primary_keywords') and keywords.primary_keywords:
                md_lines.append(f"**Primary:** {', '.join(keywords.primary_keywords)}")
            
            if hasattr(keywords, 'secondary_keywords') and keywords.secondary_keywords:
                md_lines.append(f"**Secondary:** {', '.join(keywords.secondary_keywords)}")
            
            if hasattr(keywords, 'confidence_score'):
                md_lines.append(f"**Confidence:** {keywords.confidence_score:.2f}")
            
            md_lines.append("")
        
        # Papers section
        if result.papers:
            md_lines.append("## Papers")
            md_lines.append("")
            
            for i, paper in enumerate(result.papers, 1):
                md_lines.extend(self._format_paper_markdown(
                    paper, i, include_abstracts, highlight_keywords,
                    result.extracted_keywords
                ))
                md_lines.append("")
        
        # Recommendations
        if result.recommendations:
            md_lines.append("## Recommendations")
            md_lines.append("")
            for rec in result.recommendations:
                md_lines.append(f"- {rec}")
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _format_paper_markdown(
        self,
        paper: Paper,
        index: int,
        include_abstracts: bool,
        highlight_keywords: bool,
        extracted_keywords: Any
    ) -> List[str]:
        """Format a single paper as markdown."""
        lines = []
        
        # Paper title and basic info
        lines.append(f"### {index}. {paper.title}")
        lines.append("")
        
        # Authors
        author_str = format_author_names([a.name for a in paper.authors])
        lines.append(f"**Authors:** {author_str}")
        
        # Basic metadata
        lines.append(f"**ArXiv ID:** [{paper.arxiv_id}]({paper.abstract_url})")
        lines.append(f"**Published:** {paper.published_date.strftime('%Y-%m-%d')}")
        lines.append(f"**Categories:** {', '.join(paper.categories)}")
        
        if paper.relevance_score > 0:
            lines.append(f"**Relevance Score:** {paper.relevance_score:.3f}")
        
        # Journal reference if available
        if paper.journal_ref:
            lines.append(f"**Journal:** {paper.journal_ref}")
        
        # DOI if available
        if paper.doi:
            lines.append(f"**DOI:** {paper.doi}")
        
        lines.append("")
        
        # Abstract
        if include_abstracts and paper.abstract:
            lines.append("**Abstract:**")
            
            abstract_text = paper.abstract
            
            # Highlight keywords if requested
            if highlight_keywords and extracted_keywords:
                if hasattr(extracted_keywords, 'all_keywords'):
                    keywords = extracted_keywords.all_keywords
                elif hasattr(extracted_keywords, 'primary_keywords'):
                    keywords = extracted_keywords.primary_keywords
                else:
                    keywords = []
                
                if keywords:
                    abstract_text = highlight_keywords_in_text(abstract_text, keywords)
            
            lines.append(abstract_text)
            lines.append("")
        
        # Links
        lines.append("**Links:**")
        lines.append(f"- [Abstract]({paper.abstract_url})")
        lines.append(f"- [PDF]({paper.pdf_url})")
        
        # Keywords matched
        if paper.keywords_matched:
            lines.append(f"- **Matched Keywords:** {', '.join(paper.keywords_matched)}")
        
        return lines
    
    def format_as_plain_text(
        self,
        result: LiteratureResult,
        include_abstracts: bool = True,
        max_abstract_length: int = 200
    ) -> str:
        """Format search results as plain text."""
        
        lines = []
        
        # Header
        lines.append("LITERATURE SEARCH RESULTS")
        lines.append("=" * 50)
        lines.append(f"Query: {result.original_query}")
        lines.append(f"Results: {len(result.papers)} papers found")
        lines.append(f"Processing time: {result.processing_time:.2f}s")
        lines.append("")
        
        # Papers
        for i, paper in enumerate(result.papers, 1):
            lines.append(f"{i}. {paper.title}")
            lines.append(f"   Authors: {format_author_names([a.name for a in paper.authors])}")
            lines.append(f"   ArXiv ID: {paper.arxiv_id}")
            lines.append(f"   Published: {paper.published_date.strftime('%Y-%m-%d')}")
            lines.append(f"   Categories: {', '.join(paper.categories)}")
            
            if paper.relevance_score > 0:
                lines.append(f"   Relevance: {paper.relevance_score:.3f}")
            
            if include_abstracts and paper.abstract:
                abstract = truncate_text(paper.abstract, max_abstract_length)
                lines.append(f"   Abstract: {abstract}")
            
            lines.append(f"   PDF: {paper.pdf_url}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_as_json(self, result: LiteratureResult) -> Dict[str, Any]:
        """Format search results as JSON-serializable dictionary."""
        
        # Convert result to dictionary
        result_dict = {
            "query": result.original_query,
            "papers": [],
            "total_found": result.total_found,
            "processing_time": result.processing_time,
            "strategies_used": result.search_strategies_used,
            "timestamp": result.timestamp.isoformat() if hasattr(result, 'timestamp') else datetime.now().isoformat()
        }
        
        # Add keywords if available
        if result.extracted_keywords:
            keywords = result.extracted_keywords
            keywords_dict = {}
            
            if hasattr(keywords, 'primary_keywords'):
                keywords_dict["primary"] = keywords.primary_keywords
            
            if hasattr(keywords, 'secondary_keywords'):
                keywords_dict["secondary"] = keywords.secondary_keywords
            
            if hasattr(keywords, 'confidence_score'):
                keywords_dict["confidence_score"] = keywords.confidence_score
            
            if hasattr(keywords, 'domain_category'):
                keywords_dict["domain_category"] = keywords.domain_category
            
            result_dict["extracted_keywords"] = keywords_dict
        
        # Add papers
        for paper in result.papers:
            paper_dict = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": [
                    {
                        "name": author.name,
                        "affiliation": author.affiliation
                    }
                    for author in paper.authors
                ],
                "abstract": paper.abstract,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "published_date": paper.published_date.isoformat(),
                "updated_date": paper.updated_date.isoformat(),
                "pdf_url": paper.pdf_url,
                "abstract_url": paper.abstract_url,
                "relevance_score": paper.relevance_score
            }
            
            # Optional fields
            if paper.doi:
                paper_dict["doi"] = paper.doi
            
            if paper.journal_ref:
                paper_dict["journal_ref"] = paper.journal_ref
            
            if paper.comments:
                paper_dict["comments"] = paper.comments
            
            if paper.keywords_matched:
                paper_dict["keywords_matched"] = paper.keywords_matched
            
            result_dict["papers"].append(paper_dict)
        
        # Add recommendations
        if result.recommendations:
            result_dict["recommendations"] = result.recommendations
        
        return result_dict
    
    def format_as_bibtex(self, papers: List[Paper]) -> str:
        """Format papers as BibTeX entries."""
        
        bibtex_entries = []
        
        for paper in papers:
            # Generate BibTeX entry
            entry_key = f"arxiv{paper.arxiv_id.replace('.', '_').replace('/', '_')}"
            
            # Format authors for BibTeX
            authors_bibtex = " and ".join([author.name for author in paper.authors])
            
            year = paper.published_date.year
            
            # Build BibTeX entry
            bibtex = f"""@article{{{entry_key},
    title={{{paper.title}}},
    author={{{authors_bibtex}}},
    journal={{arXiv preprint arXiv:{paper.arxiv_id}}},
    year={{{year}}},
    url={{{paper.abstract_url}}}"""
            
            # Add optional fields
            if paper.doi:
                bibtex += f",\n    doi={{{paper.doi}}}"
            
            if paper.journal_ref:
                bibtex += f",\n    note={{{paper.journal_ref}}}"
            
            bibtex += "\n}"
            
            bibtex_entries.append(bibtex)
        
        return "\n\n".join(bibtex_entries)
    
    def create_summary_report(self, result: LiteratureResult) -> str:
        """Create a concise summary report of the search results."""
        
        lines = []
        
        # Basic statistics
        lines.append("LITERATURE SEARCH SUMMARY")
        lines.append("=" * 30)
        lines.append(f"Query: {result.original_query}")
        lines.append(f"Total papers found: {len(result.papers)}")
        lines.append(f"Processing time: {result.processing_time:.2f}s")
        lines.append("")
        
        if result.papers:
            # Relevance statistics
            relevance_scores = [p.relevance_score for p in result.papers if p.relevance_score > 0]
            if relevance_scores:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                lines.append(f"Average relevance score: {avg_relevance:.3f}")
            
            # Top categories
            category_counts = {}
            for paper in result.papers:
                for category in paper.categories:
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            if category_counts:
                top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                lines.append("Top categories:")
                for category, count in top_categories[:5]:
                    lines.append(f"  - {category}: {count} papers")
            
            lines.append("")
            
            # Top papers
            lines.append("Top 3 most relevant papers:")
            for i, paper in enumerate(result.papers[:3], 1):
                lines.append(f"{i}. {paper.title}")
                lines.append(f"   Authors: {format_author_names([a.name for a in paper.authors])}")
                lines.append(f"   Relevance: {paper.relevance_score:.3f}")
                lines.append("")
        
        # Recommendations
        if result.recommendations:
            lines.append("Recommendations:")
            for rec in result.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def format_for_export(
        self,
        result: LiteratureResult,
        format_type: str = "csv"
    ) -> str:
        """Format results for data export (CSV, TSV, etc.)."""
        
        if format_type.lower() == "csv":
            return self._format_as_csv(result.papers)
        elif format_type.lower() == "tsv":
            return self._format_as_csv(result.papers, delimiter="\t")
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _format_as_csv(self, papers: List[Paper], delimiter: str = ",") -> str:
        """Format papers as CSV/TSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        
        # Header
        writer.writerow([
            "ArXiv ID", "Title", "Authors", "Primary Category", "All Categories",
            "Published Date", "Abstract", "PDF URL", "DOI", "Relevance Score"
        ])
        
        # Data rows
        for paper in papers:
            authors_str = "; ".join([author.name for author in paper.authors])
            categories_str = "; ".join(paper.categories)
            
            writer.writerow([
                paper.arxiv_id,
                paper.title,
                authors_str,
                paper.primary_category,
                categories_str,
                paper.published_date.strftime("%Y-%m-%d"),
                paper.abstract,
                paper.pdf_url,
                paper.doi or "",
                paper.relevance_score
            ])
        
        return output.getvalue()


# Usage example
if __name__ == "__main__":
    from datetime import datetime
    from ..models.literature_models import Paper, Author, LiteratureResult
    from ..models.query_models import KeywordResult
    
    # Create sample data for testing
    authors = [Author("John Doe", "University"), Author("Jane Smith", "Institute")]
    
    paper = Paper(
        arxiv_id="2301.12345",
        title="Sample Paper on Machine Learning",
        authors=authors,
        abstract="This is a sample abstract about machine learning applications.",
        categories=["cs.LG", "cs.AI"],
        primary_category="cs.LG",
        published_date=datetime(2023, 1, 15),
        updated_date=datetime(2023, 1, 15),
        pdf_url="http://arxiv.org/pdf/2301.12345.pdf",
        abstract_url="http://arxiv.org/abs/2301.12345",
        relevance_score=0.85
    )
    
    keywords = KeywordResult(
        primary_keywords=["machine learning", "deep learning"],
        secondary_keywords=["neural networks", "artificial intelligence"],
        confidence_score=0.8
    )
    
    result = LiteratureResult(
        original_query="machine learning applications",
        extracted_keywords=keywords,
        papers=[paper],
        total_found=1,
        search_strategies_used=["primary_title_focused"],
        processing_time=1.5,
        recommendations=["Consider exploring deep learning approaches"]
    )
    
    # Test formatting
    formatter = ResultFormatter()
    
    print("=== MARKDOWN FORMAT ===")
    print(formatter.format_as_markdown(result))
    
    print("\n=== PLAIN TEXT FORMAT ===")
    print(formatter.format_as_plain_text(result))
    
    print("\n=== SUMMARY REPORT ===")
    print(formatter.create_summary_report(result))