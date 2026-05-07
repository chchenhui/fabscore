"""
Literature and paper data models for the literature search system.

This module defines data structures for representing research papers, authors,
ArXiv responses, and complete literature query results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


@dataclass
class Author:
    """
    Author information for research papers.
    
    Represents author details including name and optional affiliation information.
    """
    name: str
    affiliation: Optional[str] = None
    
    def __post_init__(self):
        """Validate and clean author data."""
        if not self.name or not self.name.strip():
            raise ValueError("Author name cannot be empty")
        
        # Clean up name formatting
        self.name = self.name.strip()
        
        # Clean up affiliation if provided
        if self.affiliation:
            self.affiliation = self.affiliation.strip()
            if not self.affiliation:
                self.affiliation = None
    
    @property
    def display_name(self) -> str:
        """Get formatted display name."""
        return self.name
    
    @property
    def has_affiliation(self) -> bool:
        """Check if author has affiliation information."""
        return self.affiliation is not None and len(self.affiliation) > 0


@dataclass
class Paper:
    """
    Research paper metadata and content.
    
    Comprehensive representation of a research paper from ArXiv including
    bibliographic information, content, and enhanced metadata.
    """
    arxiv_id: str
    title: str
    authors: List[Author]
    abstract: str
    categories: List[str]
    primary_category: str
    published_date: datetime
    updated_date: datetime
    pdf_url: str
    abstract_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comments: Optional[str] = None
    
    # Enhanced fields
    summary: Optional[str] = None  # LLM-generated summary
    relevance_score: float = 0.0   # Computed relevance to query
    keywords_matched: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and process paper data."""
        # Required field validation
        if not self.arxiv_id or not self.title:
            raise ValueError("Paper must have arxiv_id and title")
        
        if not self.authors:
            raise ValueError("Paper must have at least one author")
        
        if not self.abstract:
            raise ValueError("Paper must have an abstract")
        
        # Clean and validate data
        self.title = self.title.strip()
        self.abstract = self.abstract.strip()
        
        # Validate ArXiv ID format
        if not self._validate_arxiv_id(self.arxiv_id):
            raise ValueError(f"Invalid ArXiv ID format: {self.arxiv_id}")
        
        # Ensure categories is not empty
        if not self.categories:
            raise ValueError("Paper must have at least one category")
        
        # Validate primary category is in categories
        if self.primary_category not in self.categories:
            self.categories.insert(0, self.primary_category)
        
        # Validate relevance score
        if self.relevance_score < 0.0 or self.relevance_score > 1.0:
            self.relevance_score = max(0.0, min(1.0, self.relevance_score))
    
    def _validate_arxiv_id(self, arxiv_id: str) -> bool:
        """Validate ArXiv ID format."""
        # Modern format: YYMM.NNNNN (e.g., 2301.12345)
        modern_pattern = r'^\d{4}\.\d{4,5}$'
        
        # Legacy format: subject-class/YYMMnnn (e.g., cs.AI/0301012)
        legacy_pattern = r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$'
        
        return (re.match(modern_pattern, arxiv_id) is not None or 
                re.match(legacy_pattern, arxiv_id) is not None)
    
    @property
    def author_names(self) -> List[str]:
        """Get list of author names."""
        return [author.name for author in self.authors]
    
    @property
    def primary_author(self) -> Author:
        """Get the primary (first) author."""
        return self.authors[0] if self.authors else None
    
    @property
    def author_count(self) -> int:
        """Get number of authors."""
        return len(self.authors)
    
    def is_recent(self, days: int = 30) -> bool:
        """Check if paper was published recently (within specified days)."""
        from datetime import datetime, timedelta, timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Convert published_date to UTC if it has timezone info
        pub_date = self.published_date
        if pub_date.tzinfo is not None:
            pub_date = pub_date.astimezone(timezone.utc)
        else:
            # If published_date is naive, assume it's UTC
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        
        return pub_date >= cutoff_date
    
    @property
    def abstract_word_count(self) -> int:
        """Get word count of abstract."""
        return len(self.abstract.split()) if self.abstract else 0
    
    @property
    def has_journal_publication(self) -> bool:
        """Check if paper has been published in a journal."""
        return bool(self.journal_ref)
    
    @property
    def has_doi(self) -> bool:
        """Check if paper has a DOI."""
        return bool(self.doi)
    
    def get_citation_text(self, style: str = "arxiv") -> str:
        """
        Generate citation text for the paper.
        
        Args:
            style: Citation style ("arxiv", "bibtex", "apa")
            
        Returns:
            Formatted citation string
        """
        if style.lower() == "arxiv":
            authors_str = ", ".join(self.author_names)
            year = self.published_date.year
            return f"{authors_str}. \"{self.title}\". arXiv preprint arXiv:{self.arxiv_id} ({year})."
        
        elif style.lower() == "bibtex":
            # Generate BibTeX entry
            authors_bibtex = " and ".join(self.author_names)
            year = self.published_date.year
            entry_key = f"arxiv{self.arxiv_id.replace('.', '_')}"
            
            bibtex = f"""@article{{{entry_key},
    title={{{self.title}}},
    author={{{authors_bibtex}}},
    journal={{arXiv preprint arXiv:{self.arxiv_id}}},
    year={{{year}}}
}}"""
            return bibtex
        
        elif style.lower() == "apa":
            # APA style citation
            authors_apa = self._format_authors_apa()
            year = self.published_date.year
            return f"{authors_apa} ({year}). {self.title}. arXiv preprint arXiv:{self.arxiv_id}."
        
        else:
            raise ValueError(f"Unsupported citation style: {style}")
    
    def _format_authors_apa(self) -> str:
        """Format authors in APA style."""
        if not self.authors:
            return ""
        
        if len(self.authors) == 1:
            return self.authors[0].name
        elif len(self.authors) == 2:
            return f"{self.authors[0].name} & {self.authors[1].name}"
        else:
            # More than 2 authors - use first author + et al.
            return f"{self.authors[0].name} et al."


@dataclass
class ArXivResponse:
    """
    Raw ArXiv API response data.
    
    Represents the complete response from an ArXiv API query including
    papers, pagination information, and request metadata.
    """
    papers: List[Paper]
    total_results: int
    start_index: int
    items_per_page: int
    query_url: str
    response_time: float
    
    def __post_init__(self):
        """Validate ArXiv response data."""
        if self.total_results < 0:
            raise ValueError("total_results cannot be negative")
        
        if self.start_index < 0:
            raise ValueError("start_index cannot be negative")
        
        if self.items_per_page <= 0:
            raise ValueError("items_per_page must be positive")
        
        if self.response_time < 0:
            raise ValueError("response_time cannot be negative")
        
        # Validate that papers count doesn't exceed items_per_page
        if len(self.papers) > self.items_per_page:
            raise ValueError("Number of papers exceeds items_per_page")
    
    @property
    def has_more_results(self) -> bool:
        """Check if there are more results available."""
        return (self.start_index + len(self.papers)) < self.total_results
    
    @property
    def next_start_index(self) -> int:
        """Get the start index for the next page."""
        return self.start_index + len(self.papers)


@dataclass
class LiteratureResult:
    """
    Complete literature query results.
    
    Comprehensive results from a literature query including the original query,
    extracted keywords, found papers, and processing metadata.
    """
    original_query: str
    extracted_keywords: Optional[Any]  # KeywordResult from query_models
    papers: List[Paper]
    total_found: int
    search_strategies_used: List[str]
    processing_time: float
    cache_hit: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate literature result data."""
        if not self.original_query or not self.original_query.strip():
            raise ValueError("original_query cannot be empty")
        
        if self.total_found < 0:
            raise ValueError("total_found cannot be negative")
        
        if self.processing_time < 0:
            raise ValueError("processing_time cannot be negative")
        
        # Validate that papers count doesn't exceed total_found
        if len(self.papers) > self.total_found:
            self.total_found = len(self.papers)
    
    @property
    def top_papers(self) -> List[Paper]:
        """Get top 10 most relevant papers sorted by relevance score."""
        return sorted(
            self.papers, 
            key=lambda p: p.relevance_score, 
            reverse=True
        )[:10]
    
    @property
    def recent_papers(self) -> List[Paper]:
        """Get papers published in the last 30 days."""
        return [paper for paper in self.papers if paper.is_recent(30)]
    
    @property
    def paper_count(self) -> int:
        """Get number of papers in results."""
        return len(self.papers)
    
    @property
    def unique_authors(self) -> List[str]:
        """Get list of unique author names across all papers."""
        authors = set()
        for paper in self.papers:
            authors.update(paper.author_names)
        return sorted(list(authors))
    
    @property
    def category_distribution(self) -> Dict[str, int]:
        """Get distribution of papers by primary category."""
        distribution = {}
        for paper in self.papers:
            category = paper.primary_category
            distribution[category] = distribution.get(category, 0) + 1
        
        # Sort by count (descending)
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    @property
    def average_relevance_score(self) -> float:
        """Calculate average relevance score of all papers."""
        if not self.papers:
            return 0.0
        
        total_score = sum(paper.relevance_score for paper in self.papers)
        return total_score / len(self.papers)
    
    def get_papers_by_category(self, category: str) -> List[Paper]:
        """Get all papers in a specific category."""
        return [paper for paper in self.papers if category in paper.categories]
    
    def get_papers_above_threshold(self, relevance_threshold: float = 0.7) -> List[Paper]:
        """Get papers with relevance score above threshold."""
        return [
            paper for paper in self.papers 
            if paper.relevance_score >= relevance_threshold
        ]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert result to summary dictionary for serialization."""
        return {
            "query": self.original_query,
            "paper_count": self.paper_count,
            "total_found": self.total_found,
            "processing_time": self.processing_time,
            "cache_hit": self.cache_hit,
            "top_categories": list(self.category_distribution.keys())[:5],
            "average_relevance": round(self.average_relevance_score, 3),
            "recent_papers_count": len(self.recent_papers),
            "strategies_used": self.search_strategies_used,
            "timestamp": self.timestamp.isoformat()
        }