"""
Query-related data models for literature search system.

This module defines the data structures used for representing search queries,
keyword extraction results, and query configuration options.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class SearchMode(Enum):
    """Search strategy modes for keyword extraction and query processing."""
    FOCUSED = "focused"        # Precise, narrow search targeting specific topics
    COMPREHENSIVE = "comprehensive"  # Broad coverage with multiple search strategies
    EXPLORATORY = "exploratory"     # Discovery-oriented with related concepts


class SortOrder(Enum):
    """ArXiv API sorting options."""
    RELEVANCE = "relevance"         # Sort by relevance (default)
    RECENT = "lastUpdatedDate"      # Sort by last updated date
    SUBMITTED = "submittedDate"     # Sort by original submission date


@dataclass
class DateRange:
    """Date range specification for temporal filtering."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def __post_init__(self):
        """Validate date range."""
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError("start_date cannot be after end_date")


@dataclass
class SearchQuery:
    """
    ArXiv search query specification.
    
    Represents a structured query to be executed against the ArXiv API,
    including keywords, domain categories, sorting preferences, and filters.
    """
    keywords: List[str]
    domain_categories: Optional[List[str]] = None
    max_results: int = 50
    sort_by: SortOrder = SortOrder.RELEVANCE
    sort_order: str = "descending"  # "ascending" or "descending"
    date_range: Optional[DateRange] = None
    search_fields: Optional[List[str]] = None  # ['title', 'abstract', 'authors', 'all']
    
    def __post_init__(self):
        """Validate search query parameters."""
        if not self.keywords:
            raise ValueError("At least one keyword is required")
        
        if self.max_results < 1 or self.max_results > 2000:
            raise ValueError("max_results must be between 1 and 2000")
        
        if self.sort_order not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'")
        
        # Set default search fields
        if self.search_fields is None:
            self.search_fields = ["title", "abstract"]


@dataclass
class KeywordResult:
    """
    Results from LLM-powered keyword extraction.
    
    Contains the extracted keywords organized by priority, suggested search queries,
    domain classification, and metadata about the extraction process.
    """
    primary_keywords: List[str]
    secondary_keywords: List[str] 
    domain_category: Optional[str] = None
    search_queries: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    extraction_reasoning: Optional[str] = None
    suggested_fields: List[str] = field(default_factory=lambda: ["title", "abstract"])
    
    def __post_init__(self):
        """Validate keyword extraction results."""
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        
        # Remove duplicates while preserving order
        self.primary_keywords = list(dict.fromkeys(self.primary_keywords))
        self.secondary_keywords = list(dict.fromkeys(self.secondary_keywords))
        
        # Remove secondary keywords that are already in primary
        self.secondary_keywords = [
            kw for kw in self.secondary_keywords 
            if kw not in self.primary_keywords
        ]
    
    @property
    def all_keywords(self) -> List[str]:
        """Get all keywords (primary + secondary) as a single list."""
        return self.primary_keywords + self.secondary_keywords
    
    @property
    def keyword_count(self) -> int:
        """Total number of extracted keywords."""
        return len(self.primary_keywords) + len(self.secondary_keywords)


@dataclass
class QueryOptions:
    """
    Configuration options for literature query processing.
    
    Controls various aspects of the query execution including search strategy,
    result processing, caching behavior, and performance settings.
    """
    mode: SearchMode = SearchMode.COMPREHENSIVE
    max_results: int = 20
    include_summaries: bool = True
    enable_caching: bool = True
    parallel_strategies: int = 3
    relevance_threshold: float = 0.5
    timeout_seconds: int = 30
    
    # Advanced options
    deduplicate_results: bool = True
    min_abstract_length: int = 100
    max_paper_age_days: Optional[int] = None
    preferred_categories: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate query options."""
        if self.max_results < 1 or self.max_results > 200:
            raise ValueError("max_results must be between 1 and 200")
        
        if self.relevance_threshold < 0.0 or self.relevance_threshold > 1.0:
            raise ValueError("relevance_threshold must be between 0.0 and 1.0")
        
        if self.parallel_strategies < 1 or self.parallel_strategies > 10:
            raise ValueError("parallel_strategies must be between 1 and 10")
        
        if self.timeout_seconds < 5 or self.timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 5 and 300")


@dataclass
class SearchStrategy:
    """
    Individual search strategy configuration.
    
    Represents a specific approach to searching ArXiv, including the query string,
    targeted fields, and expected result characteristics.
    """
    name: str
    query_string: str
    search_fields: List[str]
    expected_results: int
    priority: float = 1.0
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate search strategy."""
        if not self.name or not self.query_string:
            raise ValueError("name and query_string are required")
        
        if self.priority < 0.0 or self.priority > 1.0:
            raise ValueError("priority must be between 0.0 and 1.0")


@dataclass
class QueryMetrics:
    """
    Metrics and performance information for query execution.
    
    Tracks timing, success rates, cache hits, and other operational metrics
    for monitoring and optimization purposes.
    """
    query_start_time: datetime
    query_end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    keyword_extraction_time: float = 0.0
    arxiv_query_time: float = 0.0
    result_processing_time: float = 0.0
    
    # Request statistics
    total_arxiv_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Result statistics
    total_papers_found: int = 0
    papers_after_deduplication: int = 0
    papers_after_filtering: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate request success rate."""
        if self.total_arxiv_requests == 0:
            return 0.0
        return self.successful_requests / self.total_arxiv_requests
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests
    
    def mark_completed(self):
        """Mark query as completed and calculate final processing time."""
        if self.query_end_time is None:
            self.query_end_time = datetime.now()
            self.total_processing_time = (
                self.query_end_time - self.query_start_time
            ).total_seconds()