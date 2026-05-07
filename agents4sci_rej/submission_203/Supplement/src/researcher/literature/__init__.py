"""
LLM-based Reference Literature Query System

A comprehensive system for intelligent literature discovery using Large Language Models
to extract keywords and query ArXiv for relevant research papers.

Main components:
- KeywordExtractor: LLM-powered keyword extraction from research questions
- ArXivClient: Robust ArXiv API client with rate limiting and error handling  
- QueryProcessor: Intelligent query orchestration and result processing
- LiteratureService: Main service interface for literature queries

Usage:
    from researcher.literature import LiteratureService
    
    service = LiteratureService()
    result = await service.query_literature(
        "How can transformers be applied to time series forecasting?"
    )
"""

from .services.literature_service import LiteratureService
from .models.query_models import SearchMode, KeywordResult, SearchQuery
from .models.literature_models import Paper, LiteratureResult
from .config.literature_config import get_config, load_config
from .core.pdf_extractor import PDFExtractor, ExtractionResult, extract_pdf_to_markdown, extract_pdf_with_metadata

__version__ = "1.0.0"
__author__ = "OneSim Research Team"

__all__ = [
    "LiteratureService",
    "SearchMode",
    "KeywordResult",
    "SearchQuery",
    "Paper",
    "LiteratureResult",
    "get_config",
    "load_config",
    "PDFExtractor",
    "ExtractionResult",
    "extract_pdf_to_markdown",
    "extract_pdf_with_metadata"
]