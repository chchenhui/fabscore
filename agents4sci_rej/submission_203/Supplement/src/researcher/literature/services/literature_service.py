"""
Literature service interface for related work generation.

This module provides the primary service interface for literature queries,
coordinating components to provide comprehensive literature search capabilities.
"""

import asyncio
from typing import Optional, List
from datetime import datetime
from loguru import logger

from ..models.query_models import QueryOptions, SearchMode, KeywordResult
from ..models.literature_models import Paper, LiteratureResult
from ..config.literature_config import get_config
from ..core.keyword_extractor import KeywordExtractor
from ..core.arxiv_client import ArXivClient
from ..core.query_processor import QueryProcessor
from ..core.result_formatter import ResultFormatter

# Import OneSim ModelManager if available
try:
    from onesim.models import ModelManager
except ImportError:
    logger.warning("OneSim ModelManager not found, using fallback")
    ModelManager = None


class LiteratureService:
    """
    Main service interface for literature queries and related work generation.

    Provides comprehensive interface for literature search including:
    - Intelligent keyword extraction using LLMs
    - Multi-strategy ArXiv searching
    - Result formatting for academic papers
    """

    def __init__(self, model_manager=None, config=None):
        """
        Initialize literature service.

        Args:
            model_manager: OneSim ModelManager instance (optional)
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()
        self.model_manager = model_manager

        # Initialize components
        self.keyword_extractor = KeywordExtractor(self.model_manager)
        self.arxiv_client = ArXivClient(self.config)

        self.query_processor = QueryProcessor(
            keyword_extractor=self.keyword_extractor,
            arxiv_client=self.arxiv_client,
            cache_service=None
        )

        self.result_formatter = ResultFormatter()

        # Service state
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize the service and all components."""
        if self._initialized:
            return

        try:
            # Initialize ArXiv client session
            await self.arxiv_client._ensure_session()

            self._initialized = True
            logger.info("Literature service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize literature service: {e}")
            raise

    async def close(self):
        """Close the service and cleanup resources."""
        if not self._initialized:
            return

        try:
            # Close ArXiv client
            await self.arxiv_client.close()

            self._initialized = False
            logger.info("Literature service closed")

        except Exception as e:
            logger.error(f"Error closing literature service: {e}")

    async def query_literature(
        self,
        query: str,
        domain: Optional[str] = None,
        max_results: int = 20,
        mode: SearchMode = SearchMode.COMPREHENSIVE,
        include_summaries: bool = True,
        **kwargs
    ) -> LiteratureResult:
        """
        Primary interface for literature queries and related work generation.

        Args:
            query: Natural language research question/topic
            domain: Optional domain hint (e.g., 'machine learning', 'physics')
            max_results: Maximum papers to return (default: 20)
            mode: Search strategy mode
            include_summaries: Generate LLM summaries of abstracts
            **kwargs: Additional query options

        Returns:
            LiteratureResult with papers, metadata, and processing info
        """
        if not self._initialized:
            await self.initialize()

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing literature query: '{query}' (domain: {domain}, mode: {mode.value})")

        start_time = datetime.now()

        try:
            # Build query options
            options = QueryOptions(
                mode=mode,
                max_results=max_results,
                include_summaries=include_summaries,
                enable_caching=False,
                **kwargs
            )

            # Process the query
            result = await self.query_processor.process_literature_query(
                user_input=query,
                options=options
            )

            # Generate summaries if requested and not already present
            if include_summaries:
                result = await self._enhance_with_summaries(result)

            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            logger.info(f"Query completed: {len(result.papers)} papers found in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Literature query failed: {e}")
            raise

    async def extract_keywords_only(
        self,
        text: str,
        domain: Optional[str] = None,
        mode: SearchMode = SearchMode.COMPREHENSIVE
    ) -> KeywordResult:
        """
        Standalone keyword extraction service for research text analysis.

        Args:
            text: Text to extract keywords from
            domain: Optional domain hint
            mode: Extraction mode

        Returns:
            KeywordResult with extracted keywords
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Extracting keywords from: '{text[:50]}...'")

        try:
            result = await self.keyword_extractor.extract_keywords(
                text=text,
                domain_hint=domain,
                mode=mode
            )

            logger.info(f"Extracted {result.keyword_count} keywords (confidence: {result.confidence_score:.2f})")

            return result

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise

    async def _enhance_with_summaries(self, result: LiteratureResult) -> LiteratureResult:
        """Enhance papers with LLM-generated summaries."""
        if not self.model_manager or not result.papers:
            return result

        logger.info("Generating summaries for top papers...")

        # Generate summaries for top papers only (to manage token usage)
        top_papers = result.papers[:min(10, len(result.papers))]

        for paper in top_papers:
            if not paper.summary:  # Only generate if not already present
                try:
                    paper.summary = await self._generate_summary(paper.abstract)
                except Exception as e:
                    logger.warning(f"Failed to generate summary for {paper.arxiv_id}: {e}")
                    paper.summary = "Summary generation failed"

        return result

    async def _generate_summary(self, abstract: str) -> str:
        """Generate LLM summary of paper abstract."""
        if not self.model_manager or not abstract:
            return "No summary available"

        prompt = f"""Please provide a concise 2-3 sentence summary of this research paper abstract, focusing on the main contribution and key findings:

Abstract: {abstract}

Summary:"""

        try:
            response = await self.model_manager.generate_response(
                prompt=prompt,
                model_name=self.config.llm.model_name,
                temperature=0.3,
                max_tokens=200
            )

            # Clean up the response
            summary = response.strip()
            if summary.lower().startswith("summary:"):
                summary = summary[8:].strip()

            return summary

        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Summary generation failed"

    def format_results(
        self,
        result: LiteratureResult,
        format_type: str = "markdown",
        **format_options
    ) -> str:
        """
        Format search results in specified format for integration into research papers.

        Args:
            result: Literature search results
            format_type: Output format ('markdown', 'text', 'latex', 'bibtex')
            **format_options: Format-specific options

        Returns:
            Formatted results string
        """
        if format_type.lower() == "markdown":
            return self.result_formatter.format_as_markdown(result, **format_options)
        elif format_type.lower() == "text":
            return self.result_formatter.format_as_plain_text(result, **format_options)
        elif format_type.lower() == "json":
            import json
            return json.dumps(
                self.result_formatter.format_as_json(result),
                indent=2,
                ensure_ascii=False
            )
        elif format_type.lower() == "latex":
            return self._format_as_latex(result, **format_options)
        elif format_type.lower() == "bibtex":
            return self.result_formatter.format_as_bibtex(result.papers)
        elif format_type.lower() == "summary":
            return self.result_formatter.create_summary_report(result)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _format_as_latex(self, result: LiteratureResult, **options) -> str:
        """Format results as LaTeX for academic papers."""
        latex_content = []
        include_section_header = options.get('include_section_header', True)
        max_papers = options.get('max_papers', 15)

        if include_section_header:
            latex_content.append("% Related Work Section")
            latex_content.append("\\section{Related Work}")
            latex_content.append("")

        if result.papers:
            grouped_papers = self._group_papers_by_theme(result.papers[:max_papers])

            for theme, papers in grouped_papers.items():
                if len(grouped_papers) > 1:
                    latex_content.append(f"\\subsection{{{theme}}}")
                    latex_content.append("")

                for paper in papers:
                    authors = ", ".join(paper.author_names[:3])
                    if len(paper.author_names) > 3:
                        authors += " et al."

                    year = paper.published_date.year if paper.published_date else "Unknown"

                    # Create citation with brief description
                    citation_text = f"{authors} \\cite{{{paper.arxiv_id}}} ({year}) "

                    if paper.summary:
                        citation_text += f"{paper.summary} "
                    elif paper.abstract:
                        abstract_brief = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                        citation_text += f"{abstract_brief} "

                    latex_content.append(citation_text)
                    latex_content.append("")

        return "\n".join(latex_content)

    def _group_papers_by_theme(self, papers: List[Paper]) -> dict:
        """Group papers by research themes for better organization."""
        # For now, return all papers under a single theme
        # This could be enhanced with clustering algorithms
        return {"Literature Overview": papers}