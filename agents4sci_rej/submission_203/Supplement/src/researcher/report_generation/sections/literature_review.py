"""
Literature Review Generator with Automated Search
"""

import asyncio
from typing import List, Dict, Any
from loguru import logger

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class LiteratureReviewGenerator(SectionGenerator):
    """Generates comprehensive literature review with automated paper search"""

    def __init__(self, model_config_name: str = None):
        super().__init__(model_config_name)
        self._arxiv_client = None
        self._keyword_extractor = None

    def get_section_name(self) -> str:
        return "literature_review"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate literature review section"""

        # Search and collect relevant papers
        papers = self._search_literature(context, config)
        # Generate BibTeX entries if needed
        if config.include_bibliography:
            bibtex_entries = self._generate_bibtex(papers)
            context.citation_entries.update(bibtex_entries)

        # Create literature summary
        literature_summary = self._create_literature_summary(papers)

        # Get citation information
        citation_info = self._get_citation_guidance(context)

        section_instructions = f"""
Generate a comprehensive literature review covering:
1. Theoretical foundations and key concepts
2. Related methodological approaches
3. Previous empirical findings
4. Research gaps and limitations in existing work
5. Positioning of current research

Use the following literature as foundation:
{literature_summary}

{citation_info}

CRITICAL CITATION REQUIREMENT:
- ONLY use citation keys that are provided above
- DO NOT create fictional citations
- Every \\cite{{key}} must exist in the available citation keys
- If citing specific concepts, use the available papers that best relate to those concepts

Organize by themes rather than chronologically, and clearly identify gaps that motivate the current research.
Use proper LaTeX citation format: \\cite{{key}} to reference papers.
"""

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Related Work}}\n{content}\n"

    def _search_literature(self, context: ReportContext, config: ReportConfig) -> List[Dict[str, Any]]:
        """Search for relevant literature"""
        try:
            return asyncio.run(self._async_search_literature(context, config))
        except Exception as e:
            logger.warning(f"Literature search failed: {e}")
            return []

    async def _async_search_literature(self, context: ReportContext, config: ReportConfig) -> List[Dict[str, Any]]:
        """Async literature search"""
        # Import here to avoid circular dependencies
        try:
            from researcher.literature.core.arxiv_client import ArXivClient
            from researcher.literature.core.keyword_extractor import KeywordExtractor
            from researcher.literature.models.query_models import SearchQuery, SortOrder
        except ImportError:
            logger.warning("Literature search modules not available")
            return []

        # Extract keywords from research context
        keyword_extractor = KeywordExtractor(self.model_config_name)
        text_content = f"{context.research_question} {context.analysis_data[:1000]}"

        keyword_result = await keyword_extractor.extract_keywords(
            text=text_content,
            domain_hint="social simulation and multi-agent systems"
        )

        keywords = keyword_result.primary_keywords[:10] if keyword_result.primary_keywords else [
            "agent-based modeling", "social simulation", "complex systems"
        ]

        # Search papers
        search_query = SearchQuery(
            keywords=keywords,
            max_results=config.max_literature_papers,
            search_fields=["title", "abstract"],
            sort_by=SortOrder.RELEVANCE,
            domain_categories=["cs.MA", "cs.AI", "cs.CY", "physics.soc-ph"]
        )
        async with ArXivClient() as client:
            response = await client.search_papers(search_query)
            return [self._paper_to_dict(paper) for paper in response.papers]

    def _paper_to_dict(self, paper) -> Dict[str, Any]:
        """Convert paper object to dictionary"""
        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.abstract,
            "arxiv_id": paper.arxiv_id,
            "published": str(paper.published_date) if hasattr(paper, 'published_date') else ""
        }

    def _generate_bibtex(self, papers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate BibTeX entries for papers"""
        bibtex_entries = {}

        for i, paper in enumerate(papers):
            key = f"paper_{i+1:03d}"
            authors = " and ".join(paper["authors"][:3])  # Limit to first 3 authors

            bibtex_entry = f"""@article{{{key},
    title={{{paper["title"]}}},
    author={{{authors}}},
    journal={{arXiv preprint}},
    year={{2024}},
    note={{arXiv:{paper["arxiv_id"]}}}
}}"""

            bibtex_entries[key] = bibtex_entry

        return bibtex_entries

    def _create_literature_summary(self, papers: List[Dict[str, Any]]) -> str:
        """Create summary of found literature"""
        if not papers:
            return "No relevant literature found through automated search."

        summary_parts = []
        for i, paper in enumerate(papers[:10]):  # Limit to top 10
            authors_str = ", ".join(paper["authors"][:2])
            if len(paper["authors"]) > 2:
                authors_str += " et al."

            summary_parts.append(
                f"**{paper['title'][:80]}** - {authors_str}\n"
                f"Abstract: {paper['abstract'][:200]}...\n"
            )

        return "\n".join(summary_parts)

    def _get_citation_guidance(self, context: ReportContext) -> str:
        """Get guidance for proper citation usage"""
        if not context.citation_entries:
            return "No citation entries available."

        citation_keys = list(context.citation_entries.keys())

        return f"""Available citation keys for referencing:
{', '.join(citation_keys[:10])}

Example usage: \\cite{{paper_001}}
Use these citations throughout the literature review to support your discussion."""