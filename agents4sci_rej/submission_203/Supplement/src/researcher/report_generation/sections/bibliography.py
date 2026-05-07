"""
Bibliography Generator
"""

from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class BibliographyGenerator(SectionGenerator):
    """Generates bibliography section"""

    def get_section_name(self) -> str:
        return "bibliography"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate bibliography section"""

        if not context.citation_entries:
            return ""

        # Create bibliography content
        bibliography_content = [
            "\\bibliographystyle{plain}",
            "\\bibliography{report}",
            ""
        ]

        return "\n".join(bibliography_content)