"""
Document Header Generator
"""

from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class DocumentHeaderGenerator(SectionGenerator):
    """Generates document title, author, and metadata"""

    def get_section_name(self) -> str:
        return "document_header"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate document header"""
        header_parts = [
            f"\\title{{{config.title}}}",
            f"\\author{{{config.author}}}",
            "\\date{\\today}",
            "",
            "\\maketitle",
            ""
        ]

        return "\n".join(header_parts)