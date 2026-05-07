"""
Section Factory - Creates appropriate section generators
"""

from typing import Dict, Type
from ..core.config import ReportConfig
from ..sections.base import SectionGenerator
from ..sections.document_header import DocumentHeaderGenerator
from ..sections.abstract import AbstractGenerator
from ..sections.introduction import IntroductionGenerator
from ..sections.literature_review import LiteratureReviewGenerator
from ..sections.methodology import MethodologyGenerator
from ..sections.results import ResultsGenerator
from ..sections.discussion import DiscussionGenerator
from ..sections.conclusion import ConclusionGenerator
from ..sections.bibliography import BibliographyGenerator


class SectionFactory:
    """Factory for creating section generators"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self._generators: Dict[str, Type[SectionGenerator]] = {
            "document_header": DocumentHeaderGenerator,
            "abstract": AbstractGenerator,
            "introduction": IntroductionGenerator,
            "literature_review": LiteratureReviewGenerator,
            "methodology": MethodologyGenerator,
            "results": ResultsGenerator,
            "discussion": DiscussionGenerator,
            "conclusion": ConclusionGenerator,
            "bibliography": BibliographyGenerator
        }

    def get_generator(self, section_name: str) -> SectionGenerator:
        """Get generator instance for specified section"""
        generator_class = self._generators.get(section_name)

        if not generator_class:
            raise ValueError(f"Unknown section: {section_name}")

        return generator_class(self.config.model_config_name)

    def register_generator(self, section_name: str, generator_class: Type[SectionGenerator]):
        """Register custom section generator"""
        self._generators[section_name] = generator_class

    def list_sections(self) -> list:
        """List available section types"""
        return list(self._generators.keys())