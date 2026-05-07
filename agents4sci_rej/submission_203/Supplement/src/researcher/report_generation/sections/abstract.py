"""
Abstract Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class AbstractGenerator(SectionGenerator):
    """Generates structured research abstract"""

    def get_section_name(self) -> str:
        return "abstract"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate abstract section"""

        section_instructions = """
Generate a cohesive, single-paragraph abstract that flows seamlessly from one element to the next. 
The abstract should naturally incorporate the following elements in a logical sequence:

- Begin with the research background and motivation
- Transition smoothly to describe the methodology and approach used
- Present the key findings and results obtained
- Conclude with the implications and significance of the work

Write the abstract as one continuous, well-connected paragraph without line breaks or section divisions. 
Ensure smooth transitions between ideas using appropriate connecting words and phrases. 
Keep the total length between 150-250 words and ensure it provides a complete, standalone summary 
of the research that readers can understand without referring to the full document.
"""

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        abstract_content = response.text.strip()

        # Wrap in abstract environment
        return f"\\begin{{abstract}}\n{abstract_content}\n\\end{{abstract}}\n"