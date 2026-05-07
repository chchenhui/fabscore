"""
Conclusion Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class ConclusionGenerator(SectionGenerator):
    """Generates conclusion section"""

    def get_section_name(self) -> str:
        return "conclusion"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate conclusion section"""

        section_instructions = """
Structure the conclusion section with:
1. Summary of key findings and contributions
2. Research question answers and objective achievement
3. Broader implications for the field
4. Practical applications and recommendations
5. Study limitations and caveats
6. Future research directions and extensions

Provide a concise but comprehensive synthesis that brings closure to the research narrative.
"""

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Conclusion}}\n{content}\n"