"""
Introduction Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class IntroductionGenerator(SectionGenerator):
    """Generates research introduction"""

    def get_section_name(self) -> str:
        return "introduction"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate introduction section"""

        # Add citation information if available
        citation_info = self._get_citation_information(context)

        section_instructions = f"""
Structure the introduction with:
1. Research domain and context establishment
2. Problem statement and research gap identification
3. Research objectives and questions
4. Approach overview and methodology preview
5. Contribution summary and paper organization

{citation_info}

CRITICAL CITATION REQUIREMENT:
- ONLY use citation keys that are provided above
- DO NOT create fictional citations like \\cite{{Borgatti2006}} or \\cite{{McCrae2000}}
- If no citations are available, write the introduction without citations
- Use proper LaTeX citation format: \\cite{{key}} ONLY for the provided keys

Ensure smooth transitions between subsections and establish clear motivation for the research.
"""

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Introduction}}\n{content}\n"

    def _get_citation_information(self, context: ReportContext) -> str:
        """Get information about available citations"""
        if not context.citation_entries:
            return "No reference citations available."

        citations_list = []
        for citation_key in list(context.citation_entries.keys())[:5]:  # Limit to 5 main references
            citations_list.append(f"- {citation_key}")

        return f"""Available reference citations:
{chr(10).join(citations_list)}

Use these citations appropriately in the introduction to establish context and support claims."""