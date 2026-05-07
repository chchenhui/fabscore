"""
Base Section Generator
"""

from abc import ABC, abstractmethod
from typing import Optional

from onesim.models import get_model_manager, SystemMessage, UserMessage
from loguru import logger

from ..core.config import ReportConfig
from ..core.context import ReportContext


class SectionGenerator(ABC):
    """Base class for all section generators"""

    def __init__(self, model_config_name: Optional[str] = None):
        self.model_config_name = model_config_name
        self.model_manager = get_model_manager()
        self.model = self._get_model(model_config_name)

    def _get_model(self, config_name: Optional[str]):
        """Get model instance with fallback"""
        try:
            return self.model_manager.get_model(config_name)
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}, using default")
            return self.model_manager.get_model(None)

    @abstractmethod
    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate section content"""
        pass

    @abstractmethod
    def get_section_name(self) -> str:
        """Get section name"""
        pass

    def _build_prompt(self, context: ReportContext, config: ReportConfig,
                     section_specific_instructions: str = "") -> str:
        """Build standardized prompt for section generation"""

        paradigm_modifier = config.get_section_modifier(self.get_section_name())
        writing_style = config.get_writing_style()

        base_prompt = f"""Generate the {self.get_section_name()} section for a research report.

## Research Context:
**Topic**: {context.research_topic}
**Question**: {context.research_question}
**Paradigm**: {context.get_paradigm_description()}

## Scenario Data:
```json
{context.scenario_data}
```

## Analysis Results:
{context.analysis_data[:2000]}

## Metrics Data:
```json
{context.metrics_data}
```"""

        if paradigm_modifier:
            base_prompt += f"\n\n## Paradigm-Specific Focus:\n{paradigm_modifier}"

        if context.reference_content:
            base_prompt += f"\n\n## Reference Style:\nFollow the writing style and structure from the provided reference material."

        if context.outline_template:
            base_prompt += f"\n\n## Template Structure:\n{context.outline_template[:1000]}"

        if section_specific_instructions:
            base_prompt += f"\n\n## Section-Specific Instructions:\n{section_specific_instructions}"

        base_prompt += f"""

## Output Requirements:
- Generate LaTeX content for the {self.get_section_name()} section only
- Do NOT include \\section{{{self.get_section_name()}}} command
- Use proper LaTeX formatting (\\textbf{{}}, \\textit{{}}, \\subsection{{}})
- CRITICAL: Use ONLY LaTeX formatting, NO Markdown (no #, **, ##, etc.)
- Use \\subsection{{}} for subsections, not ## or #
- Use \\textbf{{}} for bold text, not **text**
- Use \\textit{{}} for italic text, not *text*
- MATHEMATICAL SYMBOLS: Use proper LaTeX math mode for all mathematical symbols:
  * Greek letters: \\rho, \\alpha, \\beta, \\sigma, \\mu, etc. (not ρ, α, β)
  * Mathematical operators: $<$, $>$, $\\leq$, $\\geq$, $\\times$, $\\pm$ (not <, >, ≤, ≥, ×, ±)
  * Statistical symbols: $p$-value, $r = 0.85$, $\\chi^2$, etc.
  * Use $...$ for inline math and $$...$$ for display math
- Base content strictly on provided data
- Writing style: {writing_style}
- Language: {config.language}

Generate the section content:"""

        return base_prompt

    def _get_system_prompt(self, config: ReportConfig) -> str:
        """Get standardized system prompt"""
        language_instruction = ""
        if config.language == "zh":
            language_instruction = "CRITICAL: Write ONLY in Chinese (中文). Do NOT mix English and Chinese in the same section."
        elif config.language == "en":
            language_instruction = "CRITICAL: Write ONLY in English. Do NOT mix Chinese and English in the same section."

        return f"""You are an expert academic writer specializing in {self.get_section_name()} sections.

Your task is to write high-quality LaTeX content that is:
- Scientifically accurate and evidence-based
- Well-structured and academically rigorous
- Properly formatted in LaTeX
- Written in {config.language} language ONLY

{language_instruction}

Focus on creating content that directly serves the purpose of the {self.get_section_name()} section within the broader research report."""