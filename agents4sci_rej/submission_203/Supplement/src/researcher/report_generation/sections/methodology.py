"""
Methodology Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class MethodologyGenerator(SectionGenerator):
    """Generates methodology section with paradigm-specific focus"""

    def get_section_name(self) -> str:
        return "methodology"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate methodology section"""

        section_instructions = """
Structure the methodology section with:
1. Overall research design and approach
2. Simulation environment and agent-based model description
3. Experimental setup and parameter configuration
4. Data collection and measurement procedures
5. Analysis methods and statistical approaches

MATHEMATICAL NOTATION REQUIREMENTS:
- Grid dimensions: $10 \\times 10$ (not 10 × 10)
- Statistical tests: $\\chi^2$, $t$-test, $p$-values (not χ², p-values)
- Greek symbols: $\\alpha$, $\\beta$, $\\sigma$ (not α, β, σ)
- Inequalities: $\\leq$, $\\geq$ (not ≤, ≥)
- Use $...$ for inline math, $$...$$ for display math

Focus on providing sufficient detail for replication while highlighting the methodological choices that address the research questions.
"""

        # Add paradigm-specific methodology focus
        if context.paradigm:
            paradigm_focus = self._get_paradigm_methodology_focus(context.paradigm.value)
            section_instructions += f"\n\nParadigm-specific focus: {paradigm_focus}"

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Methodology}}\n{content}\n"

    def _get_paradigm_methodology_focus(self, paradigm: str) -> str:
        """Get paradigm-specific methodology focus"""
        focus_map = {
            "theory_validation": "Emphasize hypothesis formulation, controlled experimental design, and validation metrics.",
            "mechanism_discovery": "Focus on exploratory analysis methods, pattern detection algorithms, and inductive reasoning approaches.",
            "boundary_exploration": "Detail parameter space exploration, sensitivity analysis methods, and threshold identification techniques.",
            "attribution_analysis": "Describe factor isolation methods, contribution analysis techniques, and sensitivity measurement approaches."
        }
        return focus_map.get(paradigm, "")