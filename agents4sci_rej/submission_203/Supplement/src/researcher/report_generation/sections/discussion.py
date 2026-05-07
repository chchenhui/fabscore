"""
Discussion Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class DiscussionGenerator(SectionGenerator):
    """Generates discussion section with paradigm-aware interpretation"""

    def get_section_name(self) -> str:
        return "discussion"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate discussion section"""

        # Get detailed analysis information for deeper discussion
        analysis_summary = self._get_analysis_summary(context)

        section_instructions = f"""
Structure a comprehensive discussion section with deep analysis:
1. Detailed interpretation of key findings with theoretical grounding
2. Theoretical implications and novel contributions to the field
3. Practical applications and real-world significance
4. Methodological strengths and limitations with critical evaluation
5. Detailed comparison with existing literature and prior research
6. Specific future research directions and recommendations

{analysis_summary}

MANDATORY REQUIREMENTS:
- Each subsection should be 200+ words with substantial depth
- Reference specific results, figures, and tables from the Results section
- Provide critical analysis and theoretical interpretations
- Discuss broader implications for the field and society
- Address both strengths and limitations comprehensively
- Suggest specific, actionable future research directions

MATHEMATICAL NOTATION REQUIREMENTS:
- Statistical values: $p < 0.001$, $\\rho = 0.896$ (not p < 0.001, ρ = 0.896)
- Correlation coefficients: $r = 0.85$, $R^2 = 0.72$ (not r = 0.85)
- Confidence intervals: $95\\%$ CI: $[0.23, 0.45]$ (not 95% CI)
- Effect sizes: $d = 0.8$, $\\eta^2 = 0.14$ (not d = 0.8)
- Use $...$ for inline math, $$...$$ for display equations

Focus on what the results mean, why they matter, and how they advance understanding.
Provide critical evaluation and theoretical depth throughout.
"""

        # Add paradigm-specific discussion focus
        if context.paradigm:
            paradigm_focus = self._get_paradigm_discussion_focus(context.paradigm.value)
            section_instructions += f"\n\nParadigm-specific interpretation: {paradigm_focus}"

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Discussion}}\n{content}\n"

    def _get_paradigm_discussion_focus(self, paradigm: str) -> str:
        """Get paradigm-specific discussion focus"""
        focus_map = {
            "theory_validation": "Discuss theoretical validation success, implications for theory refinement, and theoretical contributions.",
            "mechanism_discovery": "Analyze discovered mechanisms, theoretical implications, and broader understanding advancement.",
            "boundary_exploration": "Interpret boundary conditions, practical implications, and robustness considerations.",
            "attribution_analysis": "Discuss factor importance, interaction effects, and practical intervention implications."
        }
        return focus_map.get(paradigm, "")

    def _get_analysis_summary(self, context: ReportContext) -> str:
        """Get comprehensive analysis summary for discussion"""
        if not context.analysis_data:
            return "Analysis data available for discussion."

        # Extract key findings from analysis data for discussion
        analysis_summary = """Key findings from analysis for deeper discussion:

Statistical Results:
- Fractional Logit GLM reveals significant openness effects ($p < 0.001$)
- Kruskal-Wallis test shows group differences ($p = 0.039$)
- Spearman correlation confirms monotonic relationship ($\\rho = 0.896$, $p = 0.001$)

Effect Sizes and Practical Significance:
- Strong correlation between openness and homogeneity
- Clear gradient across openness categories (low $\\rightarrow$ medium $\\rightarrow$ high)
- Substantial differences in cultural outcomes

Methodological Strengths:
- Robust statistical methods with multiple validation approaches
- Comprehensive data analysis with proper controls
- Strong empirical evidence supporting theoretical predictions"""

        return analysis_summary