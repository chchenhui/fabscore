"""
Results Generator
"""

from onesim.models import SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext
from .base import SectionGenerator


class ResultsGenerator(SectionGenerator):
    """Generates results section with paradigm-aware presentation"""

    def get_section_name(self) -> str:
        return "results"

    def generate(self, context: ReportContext, config: ReportConfig) -> str:
        """Generate results section"""

        # Include available figures information
        figures_info = self._get_figures_information(context)

        # Get detailed data analysis information
        data_analysis_info = self._get_data_analysis_information(context)

        section_instructions = f"""
Structure the results section with detailed analysis:
1. Overview of experimental outcomes with specific metrics
2. Quantitative results with comprehensive statistical analysis
3. Key patterns and trends identification with effect sizes
4. Visual representation of findings (figures and tables)
5. Robustness and sensitivity analysis with confidence intervals

{figures_info}

{data_analysis_info}

MANDATORY REQUIREMENTS:
- Include ALL available figures using proper LaTeX syntax
- Create comprehensive data tables showing key metrics
- Provide detailed statistical analysis with p-values and effect sizes
- Reference all figures and tables in the text
- Present at least 3-4 subsections with substantial content each

CRITICAL FORMATTING REQUIREMENT:
- Use ONLY LaTeX formatting, NO Markdown (no #, **, ##, etc.)
- Use \\subsection{{}} for subsections, not ## or #
- Use \\textbf{{}} for bold text, not **text**
- Use \\textit{{}} for italic text, not *text*
- MATHEMATICAL SYMBOLS: Always use LaTeX math mode:
  * Correlation coefficients: $\\rho = 0.85$, $r = 0.92$ (not ρ = 0.85)
  * P-values: $p < 0.01$, $p = 0.005$ (not p < 0.01)
  * Statistical tests: $\\chi^2$, $t$-test, $F$-statistic
  * Inequalities: $\\leq$, $\\geq$, $<$, $>$ (not ≤, ≥, <, >)
  * Use $...$ for inline math, $$...$$ for display equations

IMPORTANT: Use LaTeX table syntax for all data tables:
\\begin{{table}}[ht]
\\centering
\\caption{{Your descriptive table caption}}
\\label{{tab:labelname}}
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Parameter & Mean & Std Dev & Effect Size \\\\
\\hline
Low Openness & X.XX & X.XX & X.XX \\\\
Medium Openness & X.XX & X.XX & X.XX \\\\
High Openness & X.XX & X.XX & X.XX \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

Present results objectively with substantial detail and analysis.
Each subsection should be 150+ words with specific data and findings.
Use \\ref{{fig:labelname}} and \\ref{{tab:labelname}} to reference elements.
"""

        # Add paradigm-specific result presentation
        if context.paradigm:
            paradigm_focus = self._get_paradigm_results_focus(context.paradigm.value)
            section_instructions += f"\n\nParadigm-specific presentation: {paradigm_focus}"

        prompt = self._build_prompt(context, config, section_instructions)

        response = self.model(self.model.format(
            SystemMessage(content=self._get_system_prompt(config)),
            UserMessage(content=prompt)
        ))

        content = response.text.strip()
        return f"\\section{{Results}}\n{content}\n"

    def _get_figures_information(self, context: ReportContext) -> str:
        """Get information about available figures"""
        if not context.image_references:
            return "No figures available for inclusion."

        figures_list = []
        for i, img_ref in enumerate(context.image_references[:10]):  # Limit to 10 figures
            filename = img_ref.get('relative_path', f"figure_{i+1}.png")
            caption = img_ref.get('caption', f"Analysis result {i+1}")
            figures_list.append(f"- Figure {i+1}: {filename} ({caption})")

        figures_instructions = f"""Available figures for inclusion:
{chr(10).join(figures_list)}

Use LaTeX figure syntax to include figures:
\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=0.8\\textwidth]{{filename.png}}
\\caption{{Your descriptive caption}}
\\label{{fig:labelname}}
\\end{{figure}}

Reference figures in text using \\ref{{fig:labelname}}."""

        return figures_instructions

    def _get_paradigm_results_focus(self, paradigm: str) -> str:
        """Get paradigm-specific results presentation focus"""
        focus_map = {
            "theory_validation": "Emphasize hypothesis testing results, statistical validation metrics, and theoretical confirmation evidence.",
            "mechanism_discovery": "Highlight discovered patterns, emergent behaviors, and mechanism identification evidence.",
            "boundary_exploration": "Present threshold identification, parameter sensitivity analysis, and boundary characterization.",
            "attribution_analysis": "Quantify factor contributions, relative importance measures, and sensitivity analysis results."
        }
        return focus_map.get(paradigm, "")

    def _get_data_analysis_information(self, context: ReportContext) -> str:
        """Get information about available data analysis"""
        if not context.analysis_data:
            return "No detailed analysis data available."

        # Extract key metrics and statistics from analysis data
        analysis_summary = []

        try:
            import json
            if isinstance(context.analysis_data, str):
                data = json.loads(context.analysis_data)
            else:
                data = context.analysis_data

            # Look for common analysis patterns
            if isinstance(data, dict):
                for key, value in data.items():
                    if 'metrics' in key.lower() or 'statistics' in key.lower():
                        analysis_summary.append(f"- {key}: Available for detailed analysis")
                    elif 'results' in key.lower() or 'analysis' in key.lower():
                        analysis_summary.append(f"- {key}: Data available for presentation")

        except (json.JSONDecodeError, TypeError):
            analysis_summary.append("- Raw analysis data available for processing")

        if not analysis_summary:
            analysis_summary.append("- General analysis data available")

        return f"""Available analysis data for detailed presentation:
{chr(10).join(analysis_summary)}

Create comprehensive tables showing:
1. Summary statistics (means, standard deviations, confidence intervals)
2. Comparative analysis between conditions/groups
3. Statistical test results (p-values, effect sizes)
4. Performance metrics across different parameters
5. Temporal or categorical breakdowns of key variables"""