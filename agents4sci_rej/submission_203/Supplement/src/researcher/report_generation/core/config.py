"""
Report Configuration System
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path


class ResearchParadigm(Enum):
    """Research paradigm types inferred from experimental design"""
    THEORY_VALIDATION = "theory_validation"      # T+C → O
    MECHANISM_DISCOVERY = "mechanism_discovery"  # O+C → T
    BOUNDARY_EXPLORATION = "boundary_exploration"# T+O → C_boundary
    ATTRIBUTION_ANALYSIS = "attribution_analysis"# T+O → C_weight


@dataclass
class ParadigmConfig:
    """Configuration parameters for specific research paradigms"""
    methodology_focus: str
    analysis_approach: str
    result_presentation: str
    writing_style: str
    evaluation_criteria: List[str]


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    title: str = "Research Report"
    author: str = "AI Social Researcher"
    language: str = "zh"

    include_abstract: bool = True
    include_literature_review: bool = True
    include_bibliography: bool = True
    max_literature_papers: int = 20

    output_format: str = "latex"
    enable_review: bool = True
    max_review_iterations: int = 2

    # LaTeX compilation options
    compile_pdf: bool = True
    latex_engine: str = "xelatex"
    clean_aux_files: bool = True

    model_config_name: Optional[str] = None

    reference_paper_path: Optional[str] = None
    outline_template_path: Optional[str] = None

    custom_sections: List[str] = field(default_factory=list)
    section_weights: Dict[str, float] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "technical_rigor": 3.0,
        "clarity": 3.5,
        "novelty": 3.0,
        "validation": 3.0,
        "writing_quality": 3.5
    })

    _paradigm: Optional[ResearchParadigm] = field(default=None, init=False)

    @property
    def research_paradigm(self) -> Optional[ResearchParadigm]:
        """Get inferred research paradigm"""
        return self._paradigm

    def set_paradigm(self, paradigm: ResearchParadigm):
        """Set inferred research paradigm"""
        self._paradigm = paradigm

    def get_paradigm_config(self) -> Optional[ParadigmConfig]:
        """Get paradigm-specific configuration"""
        if not self._paradigm:
            return None

        paradigm_configs = {
            ResearchParadigm.THEORY_VALIDATION: ParadigmConfig(
                methodology_focus="hypothesis_testing",
                analysis_approach="deductive",
                result_presentation="validation_metrics",
                writing_style="formal_theoretical",
                evaluation_criteria=["theoretical_rigor", "hypothesis_clarity", "statistical_validation"]
            ),
            ResearchParadigm.MECHANISM_DISCOVERY: ParadigmConfig(
                methodology_focus="pattern_identification",
                analysis_approach="inductive",
                result_presentation="emergent_patterns",
                writing_style="exploratory_analytical",
                evaluation_criteria=["pattern_discovery", "inductive_reasoning", "generalizability"]
            ),
            ResearchParadigm.BOUNDARY_EXPLORATION: ParadigmConfig(
                methodology_focus="parameter_analysis",
                analysis_approach="parametric",
                result_presentation="threshold_identification",
                writing_style="systematic_comparative",
                evaluation_criteria=["parameter_coverage", "threshold_identification", "robustness"]
            ),
            ResearchParadigm.ATTRIBUTION_ANALYSIS: ParadigmConfig(
                methodology_focus="factor_isolation",
                analysis_approach="analytical",
                result_presentation="contribution_quantification",
                writing_style="quantitative_analytical",
                evaluation_criteria=["factor_isolation", "contribution_quantification", "sensitivity"]
            )
        }
        return paradigm_configs[self._paradigm]

    def get_writing_style(self) -> str:
        """Determine writing style based on available references"""
        if self.reference_paper_path and Path(self.reference_paper_path).exists():
            return "reference_based"
        elif self.outline_template_path and Path(self.outline_template_path).exists():
            return "template_based"
        else:
            return "standard_academic"

    def get_section_modifier(self, section_name: str) -> str:
        """Get section-specific prompt modifier"""
        if not self._paradigm:
            return ""

        modifiers = {
            ResearchParadigm.THEORY_VALIDATION: {
                "methodology": "Focus on hypothesis formulation and testing procedures.",
                "results": "Emphasize statistical validation and theoretical confirmation.",
                "discussion": "Discuss theoretical implications and validation success."
            },
            ResearchParadigm.MECHANISM_DISCOVERY: {
                "methodology": "Emphasize exploratory analysis and pattern detection methods.",
                "results": "Highlight discovered patterns and emergent mechanisms.",
                "discussion": "Analyze mechanism validity and broader implications."
            },
            ResearchParadigm.BOUNDARY_EXPLORATION: {
                "methodology": "Detail parameter space exploration and sensitivity analysis.",
                "results": "Present threshold identification and boundary characterization.",
                "discussion": "Discuss boundary stability and practical implications."
            },
            ResearchParadigm.ATTRIBUTION_ANALYSIS: {
                "methodology": "Describe factor isolation and contribution analysis methods.",
                "results": "Quantify factor contributions and relative importance.",
                "discussion": "Interpret factor interactions and practical significance."
            }
        }

        return modifiers.get(self._paradigm, {}).get(section_name.lower(), "")

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if not self.title.strip():
            issues.append("Title cannot be empty")

        if not self.author.strip():
            issues.append("Author cannot be empty")

        if self.max_literature_papers < 5:
            issues.append("Max literature papers should be at least 5")

        if self.max_review_iterations < 1:
            issues.append("Max review iterations should be at least 1")

        if self.reference_paper_path and not Path(self.reference_paper_path).exists():
            issues.append(f"Reference paper not found: {self.reference_paper_path}")

        if self.outline_template_path and not Path(self.outline_template_path).exists():
            issues.append(f"Outline template not found: {self.outline_template_path}")

        return issues