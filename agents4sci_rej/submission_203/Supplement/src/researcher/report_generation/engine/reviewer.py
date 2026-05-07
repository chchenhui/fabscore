"""
Review Engine for iterative report improvement
"""

import re
import json
from typing import Dict
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from onesim.models import get_model_manager, SystemMessage, UserMessage
from ..core.config import ReportConfig
from ..core.context import ReportContext


@dataclass
class QualityScore:
    """Simple, clean quality scoring without dict complexity"""
    technical_rigor: float = 2.5
    clarity: float = 2.5
    validation: float = 2.5
    writing_quality: float = 2.5

    @property
    def overall(self) -> float:
        return (self.technical_rigor + self.clarity +
                self.validation + self.writing_quality) / 4

    def meets_threshold(self, thresholds: Dict[str, float]) -> bool:
        """No special cases - just check each score"""
        return all([
            getattr(self, field) >= thresholds.get(field, 3.0)
            for field in ['technical_rigor', 'clarity', 'validation', 'writing_quality']
        ])

    def to_dict(self) -> Dict[str, float]:
        """For backward compatibility only"""
        return {
            'technical_rigor': self.technical_rigor,
            'clarity': self.clarity,
            'validation': self.validation,
            'writing_quality': self.writing_quality
        }


@dataclass
class QualityAssessment:
    """Complete quality assessment without nested dict hell"""
    scores: QualityScore
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    overall_assessment: str

    @classmethod
    def default(cls) -> 'QualityAssessment':
        return cls(
            scores=QualityScore(),
            strengths=['Analysis provided'],
            weaknesses=['Needs improvement'],
            suggestions=['Continue refinement'],
            overall_assessment='Report requires further development'
        )


class ReviewerEngine:
    """Clean, simple review engine without complexity hell"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.model_manager = get_model_manager()
        self.model = self._get_model()

    def _get_model(self):
        """Get model - if it fails, it fails"""
        try:
            return self.model_manager.get_model(self.config.model_config_name)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return self.model_manager.get_model(None)

    def review_full_document_then_sections(self, full_document: str, sections_dict: dict, context: ReportContext) -> str:
        """Iteratively review and improve the document until thresholds are met"""
        logger.info("Starting iterative document review")

        current_sections = sections_dict
        current_document = self._assemble_improved_document(current_sections, context)

        for iteration in range(self.config.max_review_iterations):
            logger.info(f"Global review iteration {iteration + 1}/{self.config.max_review_iterations}")

            # Full-document structured analysis (overall scores + per-section issues)
            analysis = self._analyze_full_document(current_document, context)
            structured = analysis.get('structured_analysis', {})
            overall_scores = structured.get('overall_scores', {})
            section_issues = structured.get('section_issues', {})

            # Stop if scores meet thresholds and no section has issues
            if (not self._needs_whole_document_improvement(overall_scores)
                    and not self._has_open_section_issues(section_issues)):
                logger.info("Document meets quality thresholds and has no outstanding section issues")
                break

            # Improve only the sections with identified issues
            current_sections = self._review_and_improve_sections(current_sections, analysis, context)
            current_document = self._assemble_improved_document(current_sections, context)

        # Final detailed quality save (keeps backward compatibility for outputs)
        final_quality = self._assess_final_quality(current_document, context)
        self._save_quality_assessment(final_quality, context)

        return current_document

    def review_and_improve(self, content: str, context: ReportContext) -> str:
        """Simple iterative improvement - no complex logic"""
        current_content = content

        for iteration in range(self.config.max_review_iterations):
            logger.info(f"Review iteration {iteration + 1}")

            quality = self._assess_quality(current_content)
            if self._meets_quality_threshold(quality):
                logger.info(f"Quality threshold met")
                break

            current_content = self._improve_content(current_content, context)

        return current_content

    def _assess_quality(self, content: str) -> QualityAssessment:
        """Simple quality assessment without dict hell"""
        prompt = self._build_quality_prompt(content)
        response = self._call_model(prompt, "academic reviewer")
        return self._parse_quality_response(response.text)

    def _parse_quality_response(self, response_text: str) -> QualityAssessment:
        """Parse quality response - fail fast, no complex fallbacks"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return QualityAssessment.default()

            data = json.loads(json_match.group())

            # Extract scores
            scores_data = data.get('scores', {})
            scores = QualityScore(
                technical_rigor=scores_data.get('technical_rigor', 2.5),
                clarity=scores_data.get('clarity', 2.5),
                validation=scores_data.get('validation', 2.5),
                writing_quality=scores_data.get('writing_quality', 2.5)
            )

            # Extract feedback
            feedback = data.get('detailed_feedback', {})

            return QualityAssessment(
                scores=scores,
                strengths=feedback.get('strengths', ['Analysis provided']),
                weaknesses=feedback.get('weaknesses', ['Needs improvement']),
                suggestions=feedback.get('suggestions', ['Continue refinement']),
                overall_assessment=data.get('overall_assessment', 'Assessment completed')
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Quality parsing failed: {e}")
            return QualityAssessment.default()

    def _build_quality_prompt(self, content: str) -> str:
        """Build quality assessment prompt"""
        return f"""Evaluate this research report and provide assessment in JSON format.

## Report Content:
{content[:2000]}...

## Required JSON Output:
{{
  "scores": {{
    "technical_rigor": <1-5>,
    "clarity": <1-5>,
    "validation": <1-5>,
    "writing_quality": <1-5>
  }},
  "detailed_feedback": {{
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "suggestions": ["suggestion 1", "suggestion 2"]
  }},
  "overall_assessment": "brief evaluation"
}}

Provide ONLY the JSON response."""

    def _call_model(self, prompt: str, role: str) -> object:
        """Simple model call wrapper"""
        return self.model(self.model.format(
            SystemMessage(content=f"You are an {role}. Provide structured assessment."),
            UserMessage(content=prompt)
        ))

    def _meets_quality_threshold(self, quality: QualityAssessment) -> bool:
        """Simple threshold check"""
        return quality.scores.meets_threshold(self.config.quality_thresholds)

    def _has_open_section_issues(self, section_issues: dict) -> bool:
        """Return True if any section still has issues listed"""
        try:
            for _, data in section_issues.items():
                if data.get('issues'):
                    return True
            return False
        except Exception:
            return False

    def _section_needs_improvement(self, section_name: str, analysis: dict) -> bool:
        """Simple check if section needs work"""
        section_data = analysis.get('section_issues', {}).get(section_name, {})
        return bool(section_data.get('issues', []))

    def _assess_final_quality(self, document: str, context: ReportContext) -> QualityAssessment:
        """Final quality check"""
        quality = self._assess_quality(document)
        logger.info(f"Final quality: {quality.scores.overall:.2f}")
        return quality

    def _improve_content(self, content: str, context: ReportContext) -> str:
        """Improve content based on quality assessment"""
        analysis_summary = self._truncate_text(context.analysis_data, 2000)

        prompt = f"""Improve this research report to meet academic publication standards.

## Current Report:
{content}

## Research Data:
**Analysis**: {analysis_summary}
**Paradigm**: {context.get_paradigm_description()}

## Improvement Objectives:
- Enhance technical depth and methodological rigor
- Improve clarity and logical structure
- Strengthen evidence support for all claims
- Refine academic writing style

## Requirements:
- Return ONLY the complete LaTeX document
- Start with \\documentclass and end with \\end{{document}}
- Maintain proper LaTeX formatting
- Incorporate all research data meaningfully
- Ensure publication-ready quality
- MANDATORY: Use \\bibliographystyle{{plain}} and \\bibliography{{report}} for references
- Include ALL available figures with proper LaTeX figure syntax
- Create comprehensive data tables with detailed analysis

Generate the improved LaTeX document:"""

        response = self.model(self.model.format(
            SystemMessage(content="You are an expert academic editor specializing in research report improvement."),
            UserMessage(content=prompt)
        ))

        return self._clean_latex_output(response.text)

    def _clean_latex_output(self, content: str) -> str:
        """Clean and validate LaTeX output"""
        # Remove markdown code blocks
        content = re.sub(r'```latex\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = content.strip()

        # Find LaTeX document boundaries
        latex_start = content.find('\\documentclass')
        latex_end = content.rfind('\\end{document}')

        if latex_start != -1 and latex_end != -1:
            # Extract clean LaTeX content
            latex_content = content[latex_start:latex_end + len('\\end{document}')]
        else:
            # No proper document structure, use as-is but clean
            latex_content = content

        # Fix common issues
        latex_content = self._fix_mathematical_symbols(latex_content)
        latex_content = self._fix_latex_structure(latex_content)

        # Validate basic LaTeX structure
        if self._validate_latex_structure(latex_content):
            return latex_content
        else:
            logger.warning("LaTeX structure validation failed, returning cleaned content")
            return latex_content

    def _fix_latex_structure(self, content: str) -> str:
        """Fix common LaTeX structural issues"""
        # Ensure proper section endings
        content = re.sub(r'\\section\{([^}]+)\}\s*\n\s*\\section', r'\\section{\1}\n\n\\section', content)

        # Fix common environment issues
        content = re.sub(r'\\begin\{([^}]+)\}([^\\]*?)\\end\{([^}]+)\}',
                        lambda m: f'\\begin{{{m.group(1)}}}{m.group(2)}\\end{{{m.group(1)}}}' if m.group(1) != m.group(3) else m.group(0),
                        content)

        # Ensure proper spacing
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content

    def _validate_latex_structure(self, content: str) -> bool:
        """Basic LaTeX structure validation"""
        try:
            # Check for basic document structure
            has_documentclass = '\\documentclass' in content
            has_begin_doc = '\\begin{document}' in content
            has_end_doc = '\\end{document}' in content

            # Check for balanced braces (basic check)
            open_braces = content.count('{')
            close_braces = content.count('}')
            balanced_braces = abs(open_braces - close_braces) <= 2  # Allow small discrepancy

            # Check for common environments balance
            begin_count = len(re.findall(r'\\begin\{([^}]+)\}', content))
            end_count = len(re.findall(r'\\end\{([^}]+)\}', content))
            balanced_envs = abs(begin_count - end_count) <= 1

            return (has_documentclass or (has_begin_doc and has_end_doc)) and balanced_braces and balanced_envs

        except Exception as e:
            logger.warning(f"LaTeX validation error: {e}")
            return True  # If validation fails, assume it's OK

    def _fix_mathematical_symbols(self, content: str) -> str:
        """Fix common mathematical symbol issues in LaTeX content"""
        # Fix common Unicode mathematical symbols
        replacements = {
            # Greek letters
            'ρ': '$\\rho$',
            'α': '$\\alpha$',
            'β': '$\\beta$',
            'γ': '$\\gamma$',
            'σ': '$\\sigma$',
            'μ': '$\\mu$',
            'π': '$\\pi$',
            'θ': '$\\theta$',
            'λ': '$\\lambda$',
            'δ': '$\\delta$',
            'χ': '$\\chi$',

            # Mathematical operators
            '≤': '$\\leq$',
            '≥': '$\\geq$',
            '×': '$\\times$',
            '±': '$\\pm$',
            '≠': '$\\neq$',
            '≈': '$\\approx$',
            '∞': '$\\infty$',
            '→': '$\\rightarrow$',
            '←': '$\\leftarrow$',
            '↔': '$\\leftrightarrow$',
        }

        for symbol, latex_equiv in replacements.items():
            content = content.replace(symbol, latex_equiv)

        # Fix common pattern issues
        content = re.sub(r'(?<![\$\\])p\s*<\s*0\.(\d+)', r'$p < 0.\1$', content)
        content = re.sub(r'(?<![\$\\])p\s*=\s*0\.(\d+)', r'$p = 0.\1$', content)
        content = re.sub(r'(?<![\$\\])r\s*=\s*0\.(\d+)', r'$r = 0.\1$', content)

        return content

    def _analyze_full_document(self, full_document: str, context: ReportContext) -> dict:
        """Analyze document and return structured section assessments"""

        prompt = f"""Analyze this research report and provide structured assessment in JSON format.

## Document:
{full_document}

## Required JSON Output:
{{
  "overall_scores": {{
    "technical_rigor": <1-5>,
    "clarity": <1-5>,
    "validation": <1-5>,
    "writing_quality": <1-5>
  }},
  "section_issues": {{
    "abstract": {{"issues": ["specific issue 1"], "recommendations": ["specific recommendation 1"]}},
    "introduction": {{"issues": [...], "recommendations": [...]}},
    "methodology": {{"issues": [...], "recommendations": [...]}},
    "results": {{"issues": [...], "recommendations": [...]}},
    "discussion": {{"issues": [...], "recommendations": [...]}},
    "conclusion": {{"issues": [...], "recommendations": [...]}},
    "bibliography": {{"issues": [...], "recommendations": [...]}}
  }},
  "global_issues": [
    "major issue 1", "major issue 2"
  ]
}}

Provide specific, actionable feedback. Score scale: 1=poor, 2=below average, 3=acceptable, 4=good, 5=excellent.
Return ONLY the JSON, no additional text."""

        response = self.model(self.model.format(
            SystemMessage(content="You are an academic reviewer. Provide structured JSON analysis."),
            UserMessage(content=prompt)
        ))

        return self._parse_structured_analysis(response.text.strip(), full_document)

    def _review_and_improve_sections(self, sections_dict: dict, document_analysis: dict, context: ReportContext) -> dict:
        """Improve sections that have identified issues"""
        improved_sections = {}
        structured_analysis = document_analysis.get('structured_analysis', {})
        section_issues = structured_analysis.get('section_issues', {})

        for section_name, section_info in sections_dict.items():
            section_data = section_issues.get(section_name, {})
            issues = section_data.get('issues', [])

            if issues:  # Only improve sections with identified issues
                logger.info(f"Improving section: {section_name} ({len(issues)} issues identified)")

                # Improve the section
                improved_content = self._improve_section_content(
                    section_name,
                    section_info['content'],
                    section_data,
                    context
                )

                improved_sections[section_name] = {
                    'content': improved_content,
                    'name': section_name,
                    'improved': True
                }
            else:
                logger.info(f"Section {section_name} has no issues, keeping unchanged")
                improved_sections[section_name] = section_info

        return improved_sections

    def _needs_whole_document_improvement(self, overall_scores: dict) -> bool:
        """Check if overall document quality is below threshold"""
        for criterion in ['technical_rigor', 'clarity', 'validation', 'writing_quality']:
            score = overall_scores.get(criterion, 3.0)
            threshold = self.config.quality_thresholds.get(criterion, 3.0)
            if score < threshold:
                return True
        return False

    

    def _improve_section_content(self, section_name: str, content: str, section_data: dict, context: ReportContext) -> str:
        """Improve section based on identified issues and recommendations"""
        issues = section_data.get('issues', [])
        recommendations = section_data.get('recommendations', [])

        issues_text = '\n'.join([f"- {issue}" for issue in issues]) if issues else "No specific issues identified"
        recommendations_text = '\n'.join([f"- {rec}" for rec in recommendations]) if recommendations else "General improvements needed"

        prompt = f"""Improve this {section_name} section by addressing the identified issues.

## Current Section:
{content}

## Issues to Address:
{issues_text}

## Recommendations:
{recommendations_text}

## Requirements:
{self._get_section_specific_requirements(section_name, context)}

## Output Requirements:
- Return ONLY the improved section content
- Address ALL identified issues
- Follow all recommendations
- Maintain proper LaTeX formatting
- {self._get_mandatory_elements(section_name, context)}

Generate the improved section:"""

        response = self.model(self.model.format(
            SystemMessage(content=f"You are an expert academic writer. Address specific issues in the {section_name} section."),
            UserMessage(content=prompt)
        ))

        return response.text.strip()

    

    def _assemble_improved_document(self, improved_sections: dict, context: ReportContext) -> str:
        """Assemble the final document from improved sections"""
        document_parts = [
            "\\documentclass[11pt]{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{amsmath, amsfonts, amssymb}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\usepackage{geometry}",
            "\\geometry{margin=1in}",
            "",
            "\\begin{document}",
            ""
        ]
        if self.config.language == "zh":
            document_parts.append("\\usepackage{ctex}")
        # Add sections in proper order
        section_order = ['document_header', 'abstract', 'introduction', 'literature_review',
                        'methodology', 'results', 'discussion', 'conclusion', 'bibliography']

        for section_name in section_order:
            if section_name in improved_sections:
                document_parts.append(improved_sections[section_name]['content'])

        document_parts.extend([
            "",
            "\\end{document}"
        ])

        return "\n".join(document_parts)

    def _get_section_specific_requirements(self, section_name: str, context: ReportContext) -> str:
        """Get specific requirements for each section type"""
        requirements = {
            'results': f"""
- MANDATORY: Include ALL available figures using \\includegraphics
- Create comprehensive data tables with statistical analysis
- Reference all figures and tables in text
- Available figures: {len(context.image_references)} figures
- Statistical data available for detailed analysis
            """,
            'discussion': """
- Provide deep theoretical interpretation of results
- Compare findings with existing literature
- Discuss practical implications and significance
- Address limitations and future research directions
            """,
            'bibliography': """
- MANDATORY: Use \\bibliographystyle{plain} and \\bibliography{report}
- Do NOT use manual \\begin{thebibliography}
            """,
            'introduction': """
- Clearly establish research context and motivation
- Reference available literature appropriately
- Use \\cite{} commands for citations
            """
        }
        return requirements.get(section_name, "Follow standard academic writing conventions.")

    

    def _get_mandatory_elements(self, section_name: str, context: ReportContext) -> str:
        """Get mandatory elements that must be included"""
        if section_name == 'results' and context.image_references:
            return f"MUST include all {len(context.image_references)} available figures"
        elif section_name == 'bibliography':
            return "MUST use \\bibliography{report} format"
        return ""

    def _parse_structured_analysis(self, response_text: str, full_document: str) -> dict:
        """Parse and validate structured analysis from LLM response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                logger.warning("No JSON found in analysis response, using fallback")
                analysis_data = self._create_fallback_analysis()

            # Validate structure
            analysis_data = self._validate_analysis_structure(analysis_data)

            return {
                'structured_analysis': analysis_data,
                'full_document': full_document
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            return {
                'structured_analysis': self._create_fallback_analysis(),
                'full_document': full_document,
                'parse_error': str(e)
            }

    def _validate_analysis_structure(self, analysis_data: dict) -> dict:
        """Ensure analysis has required structure"""
        required_sections = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'bibliography']

        # Ensure section_issues exists
        if 'section_issues' not in analysis_data:
            analysis_data['section_issues'] = {}

        # Ensure all sections have proper structure
        for section in required_sections:
            if section not in analysis_data['section_issues']:
                analysis_data['section_issues'][section] = {
                    'issues': [],
                    'recommendations': []
                }
            else:
                # Validate section structure
                section_data = analysis_data['section_issues'][section]
                if 'issues' not in section_data:
                    section_data['issues'] = []
                if 'recommendations' not in section_data:
                    section_data['recommendations'] = []

        # Ensure other required fields
        if 'overall_scores' not in analysis_data:
            analysis_data['overall_scores'] = {
                'technical_rigor': 3.0,
                'clarity': 3.0,
                'validation': 3.0,
                'writing_quality': 3.0
            }
        if 'global_issues' not in analysis_data:
            analysis_data['global_issues'] = []

        return analysis_data

    def _create_fallback_analysis(self) -> dict:
        """Create fallback analysis when parsing fails"""
        return {
            'overall_scores': {
                'technical_rigor': 2.5,
                'clarity': 3.0,
                'validation': 2.5,
                'writing_quality': 3.0
            },
            'section_issues': {
                'results': {
                    'issues': ['Missing figures and visualizations', 'Insufficient statistical analysis'],
                    'recommendations': ['Add missing figures', 'Enhance statistical analysis']
                },
                'bibliography': {
                    'issues': ['Bibliography format issues'],
                    'recommendations': ['Use proper LaTeX bibliography format']
                },
                'discussion': {
                    'issues': ['Discussion needs more depth'],
                    'recommendations': ['Expand theoretical implications']
                }
            },
            'global_issues': ['Missing visual elements', 'Bibliography formatting']
        }

    def _save_quality_assessment(self, quality: QualityAssessment, context: ReportContext):
        """Save detailed quality assessment results to JSON file"""
        try:
            if not context.output_dir:
                logger.warning("No output directory available for saving quality assessment")
                return

            # Simple extraction from new structure
            quality_scores = quality.scores.to_dict()
            overall_score = quality.scores.overall

            quality_data = {
                "timestamp": datetime.now().isoformat(),
                "quality_scores": quality_scores,
                "overall_score": overall_score,
                "detailed_feedback": {
                    "strengths": quality.strengths,
                    "weaknesses": quality.weaknesses,
                    "suggestions": quality.suggestions
                },
                "overall_assessment": quality.overall_assessment,
                "assessment_details": {
                    "criteria": {
                        "technical_rigor": "Methodological soundness, validation completeness",
                        "clarity": "Structure, readability, logical flow",
                        "validation": "Experimental design, result support",
                        "writing_quality": "Academic style, language precision"
                    },
                    "score_range": "1-5 scale",
                    "threshold_recommendations": {
                        "excellent": ">= 4.0",
                        "good": ">= 3.5",
                        "acceptable": ">= 3.0",
                        "needs_improvement": "< 3.0"
                    }
                },
                "research_context": {
                    "research_topic": context.research_topic,
                    "research_question": context.research_question,
                    "paradigm": context.get_paradigm_description() if context.paradigm else None
                }
            }

            # Determine overall quality level
            if overall_score >= 4.0:
                quality_data["quality_level"] = "excellent"
            elif overall_score >= 3.5:
                quality_data["quality_level"] = "good"
            elif overall_score >= 3.0:
                quality_data["quality_level"] = "acceptable"
            else:
                quality_data["quality_level"] = "needs_improvement"

            # Save to JSON file
            quality_file = context.output_dir / "quality_assessment.json"
            with open(quality_file, 'w', encoding='utf-8') as f:
                json.dump(quality_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Quality assessment saved to: {quality_file}")
            logger.info(f"Overall quality score: {overall_score:.2f} ({quality_data['quality_level']})")

            # Log detailed feedback
            if quality.strengths:
                logger.info(f"Strengths: {len(quality.strengths)} identified")
            if quality.weaknesses:
                logger.info(f"Weaknesses: {len(quality.weaknesses)} identified")

        except Exception as e:
            logger.error(f"Failed to save quality assessment: {e}")

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')

        if last_period > max_length * 0.8:  # Keep if reasonably close to max
            return truncated[:last_period + 1]
        else:
            return truncated + "..."