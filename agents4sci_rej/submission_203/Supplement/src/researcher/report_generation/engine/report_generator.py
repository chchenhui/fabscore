"""
Unified Report Generator
Single algorithm handles all research paradigms
"""

from pathlib import Path
from loguru import logger

from ..core.config import ReportConfig
from ..core.context import ReportContext
from .section_factory import SectionFactory
from .reviewer import ReviewerEngine
from .latex_compiler import LatexCompiler


class ReportGenerator:
    """Main report generation engine"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.section_factory = SectionFactory(config)
        self.reviewer = ReviewerEngine(config)
        self.latex_compiler = LatexCompiler(engine=config.latex_engine)

    def generate(self, context: ReportContext) -> str:
        """Generate complete report - single algorithm for all paradigms"""
        logger.info("Starting report generation")
        
        # Sync paradigm between config and context
        if context.paradigm:
            self.config.set_paradigm(context.paradigm)

        # Load reference materials if configured
        self._load_references(context)

        # Generate sections separately and store in memory
        sections_dict = self._generate_sections_dict(context)

        # Assemble initial complete document
        initial_report = self._assemble_document_from_sections(sections_dict, context)

        # Review and improve if enabled
        if self.config.enable_review:
            final_report = self.reviewer.review_full_document_then_sections(
                initial_report, sections_dict, context
            )
        else:
            final_report = initial_report

        # Use project-specific output path instead of CLI output path
        project_output_path = context.output_dir / "report.tex"

        # Save report
        self._save_report(final_report, context, str(project_output_path))

        # Compile to PDF if enabled
        if self.config.compile_pdf:
            self._compile_to_pdf(project_output_path)

        logger.info(f"Report generated: {project_output_path}")
        return final_report

    def _load_references(self, context: ReportContext):
        """Load reference materials"""
        if self.config.reference_paper_path:
            context.load_reference_content(self.config.reference_paper_path)

        if self.config.outline_template_path:
            context.load_outline_template(self.config.outline_template_path)

    def _generate_content(self, context: ReportContext) -> str:
        """Generate complete document content"""
        sections = []

        # Get section sequence based on paradigm
        section_sequence = self._get_section_sequence()

        # Generate each section
        for section_name in section_sequence:
            logger.info(f"Generating section: {section_name}")

            generator = self.section_factory.get_generator(section_name)
            section_content = generator.generate(context, self.config)

            sections.append(section_content)

        # Combine into complete document
        return self._assemble_document(sections, context)

    def _generate_sections_dict(self, context: ReportContext) -> dict:
        """Generate individual sections and return as dictionary"""
        sections_dict = {}

        # Get section sequence based on paradigm
        section_sequence = self._get_section_sequence()

        # Generate each section separately
        for section_name in section_sequence:
            logger.info(f"Generating section: {section_name}")

            generator = self.section_factory.get_generator(section_name)
            section_content = generator.generate(context, self.config)

            sections_dict[section_name] = {
                'content': section_content,
                'generator': generator,
                'name': section_name
            }

        return sections_dict

    def _assemble_document_from_sections(self, sections_dict: dict, context: ReportContext) -> str:
        """Assemble complete document from sections dictionary"""
        document_parts = [
            self._get_document_preamble(),
            "\\begin{document}",
            ""
        ]

        # Add sections in order
        for section_info in sections_dict.values():
            document_parts.append(section_info['content'])

        document_parts.extend([
            "",
            "\\end{document}"
        ])

        return "\n".join(document_parts)

    def _get_section_sequence(self) -> list:
        """Get section sequence based on configuration"""
        base_sequence = [
            "document_header",
            "abstract" if self.config.include_abstract else None,
            "introduction",
            "literature_review" if self.config.include_literature_review else None,
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "bibliography" if self.config.include_bibliography else None
        ]

        # Filter out None values and add custom sections
        sequence = [s for s in base_sequence if s is not None]
        sequence.extend(self.config.custom_sections)

        return sequence

    def _assemble_document(self, sections: list, context: ReportContext) -> str:
        """Assemble sections into complete LaTeX document"""
        document_parts = [
            self._get_document_preamble(),
            "\\begin{document}",
            ""
        ]

        document_parts.extend(sections)

        document_parts.extend([
            "",
            "\\end{document}"
        ])

        return "\n".join(document_parts)

    def _get_document_preamble(self) -> str:
        """Get LaTeX document preamble"""
        preamble = [
            "\\documentclass[11pt]{article}",
            "\\usepackage[utf8]{inputenc}",
        ]

        # Only add ctex package for Chinese language
        if self.config.language == "zh":
            preamble.append("\\usepackage{ctex}")

        preamble.extend([
            "\\usepackage{amsmath, amsfonts, amssymb}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\usepackage{geometry}",
            "\\geometry{margin=1in}",
            ""
        ])

        return "\n".join(preamble)

    def _save_report(self, content: str,context: ReportContext, output_path: str):
        """Save report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Save bibliography if enabled
        if self.config.include_bibliography:
            bib_path = output_file.with_suffix('.bib')
            self._save_bibliography(bib_path, context)

    def _save_bibliography(self, bib_path: Path, context: ReportContext):
        """Save bibliography file"""
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write("% Generated bibliography\n\n")

            # Write citation entries
            for _, citation_content in context.citation_entries.items():
                f.write(f"{citation_content}\n\n")

    def _compile_to_pdf(self, tex_path: Path):
        """Compile LaTeX file to PDF"""
        logger.info("Compiling LaTeX to PDF...")

        success = self.latex_compiler.compile_to_pdf(
            tex_file=tex_path,
            clean_aux=self.config.clean_aux_files
        )

        if success:
            pdf_path = tex_path.with_suffix('.pdf')
            logger.info(f"✅ PDF compilation successful: {pdf_path}")
        else:
            logger.error("❌ PDF compilation failed")
            logger.info("Please check that you have LaTeX installed with required packages:")
            logger.info("  - xelatex or pdflatex")
            logger.info("  - bibtex")
            if self.config.language == "zh":
                logger.info("  - ctex package for Chinese support")


# Convenience function for simple usage
def generate_report(project_path: str, output_path: str, **config_kwargs) -> str:
    """Generate report with minimal configuration"""
    report_context = ReportContext.from_project_path(project_path)
    config = ReportConfig(**config_kwargs)

    generator = ReportGenerator(config)
    return generator.generate(report_context, output_path)