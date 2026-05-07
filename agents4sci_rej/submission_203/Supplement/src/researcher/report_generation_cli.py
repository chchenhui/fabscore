#!/usr/bin/env python3
"""
Report Generation CLI
Unified command-line interface for research report generation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Add project path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from researcher.report_generation import ReportGenerator, ReportConfig, ReportContext


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate research reports from simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (generates both .tex and .pdf)
  python report_generation_cli.py --project /path/to/project 

  # With custom configuration
  python report_generation_cli.py --project /path/to/project  \\
      --title "My Research" --author "Researcher" --enable-review

  # English report with specific LaTeX engine
  python report_generation_cli.py --project /path/to/project  \\
      --language en --latex-engine pdflatex

  # Generate only LaTeX file (no PDF compilation)
  python report_generation_cli.py --project /path/to/project  \\
      --no-compile

  # With reference materials and keep auxiliary files
  python report_generation_cli.py --project /path/to/project  \\
      --reference-paper reference.pdf --outline-template template.md --keep-aux
        """
    )

    # Required arguments
    parser.add_argument(
        "--project",
        required=True,
        help="Path to project directory containing simulation results"
    )

    # Report configuration
    parser.add_argument(
        "--title",
        default="Research Report",
        help="Report title"
    )

    parser.add_argument(
        "--author",
        default="AI Social Researcher",
        help="Report author"
    )

    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Report language"
    )

    # Content options
    parser.add_argument(
        "--no-abstract",
        action="store_true",
        help="Disable abstract generation"
    )

    parser.add_argument(
        "--no-literature-review",
        action="store_true",
        help="Disable literature review generation"
    )

    parser.add_argument(
        "--no-bibliography",
        action="store_true",
        help="Disable bibliography generation"
    )

    parser.add_argument(
        "--max-literature-papers",
        type=int,
        default=15,
        help="Maximum number of literature papers to include"
    )

    # Quality control
    parser.add_argument(
        "--disable-review",
        action="store_true",
        help="Disable automated review and improvement"
    )

    parser.add_argument(
        "--max-review-iterations",
        type=int,
        default=2,
        help="Maximum number of review iterations"
    )

    # Model configuration
    parser.add_argument(
        "--model-config",
        help="Model configuration name"
    )

    # Reference materials
    parser.add_argument(
        "--reference-paper",
        help="Path to reference paper for style guidance"
    )

    parser.add_argument(
        "--outline-template",
        help="Path to outline template file"
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["latex"],
        default="latex",
        help="Output format"
    )

    # LaTeX compilation options
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip PDF compilation, generate only LaTeX file"
    )

    parser.add_argument(
        "--latex-engine",
        choices=["xelatex", "pdflatex", "lualatex"],
        default="xelatex",
        help="LaTeX engine for PDF compilation"
    )

    parser.add_argument(
        "--keep-aux",
        action="store_true",
        help="Keep auxiliary files after compilation"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable all logging except errors"
    )

    return parser


def validate_arguments(args) -> list:
    """Validate command-line arguments"""
    issues = []

    # Check project directory
    project_path = Path(args.project)
    if not project_path.exists():
        issues.append(f"Project directory does not exist: {args.project}")
    elif not project_path.is_dir():
        issues.append(f"Project path is not a directory: {args.project}")

    # Check reference materials
    if args.reference_paper and not Path(args.reference_paper).exists():
        issues.append(f"Reference paper not found: {args.reference_paper}")

    if args.outline_template and not Path(args.outline_template).exists():
        issues.append(f"Outline template not found: {args.outline_template}")

    # Check numeric arguments
    if args.max_literature_papers < 1:
        issues.append("max-literature-papers must be at least 1")

    if args.max_review_iterations < 1:
        issues.append("max-review-iterations must be at least 1")

    return issues


def create_config_from_args(args) -> ReportConfig:
    """Create ReportConfig from command-line arguments"""
    return ReportConfig(
        title=args.title,
        author=args.author,
        language=args.language,
        include_abstract=not args.no_abstract,
        include_literature_review=not args.no_literature_review,
        include_bibliography=not args.no_bibliography,
        max_literature_papers=args.max_literature_papers,
        output_format=args.format,
        enable_review=not args.disable_review,
        max_review_iterations=args.max_review_iterations,
        compile_pdf=not args.no_compile,
        latex_engine=args.latex_engine,
        clean_aux_files=not args.keep_aux,
        model_config_name=args.model_config,
        reference_paper_path=args.reference_paper,
        outline_template_path=args.outline_template
    )


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        issues = validate_arguments(args)
        if issues:
            logger.error("Validation errors:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return 1

        # Create configuration
        config = create_config_from_args(args)

        # Validate configuration
        config_issues = config.validate()
        if config_issues:
            logger.error("Configuration errors:")
            for issue in config_issues:
                logger.error(f"  - {issue}")
            return 1

        # Load project context
        logger.info(f"Loading project from: {args.project}")
        context = ReportContext.from_project_path(args.project, config.model_config_name)

        # Validate context
        context_issues = context.validate()
        if context_issues:
            logger.error("Project context errors:")
            for issue in context_issues:
                logger.error(f"  - {issue}")
            return 1

        # Log inferred paradigm
        if context.paradigm:
            logger.info(f"Inferred research paradigm: {context.get_paradigm_description()}")

        # Generate report
        generator = ReportGenerator(config)
        generator.generate(context)

        logger.info("✅ Report generation completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())


#   # 基本使用（生成.tex和.pdf）
#   python report_generation_cli.py --project /path/to/project 

#   # 英文报告使用pdflatex
#   python report_generation_cli.py --project /path/to/project  \
#       --language en --latex-engine pdflatex

#   # 只生成LaTeX文件，不编译PDF
#   python report_generation_cli.py --project /path/to/project  \
#       --no-compile

#   # 保留辅助文件用于调试
#   python report_generation_cli.py --project /path/to/project  \
#       --keep-aux