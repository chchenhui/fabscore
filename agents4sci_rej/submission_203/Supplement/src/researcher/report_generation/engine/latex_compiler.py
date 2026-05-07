"""
LaTeX Compiler
Handles compilation of LaTeX documents to PDF with bibliography support
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List
from loguru import logger


class LatexCompiler:
    """Compiles LaTeX documents to PDF with full bibliography support"""

    def __init__(self, engine: str = "xelatex"):
        """
        Initialize LaTeX compiler

        Args:
            engine: LaTeX engine to use (xelatex, pdflatex, lualatex)
        """
        self.engine = engine
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required LaTeX tools are available"""
        required_tools = [self.engine, "bibtex"]

        for tool in required_tools:
            if not self._command_exists(tool):
                logger.warning(f"LaTeX tool '{tool}' not found. PDF compilation may fail.")

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, "--version"],
                         capture_output=True,
                         check=True,
                         timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def compile_to_pdf(self,
                      tex_file: Path,
                      output_dir: Optional[Path] = None,
                      clean_aux: bool = True) -> bool:
        """
        Compile LaTeX file to PDF with bibliography support

        Args:
            tex_file: Path to .tex file
            output_dir: Output directory (defaults to tex file directory)
            clean_aux: Whether to clean auxiliary files after compilation

        Returns:
            True if compilation successful, False otherwise
        """
        tex_file = Path(tex_file)
        if not tex_file.exists():
            logger.error(f"LaTeX file not found: {tex_file}")
            return False

        if output_dir is None:
            output_dir = tex_file.parent
        else:
            output_dir = Path(output_dir)

        # Check if bibliography file exists
        bib_file = tex_file.with_suffix('.bib')
        has_bibliography = bib_file.exists() and bib_file.stat().st_size > 0

        logger.info(f"Compiling {tex_file.name} to PDF...")
        if has_bibliography:
            logger.info(f"Bibliography detected: {bib_file.name}")

        try:
            # Change to the directory containing the tex file for compilation
            original_cwd = os.getcwd()
            os.chdir(tex_file.parent)

            success = self._compile_with_bibliography(tex_file.name, has_bibliography)

            if success:
                # Move PDF to output directory if different
                pdf_file = tex_file.with_suffix('.pdf')
                if output_dir != tex_file.parent and pdf_file.exists():
                    target_pdf = output_dir / pdf_file.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    pdf_file.rename(target_pdf)
                    logger.info(f"PDF moved to: {target_pdf}")
                else:
                    logger.info(f"PDF generated: {pdf_file}")

                # Clean auxiliary files if requested
                if clean_aux:
                    self._clean_auxiliary_files(tex_file)

            return success

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def _compile_with_bibliography(self, tex_filename: str, has_bibliography: bool) -> bool:
        """
        Perform complete LaTeX compilation with bibliography

        The standard compilation sequence for documents with bibliography:
        1. latex/xelatex (first pass)
        2. bibtex (process bibliography)
        3. latex/xelatex (second pass - resolve citations)
        4. latex/xelatex (third pass - resolve cross-references)
        """
        try:
            # First LaTeX pass
            logger.info("Running first LaTeX pass...")
            if not self._run_latex(tex_filename):
                return False

            if has_bibliography:
                # Run bibtex
                logger.info("Processing bibliography with bibtex...")
                if not self._run_bibtex(tex_filename):
                    logger.warning("Bibtex failed, continuing without bibliography processing")

                # Second LaTeX pass (resolve citations)
                logger.info("Running second LaTeX pass...")
                if not self._run_latex(tex_filename):
                    return False

            # Final LaTeX pass (resolve cross-references)
            logger.info("Running final LaTeX pass...")
            return self._run_latex(tex_filename)

        except Exception as e:
            logger.error(f"Compilation sequence failed: {e}")
            return False

    def _run_latex(self, tex_filename: str) -> bool:
        """Run LaTeX engine on the document"""
        cmd = [
            self.engine,
            "-interaction=nonstopmode",  # Don't stop on errors
            "-halt-on-error",           # Stop on first error
            tex_filename
        ]

        return self._run_command(cmd, "LaTeX")

    def _run_bibtex(self, tex_filename: str) -> bool:
        """Run bibtex on the document"""
        # bibtex expects the filename without extension
        basename = Path(tex_filename).stem
        cmd = ["bibtex", basename]

        return self._run_command(cmd, "BibTeX")

    def _run_command(self, cmd: List[str], tool_name: str) -> bool:
        """Run a command and handle output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            if result.returncode == 0:
                logger.debug(f"{tool_name} completed successfully")
                return True
            else:
                logger.error(f"{tool_name} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"{tool_name} stderr: {result.stderr}")
                if result.stdout:
                    logger.debug(f"{tool_name} stdout: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"{tool_name} timed out after 60 seconds")
            return False
        except FileNotFoundError:
            logger.error(f"{tool_name} command not found: {cmd[0]}")
            return False
        except Exception as e:
            logger.error(f"{tool_name} execution failed: {e}")
            return False

    def _clean_auxiliary_files(self, tex_file: Path):
        """Clean auxiliary files generated during compilation"""
        auxiliary_extensions = ['.aux', '.log', '.bbl', '.blg', '.toc', '.out', '.fls', '.fdb_latexmk']

        for ext in auxiliary_extensions:
            aux_file = tex_file.with_suffix(ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                    logger.debug(f"Cleaned auxiliary file: {aux_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean {aux_file.name}: {e}")