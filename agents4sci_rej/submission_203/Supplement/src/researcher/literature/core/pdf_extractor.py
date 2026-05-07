"""
PDF Literature Content Extractor

A robust PDF extraction module using PyMuPDF4LLM for converting academic papers
to Markdown format suitable for LLM processing and reference integration.

Features:
- High-quality PDF to Markdown conversion
- Support for multi-column layouts
- Image and table extraction
- Metadata preservation
- Error handling and validation
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import tempfile
import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
import pymupdf4llm


@dataclass
class ExtractionResult:
    """PDF extraction result container"""

    markdown_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    extraction_time: float = 0.0
    file_hash: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)

    @property
    def content_preview(self) -> str:
        """Get a preview of the markdown content (first 500 chars)"""
        return self.markdown_content[:500] + "..." if len(self.markdown_content) > 500 else self.markdown_content

    @property
    def content_length(self) -> int:
        """Get the length of markdown content in characters"""
        return len(self.markdown_content)


class PDFExtractor:
    """
    PDF content extractor using PyMuPDF4LLM

    Converts PDF documents to clean Markdown format optimized for LLM processing
    and research reference integration.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize PDF extractor

        Args:
            cache_dir: Optional directory for caching extracted content
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "pdf_extraction_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate MD5 hash of file for caching"""
        file_path = Path(file_path)
        hasher = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for given hash"""
        return self.cache_dir / f"{file_hash}.md"

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> ExtractionResult:
        """
        Extract content from PDF file

        Args:
            file_path: Path to PDF file
            pages: Optional list of 0-based page numbers to extract
            use_cache: Whether to use cached results if available

        Returns:
            ExtractionResult containing markdown content and metadata

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            Exception: For other extraction errors
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")

        start_time = datetime.now()
        file_hash = self._calculate_file_hash(file_path)

        # Check cache first
        if use_cache:
            cached_result = self._load_from_cache(file_hash)
            if cached_result:
                logger.info(f"Loaded PDF content from cache: {file_path.name}")
                return cached_result

        try:
            # Extract using PyMuPDF4LLM
            logger.info(f"Extracting PDF content: {file_path.name}")

            # Run extraction directly
            markdown_content = self._extract_sync(str(file_path), pages)

            extraction_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = ExtractionResult(
                markdown_content=markdown_content,
                metadata=self._extract_metadata(file_path),
                page_count=self._count_pages(markdown_content),
                extraction_time=extraction_time,
                file_hash=file_hash,
                extracted_at=start_time
            )

            # Cache result
            if use_cache:
                self._save_to_cache(result)

            logger.success(
                f"PDF extraction completed: {file_path.name} "
                f"({result.content_length} chars, {result.extraction_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract PDF content from {file_path}: {str(e)}")
            raise Exception(f"PDF extraction failed: {str(e)}") from e

    def _extract_sync(self, file_path: str, pages: Optional[List[int]] = None) -> str:
        """Synchronous extraction wrapper for pymupdf4llm"""
        if pages:
            return pymupdf4llm.to_markdown(file_path, pages=pages)
        else:
            return pymupdf4llm.to_markdown(file_path)

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic file metadata"""
        stat = file_path.stat()
        return {
            "filename": file_path.name,
            "file_size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "file_path": str(file_path.absolute())
        }

    def _count_pages(self, markdown_content: str) -> int:
        """Estimate page count from markdown content"""
        # Look for page break indicators or estimate from content length
        page_breaks = markdown_content.count('\n---\n') + markdown_content.count('\n\n---\n\n')
        if page_breaks > 0:
            return page_breaks + 1

        # Fallback: estimate based on content length (rough estimate)
        estimated_pages = max(1, len(markdown_content) // 3000)
        return estimated_pages

    def _load_from_cache(self, file_hash: str) -> Optional[ExtractionResult]:
        """Load extraction result from cache"""
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        try:
            content = cache_path.read_text(encoding='utf-8')

            # Simple cache format - in production you might want JSON with metadata
            return ExtractionResult(
                markdown_content=content,
                file_hash=file_hash,
                extracted_at=datetime.fromtimestamp(cache_path.stat().st_mtime)
            )
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_hash}: {e}")
            return None

    def _save_to_cache(self, result: ExtractionResult):
        """Save extraction result to cache"""
        cache_path = self._get_cache_path(result.file_hash)

        try:
            cache_path.write_text(result.markdown_content, encoding='utf-8')
            logger.debug(f"Cached extraction result: {result.file_hash}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        pages: Optional[List[int]] = None
    ) -> ExtractionResult:
        """
        Extract content from PDF bytes

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Filename for metadata
            pages: Optional list of page numbers to extract

        Returns:
            ExtractionResult containing markdown content and metadata
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = Path(tmp_file.name)

        try:
            result = self.extract_from_file(tmp_path, pages, use_cache=False)
            # Update metadata with provided filename
            result.metadata["filename"] = filename
            result.metadata["file_path"] = filename
            return result
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    def clear_cache(self):
        """Clear all cached extraction results"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.md"):
                cache_file.unlink()
            logger.info("PDF extraction cache cleared")


# Convenience functions for easy usage
def extract_pdf_to_markdown(
    file_path: Union[str, Path],
    pages: Optional[List[int]] = None,
    cache_dir: Optional[Path] = None
) -> str:
    """
    Convenience function to extract PDF to markdown

    Args:
        file_path: Path to PDF file
        pages: Optional list of page numbers to extract
        cache_dir: Optional cache directory

    Returns:
        Markdown content as string
    """
    extractor = PDFExtractor(cache_dir=cache_dir)
    result = extractor.extract_from_file(file_path, pages)
    return result.markdown_content


def extract_pdf_with_metadata(
    file_path: Union[str, Path],
    pages: Optional[List[int]] = None,
    cache_dir: Optional[Path] = None
) -> ExtractionResult:
    """
    Convenience function to extract PDF with full metadata

    Args:
        file_path: Path to PDF file
        pages: Optional list of page numbers to extract
        cache_dir: Optional cache directory

    Returns:
        Complete ExtractionResult with metadata
    """
    extractor = PDFExtractor(cache_dir=cache_dir)
    return extractor.extract_from_file(file_path, pages)