"""
Report Generation Module
统一的学术论文报告生成系统

Based on Linus Torvalds' "Good Taste" principle:
- Eliminate special cases through proper data structure design
- Simple algorithm handles all variants
- Clean execution with minimal code
"""

from .core.config import ReportConfig, ResearchParadigm
from .core.context import ReportContext
from .engine.report_generator import ReportGenerator
from .engine.section_factory import SectionFactory

__version__ = "2.0.0"
__all__ = [
    "ReportConfig",
    "ResearchParadigm",
    "ReportContext",
    "ReportGenerator",
    "SectionFactory"
]