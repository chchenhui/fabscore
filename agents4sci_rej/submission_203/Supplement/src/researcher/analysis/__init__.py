"""
Package exports for researcher.analysis

Provides AnalysisCoordinator class for backward compatibility with researcher.py,
wrapping the new coordinate_analysis pipeline in coordinator.py.
"""

import os
from typing import Optional, Dict, Any

# Re-export the functional pipeline
from .coordinator import coordinate_analysis  # noqa: F401


class AnalysisCoordinator:
    """
    Compatibility wrapper used by researcher.py:
    - Accepts project_dir, model_name, model_config_path
    - Optional results_dir attribute (kept for compatibility; currently unused by the new pipeline)
    - run_comprehensive_analysis triggers the full 4-stage pipeline via coordinate_analysis
    """

    def __init__(
        self,
        project_dir: str,
        model_name: Optional[str] = None,
        model_config_path: Optional[str] = None,
    ) -> None:
        self.project_dir = project_dir
        self.model_name = model_name or "openai-gpt4o"
        self.model_config_path = model_config_path
        # Kept for compatibility with previous version (researcher.py sets this)
        self.results_dir: Optional[str] = None

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline and return a summary dict.
        This will:
        - Collect scene metrics (latest runs per group)
        - Generate 3 figures
        - Produce figure_analysis.json
        - Run LLM stage3 to produce data_analysis.json (saved under analysis/data/)
        """
        # If a custom model_config_path is provided, ensure stage3 can see it
        if self.model_config_path:
            os.environ["ONESIM_MODEL_CONFIG_PATH"] = self.model_config_path

        # Delegate to the functional pipeline; project_dir may be an absolute path
        # coordinate_analysis resolves absolute project_dir correctly.
        summary = coordinate_analysis(
            project_name=self.project_dir,
            config_name=self.model_name,
            latest_runs=1,
        )
        return summary


__all__ = ["AnalysisCoordinator", "coordinate_analysis"]