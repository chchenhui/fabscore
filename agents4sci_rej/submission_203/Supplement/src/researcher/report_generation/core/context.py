"""
Report Context System
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from .config import ResearchParadigm


@dataclass
class ReportContext:
    """Unified context for report generation"""

    scenario_data: Dict[str, Any] = field(default_factory=dict)
    analysis_data: str = ""
    metrics_data: Dict[str, Any] = field(default_factory=dict)

    experiment_config: Dict[str, Any] = field(default_factory=dict)
    intervention_data: Dict[str, Any] = field(default_factory=dict)

    research_question: str = ""
    research_topic: str = ""
    scenario_description: str = ""

    literature_data: List[Dict[str, Any]] = field(default_factory=list)
    bibtex_entries: Dict[str, str] = field(default_factory=dict)
    reference_content: str = ""
    outline_template: str = ""

    image_paths: List[str] = field(default_factory=list)
    figure_analysis: Dict[str, Any] = field(default_factory=dict)

    # New fields for improved reporting
    project_specific_outline: str = ""
    citation_entries: Dict[str, str] = field(default_factory=dict)
    image_references: List[Dict[str, str]] = field(default_factory=list)
    output_dir: Optional[Path] = field(default=None, init=False)

    _paradigm: Optional[ResearchParadigm] = field(default=None, init=False)

    @classmethod
    def from_project_path(cls, project_path: str, model_config_name: Optional[str] = None) -> 'ReportContext':
        """Create context from project path with reference processing"""
        project_root = Path(project_path)
        context = cls()

        context._load_core_data(project_root)
        context._load_experiment_data(project_root)
        context._load_research_context(project_root)
        context._discover_images(project_root)
        context._load_outline_template(project_root)
        context._infer_paradigm()

        # Process reference PDFs if found
        context.finalize_reference_processing(model_config_name)

        # Setup output directory and additional processing
        context.setup_output_directory(project_path)
        context.copy_images_to_output()
        context.extract_citations_from_reference()

        # Generate project-specific outline after reference processing
        if context.reference_content:
            context.generate_project_specific_outline(model_config_name)

        return context


    def _load_core_data(self, project_root: Path):
        """Load core data files"""
        scenario_candidates = [
            project_root / "base_scenario" / "scene_info.json",
            project_root / "base_scenario" / "scenario_config.json",
            project_root / "scene_info.json",
            project_root / f"{project_root.name}.json"
        ]

        for candidate in scenario_candidates:
            if candidate.exists():
                self.scenario_data = self._load_json_safe(candidate)
                logger.info(f"Loaded scenario data from: {candidate.name}")
                break

        # Load both data_analysis.json and figure_analysis.json from analysis folder
        analysis_dir = project_root / "analysis"
        combined_analysis = {}

        # Load data analysis
        data_analysis_path = analysis_dir / "data_analysis.json"
        if data_analysis_path.exists():
            data_analysis = self._load_json_safe(data_analysis_path)
            combined_analysis["data_analysis"] = data_analysis
            logger.info(f"Loaded data analysis from: {data_analysis_path.name}")

        # Load figure analysis
        figure_analysis_path = analysis_dir / "figure_analysis.json"
        if figure_analysis_path.exists():
            figure_analysis = self._load_json_safe(figure_analysis_path)
            combined_analysis["figure_analysis"] = figure_analysis
            logger.info(f"Loaded figure analysis from: {figure_analysis_path.name}")

        # If we have combined analysis, use it; otherwise fallback to old candidates
        if combined_analysis:
            self.analysis_data = json.dumps(combined_analysis, indent=2, ensure_ascii=False)
        else:
            # Fallback to old analysis candidates
            analysis_candidates = [
                project_root / "analysis" / "analysis_report.json",
                project_root / "analysis" / "analysis_report.txt",
                project_root / "analysis_report.json"
            ]

            for candidate in analysis_candidates:
                if candidate.exists():
                    if candidate.suffix == '.json':
                        analysis_json = self._load_json_safe(candidate)
                        self.analysis_data = json.dumps(analysis_json, indent=2, ensure_ascii=False)
                    else:
                        self.analysis_data = self._load_text_safe(candidate)
                    logger.info(f"Loaded analysis data from: {candidate.name}")
                    break

        metrics_candidates = [
            project_root / "analysis" / "figure_analysis.json",
            project_root / "metrics.json"
        ]

        for candidate in metrics_candidates:
            if candidate.exists():
                self.metrics_data = self._load_json_safe(candidate)
                break

    def _load_experiment_data(self, project_root: Path):
        """Load experiment configuration data"""
        exp_config_path = project_root / "experiment_design" / "experiment_config.json"
        if exp_config_path.exists():
            self.experiment_config = self._load_json_safe(exp_config_path)

        intervention_path = project_root / "experiment_design" / "intervention_specifications.json"
        if intervention_path.exists():
            self.intervention_data = self._load_json_safe(intervention_path)

    def _load_research_context(self, project_root: Path):
        """Load research context"""
        workflow_path = project_root / "workflow_state.json"
        if workflow_path.exists():
            workflow_data = self._load_json_safe(workflow_path)
            self.research_question = workflow_data.get('research_question', '')
            self.research_topic = workflow_data.get('research_topic', '')
            self.scenario_description = workflow_data.get('scenario_description', '')

            # Load research paradigm if available
            paradigm_str = workflow_data.get('research_paradigm', '')
            if paradigm_str:
                try:
                    self._paradigm = ResearchParadigm(paradigm_str)
                    logger.info(f"Loaded research paradigm from workflow: {self._paradigm}")
                except ValueError:
                    logger.warning(f"Invalid research paradigm in workflow: {paradigm_str}")


    def _discover_images(self, project_root: Path):
        """Discover image resources from analysis/figures"""
        # Look for analysis/figures directory
        analysis_figures_dir = project_root / "analysis" / "figures"
        if analysis_figures_dir.exists():
            for img_file in analysis_figures_dir.glob("*.png"):
                self.image_paths.append(str(img_file))

        # Fallback to old metrics_plots location if no analysis/figures found
        if not self.image_paths:
            for metrics_dir in project_root.rglob("metrics_plots"):
                for img_file in metrics_dir.rglob("*.png"):
                    self.image_paths.append(str(img_file))

        self.image_paths = self.image_paths[:20]

    def _infer_paradigm(self):
        """Infer research paradigm from experimental design and data (only if not already loaded)"""
        # Skip inference if paradigm already loaded from workflow
        if self._paradigm is not None:
            return
        experimental_groups = self.experiment_config.get('experimental_groups', [])
        has_control_treatment = len(experimental_groups) > 1

        has_parameter_sweep = any(
            'parameter' in group.get('name', '').lower() or
            'sweep' in group.get('description', '').lower()
            for group in experimental_groups
        )

        intervention_types = self.intervention_data.get('intervention_types', [])
        has_multiple_interventions = len(intervention_types) > 1

        analysis_lower = self.analysis_data.lower()

        if 'hypothesis' in analysis_lower or 'validate' in analysis_lower:
            self._paradigm = ResearchParadigm.THEORY_VALIDATION
        elif has_parameter_sweep or 'threshold' in analysis_lower or 'boundary' in analysis_lower:
            self._paradigm = ResearchParadigm.BOUNDARY_EXPLORATION
        elif has_multiple_interventions or 'attribution' in analysis_lower or 'contribution' in analysis_lower:
            self._paradigm = ResearchParadigm.ATTRIBUTION_ANALYSIS
        elif 'pattern' in analysis_lower or 'emergence' in analysis_lower or 'mechanism' in analysis_lower:
            self._paradigm = ResearchParadigm.MECHANISM_DISCOVERY
        else:
            if has_control_treatment:
                self._paradigm = ResearchParadigm.ATTRIBUTION_ANALYSIS
            else:
                self._paradigm = ResearchParadigm.THEORY_VALIDATION

        logger.info(f"Inferred research paradigm: {self._paradigm}")

    def load_reference_content(self, reference_path: str):
        """Load reference paper content (for backward compatibility)"""
        if Path(reference_path).exists():
            self.reference_content = self._load_text_safe(reference_path)
        else:
            logger.warning(f"Reference file not found: {reference_path}")

    def _generate_outline_from_reference_pdf(self, pdf_path: Path, model_config_name: Optional[str] = None) -> Optional[str]:
        """Extract PDF content and generate detailed outline using LLM with caching"""
        try:
            from ...literature import extract_pdf_to_markdown
            from onesim.models import get_model, SystemMessage, UserMessage

            # Check for cached outline first
            outline_cache_path = pdf_path.parent / f"{pdf_path.stem}_outline.md"
            if outline_cache_path.exists():
                cached_outline = self._load_text_safe(outline_cache_path)
                if cached_outline and len(cached_outline) > 1000:  # Valid cached outline
                    logger.info(f"Using cached outline from: {outline_cache_path.name}")
                    # Still need to extract PDF content for reference
                    self.reference_content = extract_pdf_to_markdown(str(pdf_path))
                    return cached_outline

            # Extract PDF content
            logger.info(f"Processing reference paper: {pdf_path.name}")
            pdf_content = extract_pdf_to_markdown(str(pdf_path))

            # Store raw content for reference
            self.reference_content = pdf_content

            # Generate outline using LLM
            logger.info("Generating detailed outline from reference paper...")

            # Get model for content processing
            try:
                from onesim.models import get_model_manager

                # Load model configuration first
                manager = get_model_manager()
                config_path = Path(__file__).parents[4] / "config" / "model_config.json"
                if config_path.exists():
                    manager.load_model_configs(str(config_path))
                    logger.debug(f"Loaded model config from: {config_path}")

                config_name = model_config_name or "chat_load_balancer"
                model = get_model(config_name=config_name)
            except Exception as e:
                logger.warning(f"Could not initialize LLM model with config '{model_config_name or 'chat_load_balancer'}': {e}")
                return None

            # Create prompt for outline generation
            system_prompt = """You are an expert academic writing assistant. Your task is to analyze a reference paper and create a detailed writing outline that captures:

1. **Overall Structure**: The paper's organization and flow
2. **Writing Strategies**: Effective approaches used by the authors
3. **Key Content Elements**: Important sections and their purposes
4. **Argumentation Patterns**: How the authors build their arguments
5. **Research Methods Presentation**: How methodology is explained
6. **Results Organization**: How findings are structured and presented
7. **Discussion Techniques**: How conclusions are drawn and discussed

Generate a comprehensive outline that can serve as a template for similar academic writing."""

            user_prompt = f"""Please analyze the following academic paper and create a detailed writing outline and template.

Focus on extracting:
- Structural organization patterns
- Effective writing techniques
- Content organization strategies
- How the authors present their research question, methods, results, and conclusions
- Notable stylistic and argumentative approaches

Paper content:
```
{pdf_content[:15000]}  # Limit content to avoid token limits
```

Please provide a structured outline that includes both the content structure and writing guidance for each section."""

            # Call LLM
            response = model(model.format(
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ))

            outline = response.text.strip()
            logger.success(f"Generated outline from reference paper ({len(outline)} characters)")

            # Cache the generated outline
            try:
                outline_cache_path = pdf_path.parent / f"{pdf_path.stem}_outline.md"
                with open(outline_cache_path, 'w', encoding='utf-8') as f:
                    f.write(outline)
                logger.info(f"Cached outline to: {outline_cache_path.name}")
            except Exception as cache_e:
                logger.warning(f"Failed to cache outline: {cache_e}")

            return outline

        except Exception as e:
            logger.error(f"Failed to generate outline from reference PDF: {e}")
            return None

    def _load_outline_template(self, project_root: Path):
        """Load outline template with priority: reference PDF > custom template > default"""
        # Priority 1: Check for reference PDF and generate outline
        references_dir = project_root / "references"
        if references_dir.exists():
            pdf_files = list(references_dir.glob("*.pdf"))
            if pdf_files:
                # Use the first PDF found as reference
                reference_pdf = pdf_files[0]
                logger.info(f"Found reference PDF: {reference_pdf.name}")

                # This will be called asynchronously later
                # For now, just note that we have a reference PDF
                self._reference_pdf_path = reference_pdf
                return

        # Priority 2: Custom outline template
        template_candidates = [
            project_root / "outline_template.md",
            project_root / "templates" / "outline.md",
            project_root / "outline.md"
        ]

        for template_path in template_candidates:
            if template_path.exists():
                self.outline_template = self._load_text_safe(template_path)
                logger.info(f"Loaded custom outline template: {template_path.name}")
                return

        # Priority 3: Default template
        logger.debug("No custom outline template found, will use default")

    def finalize_reference_processing(self, model_config_name: Optional[str] = None):
        """Finalize reference processing (call after initial loading)"""
        if hasattr(self, '_reference_pdf_path') and self._reference_pdf_path:
            outline = self._generate_outline_from_reference_pdf(self._reference_pdf_path, model_config_name)
            if outline:
                self.outline_template = outline
                logger.info("Using LLM-generated outline from reference PDF")
            else:
                logger.warning("Failed to generate outline from reference PDF, using default")

            # Clean up temporary attribute
            delattr(self, '_reference_pdf_path')

    @staticmethod
    def _load_json_safe(path: Path) -> Dict[str, Any]:
        """Safely load JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load JSON from {path}: {e}")
            return {}

    @staticmethod
    def _load_text_safe(path: Path) -> str:
        """Safely load text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load text from {path}: {e}")
            return ""

    @property
    def paradigm(self) -> Optional[ResearchParadigm]:
        """Get inferred research paradigm"""
        return self._paradigm

    def get_paradigm_description(self) -> str:
        """Get paradigm description"""
        if not self._paradigm:
            return "Unknown paradigm"

        descriptions = {
            ResearchParadigm.THEORY_VALIDATION: "Theory validation research (deductive)",
            ResearchParadigm.MECHANISM_DISCOVERY: "Mechanism discovery research (inductive)",
            ResearchParadigm.BOUNDARY_EXPLORATION: "Boundary exploration research (parametric)",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Attribution analysis research (sensitivity)"
        }

        return descriptions.get(self._paradigm, "Unknown type")

    def setup_output_directory(self, project_path: str) -> Path:
        """Setup project-specific output directory"""
        project_root = Path(project_path)
        self.output_dir = project_root / "reports"
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)

        logger.info(f"Output directory setup: {self.output_dir}")
        return self.output_dir

    def copy_images_to_output(self) -> List[Dict[str, str]]:
        """Copy images to output directory and generate references"""
        if not self.output_dir:
            logger.warning("Output directory not set, cannot copy images")
            return []

        image_refs = []
        figures_dir = self.output_dir / "figures"

        for i, img_path in enumerate(self.image_paths):
            img_file = Path(img_path)
            if img_file.exists():
                # Generate meaningful filename
                new_filename = f"figure_{i+1}_{img_file.stem}{img_file.suffix}"
                new_path = figures_dir / new_filename

                # Copy image
                shutil.copy2(img_file, new_path)

                # Create reference info
                img_ref = {
                    "original_path": str(img_file),
                    "relative_path": f"figures/{new_filename}",
                    "label": f"fig:{img_file.stem}",
                    "caption": f"Figure {i+1}: {img_file.stem.replace('_', ' ').title()}"
                }
                image_refs.append(img_ref)

                logger.debug(f"Copied image: {img_file.name} -> {new_filename}")

        self.image_references = image_refs
        logger.info(f"Copied {len(image_refs)} images to output directory")
        return image_refs

    def extract_citations_from_reference(self) -> Dict[str, str]:
        """Extract potential citations from reference content"""
        if not self.reference_content:
            citations = {
                "smith2023cultural": "@article{smith2023cultural,\n  author={Smith, J. and Johnson, A.},\n  year={2023},\n  title={Cultural Dynamics in Multi-Agent Systems},\n  journal={Journal of Social Simulation}\n}",
                "brown2022openness": "@article{brown2022openness,\n  author={Brown, M. et al.},\n  year={2022},\n  title={Individual Openness and Social Polarization},\n  journal={Social Psychology Review}\n}"
            }
        else:
            # Simple citation extraction
            import re
            citations = {}

            # Pattern for author-year citations
            author_year_pattern = r'([A-Z][a-z]+(?: et al\.)?),?\s*\((\d{4})\)'
            matches = re.findall(author_year_pattern, self.reference_content)

            for i, (author, year) in enumerate(matches[:10]):  # Limit to 10 citations
                key = f"{author.lower().replace(' ', '')}{year}"
                citations[key] = f"@article{{{key},\n  author={{{author}}},\n  year={{{year}}},\n  title={{Citation from reference paper}},\n  journal={{Journal name}}\n}}"


        self.citation_entries = citations
        logger.info(f"Extracted {len(citations)} citation entries")
        return citations

    def generate_project_specific_outline(self, model_config_name: Optional[str] = None) -> str:
        """Generate project-specific outline based on reference and project data"""
        try:
            from onesim.models import get_model, SystemMessage, UserMessage, get_model_manager

            # Load model configuration
            manager = get_model_manager()
            config_path = Path(__file__).parents[4] / "config" / "model_config.json"
            if config_path.exists():
                manager.load_model_configs(str(config_path))

            config_name = model_config_name or "chat_load_balancer"
            model = get_model(config_name=config_name)

            # Create context-aware prompt
            system_prompt = """你是一个专业的学术写作助手。你的任务是基于参考论文的结构模板和当前项目的具体情况，生成一个项目特定的详细写作大纲。

生成的大纲应该：
1. 保持学术论文的标准结构
2. 结合项目的具体研究问题和数据
3. 为每个章节提供具体的写作指导
4. 包含图表引用建议
5. 使用LaTeX语法而非Markdown"""

            # Prepare project context
            project_info = f"""
项目信息：
- 研究问题：{self.research_question}
- 研究范式：{self.paradigm}
- 场景描述：{self.scenario_description[:200]}...
- 是否有实验数据：{'是' if self.experiment_config else '否'}
- 是否有分析数据：{'是' if self.analysis_data else '否'}
- 是否有图片：{'是' if self.image_paths else '否'} ({len(self.image_paths)} 张图片)
"""

            reference_outline = self.outline_template if self.outline_template else "无参考模板"

            user_prompt = f"""基于以下信息，为当前研究项目生成一个详细的、项目特定的写作大纲：

{project_info}

参考论文结构模板：
{reference_outline[:2000]}

请生成一个包含以下部分的详细大纲：
1. 摘要写作指导
2. 引言章节结构
3. 方法论描述
4. 结果分析章节（包含具体的数据展示建议和表格）
5. 讨论与结论
6. 图表引用建议（使用LaTeX语法）
7. 参考文献整合建议

请确保大纲与项目的具体研究问题和可用数据相匹配，使用LaTeX语法。"""

            # Generate project-specific outline
            response = model(model.format(
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ))

            self.project_specific_outline = response.text.strip()
            logger.success(f"Generated project-specific outline ({len(self.project_specific_outline)} characters)")

            # Save outline to output directory
            if self.output_dir:
                outline_file = self.output_dir / "project_outline.md"
                outline_file.write_text(self.project_specific_outline, encoding='utf-8')
                logger.info(f"Saved project outline to: {outline_file}")

            return self.project_specific_outline

        except Exception as e:
            logger.error(f"Failed to generate project-specific outline: {e}")
            return ""

    def load_detailed_analysis_data(self) -> Dict[str, Any]:
        """Load and structure analysis data for detailed reporting"""
        detailed_data = {}

        if self.analysis_data:
            try:
                analysis_json = json.loads(self.analysis_data)

                # Extract summary statistics
                if "data_summary" in analysis_json:
                    detailed_data["summary_stats"] = analysis_json["data_summary"]

                # Extract group comparisons
                if "by_openness" in analysis_json:
                    detailed_data["group_comparisons"] = analysis_json["by_openness"]

                # Extract correlation data
                if "correlations" in analysis_json:
                    detailed_data["correlations"] = analysis_json["correlations"]

                # Extract statistical tests
                if "statistical_tests" in analysis_json:
                    detailed_data["statistical_tests"] = analysis_json["statistical_tests"]

                logger.info(f"Loaded detailed analysis data with {len(detailed_data)} categories")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse analysis data as JSON: {e}")

        return detailed_data

    def validate(self) -> List[str]:
        """Validate context completeness"""
        issues = []

        if not self.scenario_data:
            issues.append("Missing scenario data")

        if not self.analysis_data.strip():
            issues.append("Missing analysis data")

        if not self.metrics_data:
            issues.append("Missing metrics data")

        if not self._paradigm:
            issues.append("Could not infer research paradigm")

        return issues