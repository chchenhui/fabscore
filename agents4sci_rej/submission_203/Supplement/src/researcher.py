#!/usr/bin/env python3
"""
Unified Researcher Entry Point for OneSim

This script provides a unified interface for the complete research workflow:
1. Environment Design (env_design) - Creates simulation environments from research topics
2. Experiment Execution (experiment_execution) - Runs experiments with interventions
3. Report Generation (report_generation) - Analyzes results and generates reports

The script creates projects under the '../projects/' directory and manages the complete
research lifecycle in accordance with OneSim platform standards.

Usage:
    python src/researcher.py --topic "social network dynamics" --project-name "my_research"
    python src/researcher.py --config config/researcher_config.json
    python src/researcher.py --help
"""

import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback
from enum import Enum

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from onesim.models import get_model_manager

# Import researcher components
from researcher.env_design_cli import Coordinator as EnvDesignCoordinator
from researcher.scenario_creation_cli import ScenarioCreationPipeline
from researcher.experiment_platform.experiment_project import ExperimentProject, ProjectConfig


class ResearchParadigm(Enum):
    """研究范式枚举定义"""
    AUTO_AGENT = "auto_agent"                    # 现有的Agent主导模式
    THEORY_VALIDATION = "theory_validation"      # 理论验证型 (T+C → O)
    MECHANISM_DISCOVERY = "mechanism_discovery"  # 机制发现型 (O+C → T)
    BOUNDARY_EXPLORATION = "boundary_exploration"# 边界探索型 (T+O → C边界)
    ATTRIBUTION_ANALYSIS = "attribution_analysis"# 归因分析型 (T+O → C权重)


class ResearcherWorkflow:
    """
    Unified researcher workflow coordinator that manages the complete research pipeline.
    
    This class orchestrates env_design, experiment_execution, and report_generation
    to provide a seamless research experience.
    """
    
    def __init__(
        self,
        project_name: str,
        scenario_description: Optional[str] = None,
        research_question: Optional[str] = None,
        research_paradigm: str = "auto_agent",
        theory: Optional[str] = None,
        observation: Optional[str] = None,
        condition: Optional[str] = None,
        research_topic: Optional[str] = None,
        model_name: Optional[str] = None,
        model_config_path: Optional[str] = None,
        projects_base_dir: Optional[str] = None
    ):
        """
        Initialize the researcher workflow.
        
        Args:
            project_name: Name of the research project
            scenario_description: Scenario description in natural language
            research_question: Research question in natural language
            research_paradigm: Research paradigm type
            research_topic: Research topic for environment design (backwards compatibility)
            model_name: Model configuration name to use
            model_config_path: Path to model configuration file
            projects_base_dir: Base directory for projects (defaults to ../projects)
        """
        self.project_name = project_name
        
        # 处理新的范式输入参数
        try:
            self.research_paradigm = ResearchParadigm(research_paradigm)
        except ValueError:
            logger.warning(f"Unknown research paradigm: {research_paradigm}, using AUTO_AGENT")
            self.research_paradigm = ResearchParadigm.AUTO_AGENT
        
        # 向后兼容处理：如果提供了research_topic但没有scenario_description
        if research_topic and not scenario_description:
            self.scenario_description = research_topic
            self.research_question = research_question or f"探索{research_topic}的内在机制和规律"
            logger.info("使用兼容模式：从research_topic生成scenario_description")
        else:
            self.scenario_description = scenario_description
            self.research_question = research_question
        
        # 保留原有字段用于兼容性
        self.research_topic = research_topic or scenario_description
        self.model_name = model_name
        self.model_config_path = model_config_path
        
        # Setup project paths - go up one level from src/ to project root
        self.base_dir = Path(__file__).resolve().parent.parent
        self.projects_base_dir = Path(projects_base_dir) if projects_base_dir else self.base_dir / "projects"
        self.project_dir = self.projects_base_dir / project_name
        
        # Ensure projects directory exists
        self.projects_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env_coordinator = None
        self.experiment_project = None
        self.scenario_pipeline = None
        
        # Workflow state - enhanced with paradigm information
        self.workflow_state = {
            "project_name": project_name,
            "research_paradigm": self.research_paradigm.value,
            "scenario_description": self.scenario_description,
            "research_question": self.research_question,
            "theory": theory,
            "observation": observation,
            "condition": condition,
            "research_topic": research_topic,  # 保留用于兼容性
            "created_timestamp": datetime.now().isoformat(),
            "status": "initialized",
            "phases_completed": [],
            "scene_name": None,
            "scene_path": None,
            "simulation_results": {},
            "report_paths": {}
        }
    
    def setup_model_manager(self) -> bool:
        """
        Setup the model manager with configuration.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            if not self.model_config_path:
                # Try default model config paths
                default_paths = [
                    self.base_dir / "config" / "model_config.json",
                ]
                
                for path in default_paths:
                    if path.exists():
                        self.model_config_path = str(path)
                        break
                
                if not self.model_config_path:
                    logger.error("No model configuration file found. Please specify --model-config")
                    return False
            
            model_manager = get_model_manager()
            model_manager.load_model_configs(self.model_config_path)
            logger.info(f"Loaded model configurations from: {self.model_config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model manager: {e}")
            return False
    
    def create_project_structure(self) -> bool:
        """
        Create the project directory structure following OneSim standards.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Creating project structure at: {self.project_dir}")
            
            # Create main project directory
            self.project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard project subdirectories
            subdirs = [
                "experiment_design",
                "base_scenario", 
                "groups",
                "analysis",
                "reports"
            ]
            
            for subdir in subdirs:
                (self.project_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create project README with design plan
            readme_path = self.project_dir / "README.md"
            if not readme_path.exists():
                readme_content = self._generate_enhanced_readme()
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
            
            logger.info("✓ Project structure created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project structure: {e}")
            return False
    
    def _get_paradigm_display_name(self) -> str:
        """Get research paradigm display name"""
        paradigm_names = {
            ResearchParadigm.AUTO_AGENT: "Agent-Led Exploration",
            ResearchParadigm.THEORY_VALIDATION: "Theory Validation Research",
            ResearchParadigm.MECHANISM_DISCOVERY: "Mechanism Discovery Research",
            ResearchParadigm.BOUNDARY_EXPLORATION: "Boundary Exploration Research",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Attribution Analysis Research"
        }
        return paradigm_names.get(self.research_paradigm, self.research_paradigm.value)
    
    def _generate_enhanced_readme(self) -> str:
        """Generate enhanced README with design plan integrated"""
        paradigm_descriptions = {
            ResearchParadigm.AUTO_AGENT: "AI agent-led autonomous exploration research",
            ResearchParadigm.THEORY_VALIDATION: "Validate existing theory predictions through simulation",
            ResearchParadigm.MECHANISM_DISCOVERY: "Discover underlying mechanisms from observed phenomena", 
            ResearchParadigm.BOUNDARY_EXPLORATION: "Explore boundary conditions of theories and phenomena",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Analyze relative influence weights of different factors"
        }
        
        return f"""# {self.project_name}

## Project Overview

- **Research Paradigm**: {self._get_paradigm_display_name()}
- **Paradigm Description**: {paradigm_descriptions.get(self.research_paradigm, "Autonomous exploration research")}
- **Scenario Description**: {self.scenario_description or 'Not specified'}
- **Research Question**: {self.research_question or 'Not specified'}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Research Design Plan

### Phase 1: Environment Design (env_design)
- **Goal**: Build simulation environment based on scenario description and research question
- **Paradigm Adaptation**: {self._get_paradigm_design_strategy()}
- **Expected Outputs**: 
  - Scene configuration files
  - Agent behavior rules
  - Experimental design specifications

### Phase 2: Scenario Creation (scenario_creation)  
- **Goal**: Generate complete executable simulation scenario
- **Approach**: Consistent with existing workflow
- **Expected Outputs**: Complete simulation environment code

### Phase 3: Experiment Execution (experiment_execution)
- **Goal**: Execute simulation experiments and collect data
- **Experimental Design**: {self._get_paradigm_experiment_design()}
- **Expected Outputs**: Experimental data and preliminary results

### Phase 4: Data Analysis (analysis)
- **Goal**: Deep analysis of experimental results
- **Analysis Focus**: {self._get_paradigm_analysis_focus()}
- **Expected Outputs**: Statistical analysis and pattern recognition results

### Phase 5: Report Generation (report_generation)
- **Goal**: Generate research reports
- **Report Style**: {self._get_paradigm_report_style()}
- **Expected Outputs**: Complete research report documents

## Project Structure

```
{self.project_name}/
├── README.md                   # This file with design plan
├── experiment_design/          # Experimental design files
├── base_scenario/             # Base scenario configuration
├── groups/                    # Experimental group data
├── analysis/                  # Analysis results
└── reports/                   # Generated reports
```

## Technical Configuration

### Model Configuration
- **Model Name**: {self.model_name or "Default"}
- **Config Path**: {self.model_config_path or "Default path"}

## Usage

### Full Workflow Execution
```bash
python src/researcher.py \\
    --project_name "{self.project_name}" \\
    --scenario "{self.scenario_description or ''}" \\
    --question "{self.research_question or ''}" \\
    --paradigm "{self.research_paradigm.value}"
```

### Phase-by-Phase Execution
```bash
# Environment design only
python src/researcher.py --project_name "{self.project_name}" --phase design

# Complete scenario creation
python src/researcher.py --project_name "{self.project_name}" --phase scenario

# Experiment and analysis
python src/researcher.py --project_name "{self.project_name}" --phase execute
python src/researcher.py --project_name "{self.project_name}" --phase analysis

# Report generation
python src/researcher.py --project_name "{self.project_name}" --phase report
```

## Expected Outcomes

{self._get_paradigm_expected_outcomes()}

---

*Generated by OneSim Multi-Paradigm Research Workflow*
"""

    def _get_paradigm_design_strategy(self) -> str:
        """Get paradigm-specific design strategy description"""
        strategies = {
            ResearchParadigm.AUTO_AGENT: "Adaptive environment design with AI exploring optimal research paths",
            ResearchParadigm.THEORY_VALIDATION: "Model known theoretical mechanisms, design validation experiments",
            ResearchParadigm.MECHANISM_DISCOVERY: "Reproduce observed phenomena, design mechanism exploration experiments",
            ResearchParadigm.BOUNDARY_EXPLORATION: "Parametric modeling, design boundary scanning experiments", 
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Multi-factor modeling, design sensitivity analysis experiments"
        }
        return strategies.get(self.research_paradigm, "Adaptive environment design")

    def _get_paradigm_experiment_design(self) -> str:
        """Get paradigm-specific experiment design description"""
        designs = {
            ResearchParadigm.AUTO_AGENT: "Adaptive experimental design with agent-led exploration",
            ResearchParadigm.THEORY_VALIDATION: "Controlled experiments to validate theoretical predictions",
            ResearchParadigm.MECHANISM_DISCOVERY: "Exploratory experiments to discover hidden patterns",
            ResearchParadigm.BOUNDARY_EXPLORATION: "Parameter scanning experiments to find critical points",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Factor analysis experiments to measure relative influences"
        }
        return designs.get(self.research_paradigm, "Adaptive experimental design")

    def _get_paradigm_analysis_focus(self) -> str:
        """Get paradigm-specific analysis focus description"""
        focuses = {
            ResearchParadigm.AUTO_AGENT: "Comprehensive data analysis and pattern discovery",
            ResearchParadigm.THEORY_VALIDATION: "Consistency analysis between theoretical predictions and experimental results",
            ResearchParadigm.MECHANISM_DISCOVERY: "Data pattern recognition and mechanism induction analysis", 
            ResearchParadigm.BOUNDARY_EXPLORATION: "Phase transition detection and boundary condition analysis",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Factor importance ranking and interaction effect analysis"
        }
        return focuses.get(self.research_paradigm, "Comprehensive data analysis")

    def _get_paradigm_report_style(self) -> str:
        """Get paradigm-specific report style description"""
        styles = {
            ResearchParadigm.AUTO_AGENT: "Comprehensive research report highlighting discoveries and insights",
            ResearchParadigm.THEORY_VALIDATION: "Validation research report highlighting theoretical testing results",
            ResearchParadigm.MECHANISM_DISCOVERY: "Discovery research report highlighting new mechanism proposals",
            ResearchParadigm.BOUNDARY_EXPLORATION: "Exploratory research report highlighting boundary condition discoveries",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: "Analytical research report highlighting factor influence weights"
        }
        return styles.get(self.research_paradigm, "Comprehensive research report")

    def _get_paradigm_expected_outcomes(self) -> str:
        """Get paradigm-specific expected outcomes"""
        outcomes = {
            ResearchParadigm.AUTO_AGENT: """
### Agent-Led Exploration Expected Outcomes
- 🎯 Deep analysis of research questions
- 🎯 Key findings and insights
- 🎯 Practical recommendations and guidance
- 🎯 Future research directions
""",
            ResearchParadigm.THEORY_VALIDATION: """
### Theory Validation Expected Outcomes
- ✅ Theoretical prediction accuracy assessment
- ✅ Theory applicability boundary identification  
- ✅ Theory improvement recommendations
- ✅ Validation experiment protocols
""",
            ResearchParadigm.MECHANISM_DISCOVERY: """
### Mechanism Discovery Expected Outcomes
- 🔍 Underlying mechanisms behind phenomena
- 🔍 Key driving factor identification
- 🔍 Mechanism model construction
- 🔍 New hypothesis proposals
""",
            ResearchParadigm.BOUNDARY_EXPLORATION: """
### Boundary Exploration Expected Outcomes
- 🎯 Critical parameter threshold values
- 🎯 Phase transition boundary maps
- 🎯 Stable region identification
- 🎯 Parameter optimization recommendations
""",
            ResearchParadigm.ATTRIBUTION_ANALYSIS: """
### Attribution Analysis Expected Outcomes
- 📊 Factor importance rankings
- 📊 Relative influence weights
- 📊 Interaction effect analysis
- 📊 Optimal configuration recommendations
"""
        }
        return outcomes.get(self.research_paradigm, """
### Comprehensive Exploration Expected Outcomes
- 🎯 Deep analysis of research questions
- 🎯 Key findings and insights
- 🎯 Practical recommendations and guidance
- 🎯 Future research directions
""")
    
    def phase_environment_design(self) -> bool:
        """
        Execute the environment design phase with complete scenario specifications.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*80)
            logger.info("PHASE 1: ENVIRONMENT DESIGN")
            logger.info("="*80)
            
            if not self.research_topic:
                logger.error("Research topic is required for environment design phase")
                return False
            
            # Initialize environment design coordinator
            self.env_coordinator = EnvDesignCoordinator(
                scene_name=None,  # Let it auto-generate
                model_name=self.model_name,
                research_paradigm=self.research_paradigm.value,
                save_intermediate=True
            )
            
            # Run environment design
            logger.info(f"Starting environment design for paradigm: {self.research_paradigm.value}")
            self.input_data = {
                "research_topic": self.research_topic,
                "scenario_description": self.scenario_description,
                "research_question": self.research_question,
                "theory": self.workflow_state.get("theory"),
                "condition": self.workflow_state.get("condition"),
                "observation": self.workflow_state.get("observation")
            }
            self.env_coordinator.run(self.input_data)
            
            # Get the generated environment path
            scene_name = self.env_coordinator.scene_name
            if not scene_name:
                logger.error("Environment design failed - no scene name generated")
                return False
            
            env_path = self.base_dir / "src" / "envs" / scene_name
            self.workflow_state["scene_name"] = scene_name
            self.workflow_state["scene_path"] = str(env_path)
            
            # Copy/link experiment configs to project directory
            env_research_path = env_path / "research" / "env_design"
            project_experiment_path = self.project_dir / "experiment_design"
            
            config_files = [
                "experiment_config.json",
                "intervention_specifications.json", 
                "odd_protocol.json"
            ]
            
            for config_file in config_files:
                src_file = env_research_path / config_file
                dst_file = project_experiment_path / config_file
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"✓ Copied {config_file} to project")
            
            # Process and store detailed scenario specifications
            self._process_scenario_config(env_research_path)
            
            
            self.workflow_state["phases_completed"].append("environment_design")
            self.workflow_state["status"] = "environment_design_completed"
            
            # Save workflow state after phase completion
            self.save_workflow_state()
            
            logger.info("✓ Environment design phase completed successfully")
            logger.info("✓ Phase 1 context prepared for Phase 2 integration")
            return True
            
        except Exception as e:
            logger.error(f"Environment design phase failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _process_scenario_config(self, env_research_path: Path):
        """
        Process scenario config and ensure it's available in project directory.
        """
        try:
            scenario_config_path = env_research_path / "scenario_config.json"
            
            if scenario_config_path.exists():
                with open(scenario_config_path, 'r', encoding='utf-8') as f:
                    scenario_config = json.load(f)
                
                # Also save to project base_scenario for easy access
                project_scenario_path = self.project_dir / "base_scenario" / "scenario_config.json"
                with open(project_scenario_path, 'w', encoding='utf-8') as f:
                    json.dump(scenario_config, f, indent=2, ensure_ascii=False)
                
                logger.info("✓ Scenario configuration processed and saved to files")
            
        except Exception as e:
            logger.warning(f"Could not process scenario config: {e}")
    
    
    def phase_scenario_creation(self) -> bool:
        """
        Execute the complete scenario creation phase.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*80)
            logger.info("PHASE 2: SCENARIO CREATION")
            logger.info("="*80)
            
            # Load workflow state first to check if environment design is completed
            if not self.load_workflow_state():
                logger.error("No workflow state found. Please run environment design phase first.")
                return False
            
            # Check if environment design phase is completed
            if "environment_design" not in self.workflow_state.get("phases_completed", []):
                logger.error("Scenario creation must be run after environment design phase is completed")
                return False
            
            # Get scene name from workflow state
            scene_name = self.workflow_state.get("scene_name")
            if not scene_name:
                logger.error("Scene name not found in workflow state. Environment design may not have completed successfully.")
                return False
            
            logger.info(f"Using scene name from workflow state: {scene_name}")
            
            # Initialize scenario creation pipeline
            self.scenario_pipeline = ScenarioCreationPipeline(
                scene_name=scene_name,
                research_topic=self.research_topic or "",
                model_config_path=self.model_config_path,
                selected_model=self.model_name,
                base_path=str(self.base_dir),
                project_name=self.project_name,
                save_intermediate=True
            )
            
            # Run the scenario creation pipeline
            logger.info(f"Starting scenario creation for scene: {scene_name}")
            success = self.scenario_pipeline.run_full_pipeline()
            
            if not success:
                logger.error("Scenario creation pipeline failed")
                return False
            
            # Update workflow state with the new complete scenario path
            complete_env_path = self.base_dir / "src" / "envs" / scene_name
            self.workflow_state["scenario_path"] = str(complete_env_path)
            self.workflow_state["scenario_name"] = scene_name
            
            # Copy relevant files to project directory
            if complete_env_path.exists():
                # Copy scene_info.json from complete scenario
                scene_info_src = complete_env_path / "scene_info.json"
                scene_info_dst = self.project_dir / "base_scenario" / "scene_info.json"
                if scene_info_src.exists():
                    shutil.copy2(scene_info_src, scene_info_dst)
                    logger.info("✓ Copied complete scene_info.json to project")
            
            self.workflow_state["phases_completed"].append("scenario_creation")
            self.workflow_state["status"] = "scenario_creation_completed"
            
            # Save workflow state after phase completion
            self.save_workflow_state()
            
            logger.info("✓ Complete scenario creation phase completed successfully")
            logger.info(f"✓ Complete scenario created at: {complete_env_path}")
            return True
            
        except Exception as e:
            logger.error(f"Scenario creation phase failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
    def phase_experiment_execution(self) -> bool:
        """
        Execute the experiment execution phase.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*80)
            logger.info("PHASE 3: EXPERIMENT EXECUTION")
            logger.info("="*80)
            
            # Check for required experiment configurations
            experiment_config_path = self.project_dir / "experiment_design" / "experiment_config.json"
            intervention_specs_path = self.project_dir / "experiment_design" / "intervention_specifications.json"
            
            if not experiment_config_path.exists():
                logger.error(f"Experiment config not found: {experiment_config_path}")
                return False
            
            if not intervention_specs_path.exists():
                logger.error(f"Intervention specs not found: {intervention_specs_path}")
                return False
            
            # Setup experiment project configuration
            scene_path = self.workflow_state.get("scene_path")
            if not scene_path or not Path(scene_path).exists():
                logger.error("Environment path not found. Run environment design phase first.")
                return False
            
            # Create base config path (use OneSim standard config)
            base_config_path = self.base_dir / "config" / "config.json"
            if not base_config_path.exists():
                logger.error(f"Base configuration not found: {base_config_path}")
                return False
            
            # Load simulation config from files for Phase 3
            sim_config_for_phase3 = self._read_simulation_config_for_phase3() or {}
            
            # Create project configuration with simulation parameters
            project_config = ProjectConfig(
                project_name=self.project_name,
                project_description=f"Research project: {self.research_topic or 'Generated project'}",
                base_environment=str(self.project_dir),  # Use project dir as base
                base_config_path=str(base_config_path),
                intervention_specs_path=str(intervention_specs_path),
                output_directory=str(self.project_dir),
                model_name=self.model_name,
                model_config_path=self.model_config_path
            )
            
            # Apply simulation configuration to project if available
            if sim_config_for_phase3:
                logger.info(f"Applying Phase 1 simulation configuration: mode={sim_config_for_phase3.get('mode', 'ROUND')}, max_steps={sim_config_for_phase3.get('max_steps', 50)}")
            
            # Initialize experiment project
            self.experiment_project = ExperimentProject(project_config)
            
            # Run the complete experiment workflow
            logger.info("Starting experiment execution workflow...")
            
            workflow_results = self.experiment_project.run_full_workflow_from_config(
                scene_path=scene_path,
                experiment_config_path=str(experiment_config_path)
            )
            
            # Store results
            self.workflow_state["simulation_results"] = workflow_results
            self.workflow_state["phases_completed"].append("experiment_execution")
            self.workflow_state["status"] = "experiment_execution_completed"
            
            # Save workflow state after phase completion
            self.save_workflow_state()
            
            logger.info("✓ Experiment execution phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Experiment execution phase failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def phase_analysis(self) -> bool:
        """
        Execute the analysis phase.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*80)
            logger.info("PHASE 4: ANALYSIS")
            logger.info("="*80)
            
            # Import the analysis module
            from researcher.analysis import AnalysisCoordinator
            
            # Find experiment results dynamically from project structure
            experiment_results_path = self._find_experiment_results_path()
            if not experiment_results_path or not experiment_results_path.exists():
                logger.error("Experiment results not found. Run experiment execution phase first.")
                logger.info(f"Searched for results in: {experiment_results_path}")
                return False
            
            logger.info(f"Found experiment results at: {experiment_results_path}")
            
            # Initialize analysis coordinator
            analysis_coordinator = AnalysisCoordinator(
                project_dir=str(self.project_dir),
                model_name=self.model_name,
                model_config_path=self.model_config_path
            )
            
            # Set the results directory path dynamically
            analysis_coordinator.results_dir = experiment_results_path
            
            # Run analysis on the experiment results
            logger.info("Starting comprehensive data analysis...")
            analysis_results = analysis_coordinator.run_comprehensive_analysis()
            
            # Save analysis results to the project
            analysis_output_dir = self.project_dir / "analysis"
            analysis_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as json
            analysis_report_path = analysis_output_dir / "analysis_report.json"
            
            # save as proper json format
            report_content = self._format_analysis_results_as_json(analysis_results)
            
            with open(analysis_report_path, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Analysis results saved to: {analysis_report_path}")
            
            # Update workflow state
            self.workflow_state["analysis_results"] = str(analysis_report_path)
            self.workflow_state["phases_completed"].append("analysis")
            self.workflow_state["status"] = "analysis_completed"
            
            # Save workflow state after phase completion
            self.save_workflow_state()
            
            logger.info("✓ Analysis phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            logger.error(traceback.format_exc())
            
            # 保存错误信息到文本文件
            analysis_output_dir = self.project_dir / "analysis"
            analysis_output_dir.mkdir(parents=True, exist_ok=True)
            error_path = analysis_output_dir / "analysis_error.json"
            
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Analysis Error: {str(e)}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            logger.info(f"Error details saved to: {error_path}")
            return False
    
    def _find_experiment_results_path(self) -> Optional[Path]:
        """
        Find experiment results path dynamically from project structure.
        
        Returns:
            Path to metrics_plots directory if found, None otherwise
        """
        # First try to find from latest runs per group
        latest_runs = self._find_latest_runs_per_group()
        
        if latest_runs:
            # Use the first available run (could be improved to use a specific group)
            for group_name, run_path in latest_runs.items():
                metrics_path = Path(run_path) / "metrics_plots"
                if metrics_path.exists():
                    logger.info(f"Found metrics from group {group_name}: {metrics_path}")
                    return metrics_path
        
        # Fallback: search in groups directory
        groups_dir = self.project_dir / "groups"
        if groups_dir.exists():
            for group_dir in groups_dir.iterdir():
                if group_dir.is_dir():
                    runs_dir = group_dir / "runs"
                    if runs_dir.exists():
                        # Find latest run in this group
                        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                        if run_dirs:
                            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                            metrics_path = latest_run / "metrics_plots"
                            if metrics_path.exists():
                                logger.info(f"Found metrics via fallback search: {metrics_path}")
                                return metrics_path
        
        # If still not found, check results directory
        results_dir = self.project_dir / "results"
        if results_dir.exists():
            for results_subdir in results_dir.iterdir():
                if results_subdir.is_dir():
                    metrics_path = results_subdir / "metrics_plots"
                    if metrics_path.exists():
                        logger.info(f"Found metrics in results directory: {metrics_path}")
                        return metrics_path
        
        logger.warning("No experiment results found in project structure")
        return None
    
    def _format_analysis_results_as_json(self, results):
        """将分析结果格式化为JSON格式"""
        
        def convert_for_json(obj):
            """递归转换对象为JSON兼容格式"""
            if isinstance(obj, bool):
                return obj  # Python的bool在json.dump中实际是支持的
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # 处理自定义对象
                return convert_for_json(obj.__dict__)
            else:
                # 对于其他不可序列化的类型，转换为字符串
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        return {
            "report_title": "COMPREHENSIVE ANALYSIS REPORT",
            "generated_at": datetime.now().isoformat(),
            "analysis_results": convert_for_json(results),
            "format_version": "1.0"
        }
    
    def _format_analysis_results_as_text(self, results):
        """将分析结果格式化为可读的文本"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated at: {datetime.now()}")
        lines.append("")
        
        # 递归处理结果对象
        def format_value(value, indent=0):
            indent_str = "  " * indent
            if isinstance(value, dict):
                result = []
                for k, v in value.items():
                    result.append(f"{indent_str}{k}:")
                    result.extend(format_value(v, indent + 1))
                return result
            elif isinstance(value, list):
                result = []
                for i, item in enumerate(value):
                    result.append(f"{indent_str}[{i}]:")
                    result.extend(format_value(item, indent + 1))
                return result
            else:
                # 处理所有其他类型，包括复杂对象
                return [f"{indent_str}{str(value)}"]
        
        # 处理分析结果
        if isinstance(results, dict):
            for key, value in results.items():
                lines.append(f"## {key.upper().replace('_', ' ')}")
                lines.append("-" * 60)
                lines.extend(format_value(value))
                lines.append("")
        else:
            lines.append("Analysis Results:")
            lines.extend(format_value(results))
        
        return "\n".join(lines)
    
    def phase_report_generation(self) -> bool:
        """
        Execute the report generation phase using report_generation_cli.py.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*80)
            logger.info("PHASE 5: REPORT GENERATION")
            logger.info("="*80)

            # Import report generation components
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent))

            try:
                from researcher.report_generation import ReportGenerator, ReportConfig, ReportContext
            except ImportError as e:
                logger.error(f"Failed to import report generation modules: {e}")
                logger.error("Please ensure the report_generation package is properly installed")
                return False

            # Determine report title based on project info
            if self.project_name:
                report_title = f"{self.project_name.replace('_', ' ').title()} Research Report"
            elif self.workflow_state.get("scene_name"):
                scene_name = self.workflow_state.get("scene_name")
                report_title = f"{scene_name.replace('_', ' ').title()} Research Report"
            else:
                report_title = "OneSim Research Report"

            # Create report configuration
            config = ReportConfig(
                title=report_title,
                author="OneSim AI Research Assistant",
                language="zh",  # Default to Chinese, can be made configurable
                include_abstract=True,
                include_literature_review=True,
                include_bibliography=True,
                max_literature_papers=15,
                output_format="latex",
                enable_review=True,
                max_review_iterations=2,
                compile_pdf=True,
                latex_engine="xelatex",
                clean_aux_files=True,
                model_config_name=self.model_name
            )

            # Validate configuration
            config_issues = config.validate()
            if config_issues:
                logger.error("Configuration errors:")
                for issue in config_issues:
                    logger.error(f"  - {issue}")
                return False

            # Load project context
            logger.info(f"Loading project context from: {self.project_dir}")
            try:
                context = ReportContext.from_project_path(str(self.project_dir), config.model_config_name)
            except Exception as e:
                logger.error(f"Failed to load project context: {e}")
                return False

            # Validate context
            context_issues = context.validate()
            if context_issues:
                logger.error("Project context errors:")
                for issue in context_issues:
                    logger.error(f"  - {issue}")
                return False

            # Log inferred paradigm
            if context.paradigm:
                logger.info(f"Inferred research paradigm: {context.get_paradigm_description()}")

            # Generate report
            logger.info("Starting report generation...")
            generator = ReportGenerator(config)
            generated_files = generator.generate(context)

            if not generated_files:
                logger.error("No reports were generated")
                return False

            # Store generated report paths
            self.workflow_state["report_paths"] = {}
            for i, report_path in enumerate(generated_files):
                report_path = Path(report_path)
                if report_path.exists():
                    self.workflow_state["report_paths"][f"report_{i+1}"] = str(report_path)
                    logger.info(f"✓ Generated report [{i+1}]: {report_path}")
                else:
                    logger.warning(f"Report file not found: {report_path}")

            # Update workflow state
            self.workflow_state["phases_completed"].append("report_generation")
            self.workflow_state["status"] = "completed"
            self.workflow_state["final_reports"] = [str(p) for p in generated_files]

            # Save workflow state after phase completion
            self.save_workflow_state()

            logger.info("✓ Report generation phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"报告生成阶段失败: {e}")
            logger.error(traceback.format_exc())
            return False
        

    def _find_latest_runs_per_group(self) -> Dict[str, str]:
        """
        Find the latest run directory for each group in the project structure.
        
        Returns:
            Dict[str, str]: Mapping of group_name -> latest_run_path
        """
        latest_runs = {}
        
        if not self.project_dir.exists():
            return latest_runs
        
        groups_dir = self.project_dir / "groups"
        if not groups_dir.exists():
            return latest_runs
        
        for group_dir in groups_dir.iterdir():
            if group_dir.is_dir():
                group_name = group_dir.name
                runs_dir = group_dir / "runs"
                
                if runs_dir.exists():
                    latest_run_path = None
                    latest_timestamp = None
                    
                    for run_dir in runs_dir.iterdir():
                        if run_dir.is_dir():
                            # Use directory name (timestamp) or modification time
                            try:
                                # Try to parse timestamp from directory name
                                timestamp_str = run_dir.name
                                if timestamp_str.isdigit() or '_' in timestamp_str:
                                    # Use directory modification time as proxy
                                    mtime = run_dir.stat().st_mtime
                                    if latest_timestamp is None or mtime > latest_timestamp:
                                        latest_timestamp = mtime
                                        latest_run_path = run_dir
                            except Exception:
                                continue
                    
                    if latest_run_path:
                        latest_runs[group_name] = str(latest_run_path)
        
        return latest_runs
    
    
    def save_workflow_state(self):
        """Save the current workflow state to the project directory."""
        try:
            state_file = self.project_dir / "workflow_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.workflow_state, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Workflow state saved to: {state_file}")
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
    
    def load_workflow_state(self) -> bool:
        """
        Load existing workflow state from project directory.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            state_file = self.project_dir / "workflow_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                # Merge with current state, preserving existing values
                for key, value in loaded_state.items():
                    self.workflow_state[key] = value
                
                logger.info(f"✓ Loaded existing workflow state from: {state_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load workflow state: {e}")
            return False
    
    def _read_scenario_config_from_files(self) -> Optional[Dict]:
        """
        Read scenario config from files instead of workflow_state.
        
        Returns:
            Dict containing scenario config if found, None otherwise
        """
        try:
            # Try project base_scenario first
            project_scenario_path = self.project_dir / "base_scenario" / "scenario_config.json"
            if project_scenario_path.exists():
                with open(project_scenario_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not read scenario config from files: {e}")
            return None


    def _read_simulation_config_for_phase3(self) -> Optional[Dict]:
        """
        Read simulation config for Phase 3 from scenario config files.
        
        Returns:
            Dict containing simulation config if found, None otherwise
        """
        try:
            scenario_config = self._read_scenario_config_from_files()
            if scenario_config and "simulation_config" in scenario_config:
                sim_config = scenario_config["simulation_config"]
                return sim_config
            return None
        except Exception as e:
            logger.warning(f"Could not read simulation config for Phase 3: {e}")
            return None

    def _read_experiment_configs_from_files(self) -> Dict[str, str]:
        """
        Read experiment config file paths from project directory.
        
        Returns:
            Dict mapping config names to file paths
        """
        experiment_configs = {}
        project_experiment_path = self.project_dir / "experiment_design"
        
        config_files = [
            "experiment_config.json",
            "intervention_specifications.json", 
            "scenario_config.json",
            "odd_protocol.json"
        ]
        
        for config_file in config_files:
            config_path = project_experiment_path / config_file
            if config_path.exists():
                experiment_configs[config_file] = str(config_path)
                
        return experiment_configs

    
    
    
    def run_full_workflow(self) -> bool:
        """
        Execute the complete research workflow.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("="*100)
            logger.info("ONESIM UNIFIED RESEARCHER WORKFLOW")
            logger.info("="*100)
            logger.info(f"Project: {self.project_name}")
            logger.info(f"Topic: {self.research_topic or 'Not specified'}")
            logger.info("="*100)
            
            # Setup model manager
            if not self.setup_model_manager():
                return False
            
            # Create project structure
            if not self.create_project_structure():
                return False
            
            # Try to load existing workflow state
            self.load_workflow_state()
            
            # Execute phases based on current state
            phases = [
                ("environment_design", self.phase_environment_design),
                ("scenario_creation", self.phase_scenario_creation),
                ("experiment_execution", self.phase_experiment_execution),  
                ("analysis", self.phase_analysis),
                ("report_generation", self.phase_report_generation)
            ]
            
            for phase_name, phase_func in phases:
                if phase_name not in self.workflow_state["phases_completed"]:
                    logger.info(f"\nExecuting phase: {phase_name}")
                    
                    if not phase_func():
                        logger.error(f"Phase {phase_name} failed")
                        self.save_workflow_state()
                        return False
                    
                    # Save state after each phase
                    self.save_workflow_state()
                else:
                    logger.info(f"Phase {phase_name} already completed, skipping")
            
            # Final summary
            logger.info("\n" + "="*100)
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("="*100)
            logger.info(f"✓ Project created: {self.project_dir}")
            logger.info(f"✓ Phases completed: {', '.join(self.workflow_state['phases_completed'])}")
            
            if self.workflow_state.get("report_paths"):
                logger.info(f"✓ Reports generated: {len(self.workflow_state['report_paths'])} files")
            
            logger.info("="*100)
            
            return True
            
        except Exception as e:
            logger.error(f"Full workflow failed: {e}")
            logger.error(traceback.format_exc())
            self.save_workflow_state()
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OneSim 多范式研究工作流 - Complete research workflow automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 新范式方式：配置文件
  python src/researcher.py --config configs/threshold_validation.json
  
  # 新范式方式：命令行参数
  python src/researcher.py --project_name "rumor_study" --scenario "网络谣言传播现象" --question "什么机制决定传播差异？" --paradigm "mechanism_discovery"
  
  # 兼容旧版本：research topic方式
  python src/researcher.py --topic "social_network_dynamics" --project_name "social_dynamics_study"
  
  # 阶段性执行
  python src/researcher.py --project_name "existing_project" --phase execute
  
  # 混合方式：配置文件 + 参数覆盖
  python src/researcher.py --config configs/base.json --paradigm "boundary_exploration"
        """)
    
    # 新的核心参数
    parser.add_argument(
        "--scenario",
        type=str,
        help="场景描述（自然语言）"
    )
    parser.add_argument(
        "--theory",
        type=str,
        help="社会科学理论"
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="社会科学理论的条件"
    )
    parser.add_argument(
        "--observation",
        type=str,
        help="观察现象"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="研究问题（自然语言）"
    )
    
    parser.add_argument(
        "--paradigm",
        type=str,
        choices=["auto_agent", "theory_validation", "mechanism_discovery", 
                "boundary_exploration", "attribution_analysis"],
        default="auto_agent",
        help="研究范式选择"
    )
    
    # 配置文件支持
    parser.add_argument(
        "--config",
        type=str,
        default="config/research_config.json",
        help="研究配置文件路径（JSON格式）"
    )
    
    # 向后兼容参数
    parser.add_argument(
        "--topic",
        type=str,
        help="研究主题（兼容旧版本）"
    )
    
    parser.add_argument(
        "--project_name",
        type=str,
        help="项目名称（如果config文件中没有指定则为必需）"
    )
    
    # Configuration arguments  
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model configuration name to use"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--projects_dir",
        type=str,
        help="Base directory for projects (default: ../projects)"
    )
    
    # Phase control
    parser.add_argument(
        "--phase",
        type=str,
        choices=["design", "scenario", "execute", "analysis", "report", "full"],
        default="full",
        help="Specific phase to run: design (env_design), scenario (complete scenario creation after env_design), execute, analysis, report, full (default: full)"
    )
    
    # Skip options
    parser.add_argument(
        "--skip_design",
        action="store_true",
        help="Skip environment design phase"
    )
    
    parser.add_argument(
        "--skip_execution",
        action="store_true", 
        help="Skip experiment execution phase"
    )
    
    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Skip report generation phase"
    )
    
    return parser.parse_args()


def load_research_config(config_path: str) -> Dict:
    """
    加载研究配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置内容
        
    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: 配置文件格式错误
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        logger.info(f"✓ 配置文件加载成功: {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"配置文件JSON格式错误: {e}")
        raise
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise


def validate_workflow_params(params: Dict) -> Dict:
    """
    验证和规范化工作流参数
    
    Args:
        params: 原始参数字典
        
    Returns:
        Dict: 验证后的参数字典
    """
    validated_params = params.copy()
    
    # 确保project_name存在
    if not validated_params.get("project_name"):
        raise ValueError("project_name是必需的参数")
    
    # 验证research_paradigm
    paradigm = validated_params.get("research_paradigm", "auto_agent")
    try:
        ResearchParadigm(paradigm)
    except ValueError:
        logger.warning(f"无效的研究范式: {paradigm}, 使用默认值 'auto_agent'")
        validated_params["research_paradigm"] = "auto_agent"
    
    # 确保向后兼容性
    if validated_params.get("research_topic") and not validated_params.get("scenario_description"):
        validated_params["scenario_description"] = validated_params["research_topic"]
        logger.info("为兼容性设置scenario_description = research_topic")
    
    return validated_params


def main():
    """Main entry point for the unified researcher workflow."""
    try:
        args = parse_args()
        
        # 优先级：配置文件 > 命令行参数 > 默认值
        workflow_params = {}
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = Path("config/research_config.json")
        
        if config_path.exists():
            # 从配置文件加载
            try:
                config = load_research_config(args.config)
                
                # 提取配置文件中的各部分
                workflow_params.update({
                    "project_name": config.get("project_name"),
                    "scenario_description": config.get("scenario_description"),
                    "research_question": config.get("research_question"),
                    "research_paradigm": config.get("research_paradigm", "auto_agent")
                })
                
                # 添加模型配置
                model_config = config.get("model_config", {})
                workflow_params.update({
                    "model_name": model_config.get("model_name"),
                    "model_config_path": model_config.get("model_config_path")
                })
                
                # 添加实验配置
                experimental_config = config.get("experimental_config", {})
                workflow_params.update({
                    "projects_base_dir": experimental_config.get("projects_base_dir")
                })
                
                logger.info(f"使用配置文件: {args.config}")
                
            except Exception as e:
                logger.error(f"配置文件处理失败: {e}")
                return 1
        
        # 命令行参数覆盖配置文件（如果提供）
        if args.project_name:
            workflow_params["project_name"] = args.project_name
        if args.scenario:
            workflow_params["scenario_description"] = args.scenario
        if args.theory:
            workflow_params["theory"] = args.theory
        if args.observation:
            workflow_params["observation"] = args.observation
        if args.condition:
            workflow_params["condition"] = args.condition
        if args.question:
            workflow_params["research_question"] = args.question
        if args.paradigm != "auto_agent":  # 只有非默认值时才覆盖
            workflow_params["research_paradigm"] = args.paradigm
        if args.topic:
            workflow_params["research_topic"] = args.topic
        if args.model_name:
            workflow_params["model_name"] = args.model_name
        if args.model_config:
            workflow_params["model_config_path"] = args.model_config
        if args.projects_dir:
            workflow_params["projects_base_dir"] = args.projects_dir
        
        # 验证和规范化参数
        try:
            workflow_params = validate_workflow_params(workflow_params)
        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            return 1
        
        # 输入验证：确保有足够信息进行环境设计
        paradigm = workflow_params.get("research_paradigm", "auto_agent")
        scenario = workflow_params.get("scenario_description")
        topic = workflow_params.get("research_topic")
        
        if args.phase in ["design", "full"] and not args.skip_design:
            if not (scenario or topic):
                logger.error("环境设计阶段需要提供场景描述 (--scenario) 或研究主题 (--topic)")
                return 1
        
        # 为特定阶段提供信息
        if args.phase == "scenario":
            logger.info("场景创建阶段: 将从workflow_state.json读取场景信息")
        
        # 显示当前配置信息
        logger.info(f"项目名称: {workflow_params.get('project_name')}")
        logger.info(f"研究范式: {paradigm}")
        if scenario:
            logger.info(f"场景描述: {scenario[:100]}..." if len(scenario) > 100 else f"场景描述: {scenario}")
        if workflow_params.get("research_question"):
            question = workflow_params.get("research_question")
            logger.info(f"研究问题: {question[:100]}..." if len(question) > 100 else f"研究问题: {question}")
        
        # Initialize workflow coordinator
        workflow = ResearcherWorkflow(**workflow_params)
        
        # Execute requested phase(s)
        success = False
        
        if args.phase == "full":
            success = workflow.run_full_workflow()
        elif args.phase == "design":
            if workflow.setup_model_manager() and workflow.create_project_structure():
                success = workflow.phase_environment_design()
        elif args.phase == "scenario":
            if workflow.setup_model_manager():
                success = workflow.phase_scenario_creation()
        elif args.phase == "execute":
            if workflow.setup_model_manager():
                workflow.load_workflow_state()
                success = workflow.phase_experiment_execution()
        elif args.phase == "analysis":
            if workflow.setup_model_manager():
                workflow.load_workflow_state()
                success = workflow.phase_analysis()
        elif args.phase == "report":
            if workflow.setup_model_manager():
                workflow.load_workflow_state()
                success = workflow.phase_report_generation()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

