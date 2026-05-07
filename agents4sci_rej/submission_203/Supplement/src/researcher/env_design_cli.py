#!/usr/bin/env python3
"""
Main entry point for the experimental design workflow.

This module provides a command line interface for running the
experimental design workflow, which generates a detailed simulation
specification from a vague research topic.
"""

import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # Assuming this script is in src/researcher/

from onesim.models import get_model_manager
from loguru import logger

# Import agents - adjust path as needed if agents are in a different location relative to this script
    
# from researcher.env_design import InspirationAgent 
# from researcher.env_design import EvaluatorAgent   
# from researcher.env_design import DetailerAgent    
# from researcher.env_design import AssessorAgent    
# from researcher.env_design import Experimentagent    
# from researcher.env_design import InterventionAgent
# from researcher.env_design import ScenarioAgent

from researcher.env_design.paradigm_1 import Paradigm1Agent
from researcher.env_design.paradigm_2 import Paradigm2Agent
from researcher.env_design.paradigm_3 import Paradigm3Agent
from researcher.env_design.paradigm_4 import Paradigm4Agent
from researcher.env_design.parameter_generator import ParameterGenerator


class Coordinator:
    """
    Coordinator for the experimental design workflow.
    
    The coordinator manages the workflow between the three main agents
    (InspirationAgent, EvaluatorAgent, and DetailerAgent) to generate
    a detailed simulation specification from a vague research topic.
    """
    
    def __init__(
        self,
        scene_name: Optional[str] = None,
        model_name: Optional[str] = None,
        research_paradigm: Optional[str] = None,
        save_intermediate: bool = True
    ):
        """
        Initialize the coordinator.
        
        Args:
            scene_name (str, optional): The name of the scene to create.
            model_name (str, optional): The model configuration name to use.
            save_intermediate (bool, optional): Whether to save intermediate outputs.
        """
        self.scene_name = scene_name
        self.model_name = model_name
        self.save_intermediate = save_intermediate
        self.research_paradigm = research_paradigm

        self.research_paradigm1_agent = Paradigm1Agent(model_name)
        self.research_paradigm2_agent = Paradigm2Agent(model_name)
        self.research_paradigm3_agent = Paradigm3Agent(model_name)
        self.research_paradigm4_agent = Paradigm4Agent(model_name)
        self.parameter_generator = ParameterGenerator(model_name)
    
    def setup_environment(self, scene_name: str) -> str:
        """
        Set up environment directory structure relative to the 'src/envs' directory.
        
        Args:
            scene_name (str): The name of the scene.
            
        Returns:
            str: The path to the created environment root under 'src/envs/'.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        env_base_path = project_root / "src" / "envs"
        
        env_path = env_base_path / scene_name
        research_path = env_path / "research" / "env_design"
        
        # Create directories
        env_path.mkdir(parents=True, exist_ok=True)
        research_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize package
        init_file = env_path / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write("")
        
        # Ensure coordinator tracks the active scene name
        self.scene_name = scene_name
                
        return str(env_path)
    
    def run(self, input_data) -> Dict[str, Any]:
        """
        Run the experimental design workflow.

        Args:
            input_data: Dictionary containing research parameters and context
                - scenario_description: Description of the research scenario
                - research_question: The research question
                - research_topic: Research topic (for backward compatibility)
                - theory: Theory (optional, will be generated if missing)
                - observation: Observation (optional, will be generated if missing)
                - condition: Condition (optional, will be generated if missing)

        Returns:
            Dict[str, Any]: The final detailed simulation specification.
        """

        logger.info(f"Running experimental design workflow for paradigm: {self.research_paradigm}")

        # Generate missing parameters based on research paradigm requirements
        logger.info("Generating missing parameters...")
        generated_params = self.parameter_generator.generate_missing_parameters(
            research_paradigm=self.research_paradigm,
            scenario_description=input_data.get('scenario_description') or input_data.get('research_topic'),
            research_question=input_data.get('research_question'),
            theory=input_data.get('theory'),
            observation=input_data.get('observation'),
            condition=input_data.get('condition')
        )

        # Log the parameters that were generated vs provided
        param_status = []
        for param_name in ['theory', 'observation', 'condition']:
            if param_name in generated_params:
                was_provided = input_data.get(param_name) is not None
                param_status.append(f"{param_name}: {'provided' if was_provided else 'generated'}")
        logger.info(f"Parameter status: {', '.join(param_status)}")

        # Determine project root and base path for envs
        project_root = Path(__file__).resolve().parent.parent.parent
        envs_base_path = project_root / "src" / "envs"

        # Decide scene name: use provided one if available; otherwise generate a temporary one
        selected_scene_name = self.scene_name or f"temp_outputs_{random.randint(100000, 999999)}"
        # Create a temporary directory for intermediate outputs within the project structure (e.g., in a .tmp folder or similar)
        # For simplicity here, placing it where it was, but ideally in a gitignored .tmp folder at project root.
        temp_dir = envs_base_path / selected_scene_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        env_path_str = self.setup_environment(selected_scene_name) # Returns root of env: src/envs/scene_name
        env_path = Path(env_path_str)
        research_path = env_path / "research" / "env_design" # This is src/envs/scene_name/research/env_design

        if self.research_paradigm == "theory_validation":
            logger.info("Using Research Paradigm 1: Theory Validation (T+C → O)")
            paradigm_input = {
                "theory": generated_params.get('theory'),
                "condition": generated_params.get('condition')
            }
            inspiration_output, detailer_output = self.research_paradigm1_agent.run(paradigm_input)
            self._save_json(
                    inspiration_output,
                    str(research_path / "inspiration_output.json")
                )
            logger.info(f"✓ Saved inspiration output")
            self._save_json(
                    detailer_output.get("odd_protocol", {}),
                    str(research_path / "odd_protocol.json")
                )

            # Update scene name from detailer output if available and rename directory
            self._maybe_update_scene_name(envs_base_path, selected_scene_name, detailer_output)

            return {
                "inspiration_output": inspiration_output,
                "detailer_output": detailer_output
            }
        if self.research_paradigm == "attribution_analysis":
            logger.info("Using Research Paradigm 3: Attribution Analysis (T+O → C_weight)")
            paradigm_input = {
                "theory": generated_params.get('theory'),
                "observation": generated_params.get('observation'),
                "condition": generated_params.get('condition'),  # May be None, that's ok
                "scene_name": self.scene_name
            }
            inspiration_output, detailer_output, experiment_output, intervention_outupt = self.research_paradigm3_agent.run(paradigm_input)

            self._save_json(
                    inspiration_output,
                    str(research_path / "inspiration_output.json")
                )
            logger.info(f"✓ Saved inspiration output")
            self._save_json(
                    detailer_output.get("odd_protocol", {}),
                    str(research_path / "odd_protocol.json")
                )
            logger.info(f"✓ Saved ODD protocol")
            self._save_json(
                    experiment_output,
                    str(research_path / "experiment_config.json")
                )
            logger.info(f"✓ Saved experiment config")
            self._save_json(
                    intervention_outupt,
                    str(research_path / "intervention_specifications.json")
                )
            logger.info(f"✓ Saved intervention specifications")

            # Update scene name from detailer output if available and rename directory
            self._maybe_update_scene_name(envs_base_path, selected_scene_name, detailer_output)

            return {
                "inspiration_output": inspiration_output,
                "detailer_output": detailer_output,
                "experiment_output": experiment_output,
                "intervention_outupt": intervention_outupt
            }
        if self.research_paradigm == "boundary_exploration":
            logger.info("Using Research Paradigm 4: Boundary Exploration (T+O → C_boundary)")
            paradigm_input = {
                "theory": generated_params.get('theory'),
                "observation": generated_params.get('observation'),
                "condition": generated_params.get('condition'),  # May be None, that's ok
                "scene_name": self.scene_name
            }
            inspiration_output, detailer_output, experiment_output, intervention_outupt = self.research_paradigm4_agent.run(paradigm_input)

            self._save_json(
                    inspiration_output,
                    str(research_path / "inspiration_output.json")
                )
            logger.info(f"✓ Saved inspiration output")
            self._save_json(
                    detailer_output.get("odd_protocol", {}),
                    str(research_path / "odd_protocol.json")
                )
            logger.info(f"✓ Saved ODD protocol")
            self._save_json(
                    experiment_output,
                    str(research_path / "experiment_config.json")
                )
            logger.info(f"✓ Saved experiment config")
            self._save_json(
                    intervention_outupt,
                    str(research_path / "intervention_specifications.json")
                )
            logger.info(f"✓ Saved intervention specifications")

            # Update scene name from detailer output if available and rename directory
            self._maybe_update_scene_name(envs_base_path, selected_scene_name, detailer_output)

            return {
                "inspiration_output": inspiration_output,
                "detailer_output": detailer_output,
                "experiment_output": experiment_output,
                "intervention_outupt": intervention_outupt
            }
        if self.research_paradigm == "mechanism_discovery":
            logger.info("Using Research Paradigm 2: Mechanism Discovery (O+C → T)")
            paradigm_input = {
                "observation": generated_params.get('observation'),
                "condition": generated_params.get('condition'),
                "scene_name": self.scene_name
            }
            inspiration_output, detailer_output = self.research_paradigm2_agent.run(paradigm_input)

            self._save_json(
                    inspiration_output,
                    str(research_path / "inspiration_output.json")
                )
            logger.info(f"✓ Saved inspiration output")
            self._save_json(
                    detailer_output.get("odd_protocol", {}),
                    str(research_path / "odd_protocol.json")
                )
            logger.info(f"✓ Saved ODD protocol")

            # Update scene name from detailer output if available and rename directory
            self._maybe_update_scene_name(envs_base_path, selected_scene_name, detailer_output)

            return {
                "inspiration_output": inspiration_output,
                "detailer_output": detailer_output
            }

        # Default case for unknown paradigms
        logger.warning(f"Unknown research paradigm: {self.research_paradigm}, using auto_agent fallback")
        paradigm_input = {
            "theory": generated_params.get('theory'),
            "observation": generated_params.get('observation'),
            "condition": generated_params.get('condition')
        }
        return {"generated_params": generated_params, "paradigm_input": paradigm_input}
    
    def assess(self, scene_name: str) -> Dict[str, Any]:
        """
        Assess the quality of the ODD protocol conversion based on four metrics.
        
        Args:
            scene_name (str): The name of the scene to assess.
            
        Returns:
            Dict[str, Any]: The assessment results.
        """
        logger.info(f"{'='*80}")
        logger.info(f"Starting assessment process for scene: {scene_name}")
        logger.info(f"{'='*80}")
        
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / "src" / "envs" / scene_name
        research_path = env_path / "research" / "env_design"
        
        if not env_path.exists():
            logger.error(f"Scene '{scene_name}' does not exist at path: {env_path}")
            return {"error": f"Scene '{scene_name}' does not exist"}
        
        odd_protocol_path = research_path / "odd_protocol.json"
        summary_path = research_path / "final_output_summary.json"
        
        if not odd_protocol_path.exists():
            logger.error(f"ODD protocol not found at: {odd_protocol_path}")
            return {"error": "ODD protocol not found"}
        
        with open(odd_protocol_path, 'r', encoding='utf-8') as f:
            odd_protocol = json.load(f)
        
        original_topic = ""
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                original_topic = summary.get("original_topic", "")
        
        assessment_input = {
            "original_topic": original_topic,
            "odd_protocol": odd_protocol,
            "scene_name": scene_name
        }
        
        logger.info("\nAssessing conversion quality based on four metrics...")
        assessment_output = self.assessor_agent.process(assessment_input)
        
        assessment_results_path = research_path / "assessment_results.json"
        self._save_json(assessment_output, str(assessment_results_path))
        logger.info(f"✓ Saved assessment results to {assessment_results_path.relative_to(project_root)}")
        
        logger.info("\nAssessment Results:")
        logger.info(f"Relevance:   {assessment_output.get('relevance', {}).get('score', 0)}/5 - {assessment_output.get('relevance', {}).get('summary', '')}")
        logger.info(f"Fidelity:    {assessment_output.get('fidelity', {}).get('score', 0)}/5 - {assessment_output.get('fidelity', {}).get('summary', '')}")
        logger.info(f"Feasibility: {assessment_output.get('feasibility', {}).get('score', 0)}/5 - {assessment_output.get('feasibility', {}).get('summary', '')}")
        logger.info(f"Significance: {assessment_output.get('significance', {}).get('score', 0)}/5 - {assessment_output.get('significance', {}).get('summary', '')}")
        logger.info(f"\nOverall Score: {assessment_output.get('overall_score', 0)}/20")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Assessment process completed successfully!")
        logger.info(f"{'='*80}")
        
        return assessment_output
    
    def _save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data (Dict[str, Any]): The data to save.
            file_path (str): The path to save the data to.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _maybe_update_scene_name(self, envs_base_path: Path, temp_scene_name: str, detailer_output: Dict[str, Any]) -> None:
        """
        Update self.scene_name from detailer_output if present, and rename directory from temp to final.
        """
        try:
            final_scene_name = None
            if isinstance(detailer_output, dict):
                # Common locations: direct key or nested under detail fields
                final_scene_name = detailer_output.get("scene_name") or \
                                   detailer_output.get("odd_protocol", {}).get("scene_name")

            if final_scene_name and final_scene_name != temp_scene_name:
                temp_path = envs_base_path / temp_scene_name
                final_path = envs_base_path / final_scene_name
                # Avoid collision
                if final_path.exists():
                    # If already exists, just set scene_name and do not rename
                    self.scene_name = final_scene_name
                    return
                # Rename directory tree
                temp_path.rename(final_path)
                self.scene_name = final_scene_name
        except Exception as e:
            logger.warning(f"Failed to update scene name from detailer output: {e}")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a detailed simulation specification from a vague research topic."
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="The research topic (for backward compatibility)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Description of the research scenario"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="The research question"
    )
    parser.add_argument(
        "--theory",
        type=str,
        help="Social science theory (optional, will be auto-generated if missing)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Social science theory condition (optional, will be auto-generated if missing)"
    )
    parser.add_argument(
        "--observation",
        type=str,
        help="Observation phenomenon (optional, will be auto-generated if missing)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        choices=["theory_validation", "mechanism_discovery", "attribution_analysis", "boundary_exploration", "auto_agent"],
        default="auto_agent",
        help="Research paradigm to use"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Name for the environment scene directory."
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model configuration name (config_name from JSON) to use."
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.json", # Default relative to project root
        help="Path to the model configuration JSON file (relative to project root)."
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save intermediate outputs."
    )
    
    parser.add_argument(
        "--assess",
        action="store_true",
        help="Run assessment on an existing scene."
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the experimental design workflow.
    """
    args = parse_args()
    
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        model_config_abs_path = project_root / args.model_config

        if not model_config_abs_path.exists():
            logger.error(f"Model configuration file not found at: {model_config_abs_path}")
            logger.error(f"Please ensure the path specified by --model_config is correct or the default '{args.model_config}' exists relative to the project root.")
            return 1
            
        model_manager = get_model_manager()
        model_manager.load_model_configs(str(model_config_abs_path))
        logger.info(f"Loaded model configurations from: {model_config_abs_path}")
        
        coordinator = Coordinator(
            scene_name=args.scene_name,
            model_name=args.model_name,
            save_intermediate=args.save,
            research_paradigm=args.paradigm
        )
        
        # if args.assess and not args.topic: # If only assess is true
        #      if not args.scene_name:
        #         logger.error("Scene name is required for assessment mode when no topic is provided.")
        #         # args.print_help()
        #         return 1
        #      coordinator.assess(args.scene_name)
        #      return 0

        # Build input data from all available sources
        input_data = {
            'scenario_description': getattr(args, 'scenario', None),
            'research_question': getattr(args, 'question', None),
            'research_topic': args.topic if hasattr(args, 'topic') and args.topic else None,
            'theory': args.theory if args.theory else None,
            'condition': args.condition if args.condition else None,
            'observation': args.observation if args.observation else None
        }

        # Check if we have at least some input to work with
        has_context = any([
            input_data.get('scenario_description'),
            input_data.get('research_question'),
            input_data.get('research_topic')
        ])

        if not has_context:
            logger.error("At least one of --scenario, --question, or --topic must be provided")
            return 1
        coordinator.run(input_data)
        
        if args.assess: # If assess is true along with a topic (assess after run)
            # coordinator.scene_name would have been set during the run
            if coordinator.scene_name:
                 coordinator.assess(coordinator.scene_name)
            else:
                logger.warning("Could not run assessment as scene_name was not determined during the run.")
        
        return 0
    except Exception as e:
        logger.error(f"\nAn error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) # Ensure exit code is propagated