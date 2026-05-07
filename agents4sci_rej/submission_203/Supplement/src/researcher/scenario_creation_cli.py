#!/usr/bin/env python3
"""
Complete Scenario Creation Workflow for OneSim Researcher

This script provides a comprehensive workflow for creating complete simulation scenarios,
including agent types, workflows, code generation, profiles, and relationships.
Based on the vis_initialization.py implementation but adapted for the researcher module.

Usage:
    python src/researcher/scenario_creation.py --topic "research topic" --scene-name "scenario_name"
    python src/researcher/scenario_creation.py --config config/scenario_config.json
"""

import json
import sys
import argparse
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import traceback

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import onesim
from onesim.agent import ProfileAgent, WorkflowAgent, CodeAgent, MetricAgent
from onesim.models import ModelManager, get_token_usage_stats, log_token_usage, estimate_token_cost
from onesim.profile import AgentSchema
from onesim.simulator import AgentFactory
# Import debugging functionality - try multiple approaches
try:
    from onesim.agent.debugging_agent import DebuggingAgent, DebuggingConfig
    DEBUGGING_AGENT_AVAILABLE = True
except ImportError:
    DEBUGGING_AGENT_AVAILABLE = False
    DebuggingAgent = None
    DebuggingConfig = None


class ScenarioCreationPipeline:
    """
    Complete scenario creation pipeline that orchestrates all steps from
    research topic to fully functional simulation environment.
    """
    
    def __init__(
        self,
        scene_name: str,
        research_topic: str,
        model_config_path: str = "config/model_config.json",
        selected_model: str = None,
        base_path: str = None,
        project_name: str = None,
        save_intermediate: bool = True
    ):
        """
        Initialize the scenario creation pipeline.
        
        Args:
            scene_name: Name of the scenario to create
            research_topic: Research topic/description for scenario generation
            model_config_path: Path to model configuration file
            selected_model: Model configuration name to use
            base_path: Base directory path (defaults to project root)
            save_intermediate: Whether to save intermediate results
        """
        self.scene_name = scene_name
        self.research_topic = research_topic
        self.model_config_path = model_config_path
        self.selected_model = selected_model
        self.save_intermediate = save_intermediate
        self.project_name = project_name
        
        # Setup paths
        if base_path:
            self.base_path = Path(base_path).resolve()
        else:
            self.base_path = Path(__file__).resolve().parent.parent.parent
        
        self.envs_dir = self.base_path / "src" / "envs"
        self.env_path = self.envs_dir / scene_name
        self.project_path = self.base_path / "projects" / self.project_name
        
        # Pipeline state
        self.state = {
            "base_path": str(self.base_path),
            "scene_name": scene_name,
            "scene_path": str(self.env_path),
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "selected_model": selected_model,
            "research_topic": research_topic,
            "config": {},
            "agent_types": [],
            "actions": {},
            "events": {},
            "system_data_model": {},
            "relationship_schema": {},
            "all_profiles": {},
            "all_agent_ids": {},
            "index": 0,
            "statistics": {
                "start_time": time.time(),
                "code_verification_issues": {},
                "workflow_validation_issues": [],
                "token_usage": 0,
                "files_generated": {"code": 0, "profile": 0, "other": 0},
                "total_lines_of_code": 0
            }
        }
        with open(self.env_path/'scene_info.json','r',encoding='utf-8') as f:
            self.state["odd_protocol"] = json.load(f).get("odd_protocol",{})
        # Initialize agents
        self.model = None
        self.profile_agent = None
        self.workflow_agent = None
        self.code_agent = None
        self.metric_agent = None
    
    def initialize_llm(self):
        """Initialize the LLM with model configuration."""
        asyncio.run(onesim.init(model_config_path=self.model_config_path))
        
        model_manager = ModelManager.get_instance()
        self.model = model_manager.get_model(config_name=self.selected_model)
        
        self.profile_agent = ProfileAgent(model_config_name=self.selected_model)
        self.workflow_agent = WorkflowAgent(model_config_name=self.selected_model)
        self.code_agent = CodeAgent(model_config_name=self.selected_model)
        self.metric_agent = MetricAgent(model_config_name=self.selected_model)
        
        logger.info(f"✓ LLM initialized with model: {self.selected_model}")
    
    def setup_logging(self):
        """Set up logging with filename based on environment name and creation time."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_dir = self.env_path / "log" / "create" / "create_scene"
        self.create_directory(log_dir)
        log_file = log_dir / f"{self.scene_name}_{timestamp}.log"
        logger.add(str(log_file), rotation="500 MB", compression="zip", level="INFO")
        logger.info(f"✓ Logging setup complete: {log_file}")
        return str(log_file)
    
    def create_directory(self, path: Path):
        """Create a directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
    
    def init_package(self, path: Path):
        """Create __init__.py in a package directory."""
        init_file = path / '__init__.py'
        init_file.touch()
    
    def load_scenario_config(self) -> Dict[str, Any]:
        """Load the scenario_config.json file or create it if it doesn't exist."""
        scenario_config_path = self.project_path/"base_scenario" / "scenario_config.json"
        if scenario_config_path.exists():
            with open(scenario_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def load_scene_info(self) -> Dict[str, Any]:
        """Load the scene_info.json file or create it if it doesn't exist."""
        scene_info_path = self.env_path / "scene_info.json"
        if scene_info_path.exists():
            try:
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading scene_info.json: {e}")
        
        # Create a new empty scene_info if file doesn't exist
        scene_info = {
            "domain": "",
            "scene_name": self.scene_name,
            "odd_protocol": {
                "overview": {
                    "system_goal": "",
                    "agent_types": "",
                    "environment_description": ""
                },
                "design_concepts": {
                    "interaction_patterns": "",
                    "communication_protocols": "",
                    "decision_mechanisms": ""
                },
                "details": {
                    "agent_behaviors": "",
                    "decision_algorithms": "",
                    "specific_constraints": ""
                }
            },
            "metrics": []
        }
        
        # Save the default scene_info
        try:
            with open(scene_info_path, 'w', encoding='utf-8') as f:
                json.dump(scene_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error creating default scene_info.json: {e}")
        
        return scene_info
    
    def save_scene_info(self, scene_info: Dict[str, Any], 
                       agent_types_with_descriptions: Dict = None, 
                       portrait_assignments: Dict = None):
        """Save the scene_info to JSON file."""
        scene_info_path = self.env_path / "scene_info.json"
        
        # Add agent types and descriptions if provided
        if agent_types_with_descriptions:
            if "agent_types" not in scene_info:
                scene_info["agent_types"] = {}
            scene_info["agent_types"] = agent_types_with_descriptions
        
        # Add portrait assignments if provided
        if portrait_assignments:
            if "portrait" not in scene_info:
                scene_info["portrait"] = {}
            scene_info["portrait"] = portrait_assignments
        
        try:
            with open(scene_info_path, 'w', encoding='utf-8') as f:
                json.dump(scene_info, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Scene info saved to {scene_info_path}")
        except Exception as e:
            logger.error(f"Error saving scene_info.json: {e}")
    
    def generate_agent_types(self) -> bool:
        """Generate agent types based on research topic."""
        try:
            logger.info("Generating agent types...")
            
            if self.state["scenario_config"].get("agent_config"):
                agent_types = self.state["scenario_config"].get("agent_config", {}).get("types", {})
                descriptions = f"Agent Types:\n{json.dumps(agent_types, indent=2)}\n\n{self.state['odd_protocol']}"
            else:
                descriptions = self.state["odd_protocol"]
            agent_types_with_descriptions = self.profile_agent.generate_agent_types(descriptions)
            portrait_assignments = self.profile_agent.assign_agent_portraits(agent_types_with_descriptions)
            
            agent_types = list(agent_types_with_descriptions.keys())
            self.state["agent_types"] = agent_types
            self.state["agent_types_with_descriptions"] = agent_types_with_descriptions
            
            # Save agent types to scene_info.json
            scene_info = self.load_scene_info()
            self.save_scene_info(scene_info, agent_types_with_descriptions, portrait_assignments)
            
            logger.info(f"✓ Generated agent types: {agent_types}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate agent types: {e}")
            return False
    
    def generate_workflow(self) -> bool:
        """Generate actions and events for a workflow."""
        try:
            if not self.state["agent_types"]:
                logger.error("Agent types must be generated first")
                return False
            
            logger.info("Generating workflow...")
            workflow_start = time.time()
            
            actions, events, system_data, G = self.workflow_agent.generate_workflow(
                self.state['odd_protocol'], self.state["agent_types"]
            )
            
            # Visualize the graph
            self.workflow_agent.visualize_interactive_graph(
                G, str(self.env_path / "workflow.html")
            )
            
            # Save workflow data
            with open(self.env_path / "actions.json", "w") as f:
                json.dump(actions, f, ensure_ascii=False, indent=4)
            
            with open(self.env_path / "events.json", "w") as f:
                json.dump(events, f, ensure_ascii=False, indent=4)
            
            with open(self.env_path / "system_data_model.json", "w") as f:
                json.dump(system_data, f, ensure_ascii=False, indent=4)
            
            # Update state
            self.state["actions"] = actions
            self.state["events"] = events
            self.state["system_data_model"] = system_data
            self.state["statistics"]["workflow_generation_time"] = time.time() - workflow_start
            
            logger.info("✓ Workflow generated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to generate workflow: {e}")
            return False
    
    def generate_metrics(self) -> bool:
        """Generate monitoring metrics."""
        try:
            if not self.state["agent_types"] or not self.state["system_data_model"]:
                logger.error("Agent types and system data model must be generated first")
                return False
            
            logger.info("Generating metrics...")
            description = self.state["odd_protocol"]
            if self.state["scenario_config"]:
                metric_config = self.state["scenario_config"].get("metric_config", {})
                description += f"\nMetrics:\n{json.dumps(metric_config, indent=2)}"
            metrics = self.metric_agent.generate_metrics(
                description, 
                self.state["agent_types"], 
                self.state["system_data_model"]
            )
            
            if not metrics:
                logger.warning("No metrics generated")
                return True
            
            logger.info(f"Generated {len(metrics)} monitoring metrics")
            
            # Generate metrics calculation code
            code_path = self.env_path / "code" / "metrics"
            self.create_directory(code_path)
            self.metric_agent.generate_metrics_module(
                metrics, str(code_path), self.state["system_data_model"]
            )
            
            # Update scene_info.json
            scene_info = self.load_scene_info()
            formatted_metrics = self.metric_agent.format_metrics_for_export(metrics)
            scene_info["metrics"] = formatted_metrics
            self.save_scene_info(scene_info)
            
            logger.info("✓ Metrics generated and saved")
            return True
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return False
    
    def generate_code(self) -> bool:
        """Generate code based on actions and events."""
        try:
            if not self.state["actions"] or not self.state["events"]:
                logger.error("Actions and events must be generated first")
                return False
            
            logger.info("Generating code...")
            code_start = time.time()
            
            code_path = self.env_path / "code"
            self.create_directory(code_path)
            self.init_package(code_path)
            
            self.code_agent.generate_code(
                self.state["odd_protocol"],
                self.state["actions"],
                self.state["events"],
                str(code_path)
            )
            
            # Update statistics
            self.state["statistics"]["code_generation_time"] = time.time() - code_start
            self.state["statistics"]["total_lines_of_code"] = self.count_lines_of_code(code_path)
            self.state["statistics"]["files_generated"] = self.count_files_by_type(self.env_path)
            
            logger.info("✓ Code generated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return False
    
    def save_profile_schemas(self) -> bool:
        """Generate profile schemas for each agent type."""
        try:
            if not self.state["system_data_model"]:
                logger.error("System data model must be generated first")
                return False
            
            logger.info("Saving profile schemas...")
            profile_data_path = self.env_path / "profile" / "schema"
            self.create_directory(profile_data_path)
            
            agent_data_model = self.state["system_data_model"].get('agents', {})
            for agent_type, data_model in agent_data_model.items():
                schema = self.profile_agent.generate_profile_schema(
                    self.state["odd_protocol"], agent_type, data_model
                )
                with open(profile_data_path / f"{agent_type}.json", "w") as f:
                    json.dump(schema, f, ensure_ascii=False, indent=4)
            
            logger.info("✓ Profile schemas saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile schemas: {e}")
            return False
    
    def generate_relationship_schema(self) -> bool:
        """Generate relationship schemas."""
        try:
            if not self.state["agent_types"] or not self.state["actions"] or not self.state["events"]:
                logger.error("Agent types, actions, and events must be generated first")
                return False
            
            logger.info("Generating relationship schema...")
            profile_data_path = self.env_path / "profile" / "schema"
            self.create_directory(profile_data_path)
            
            schema = self.profile_agent.generate_relationship_schema(
                self.state["agent_types"],
                self.state["actions"],
                self.state["events"]
            )
            
            with open(profile_data_path / "Relationship.json", "w") as f:
                json.dump(schema, f, ensure_ascii=False, indent=4)
            
            self.state["relationship_schema"] = schema
            logger.info("✓ Relationship schema generated and saved")
            return True
        except Exception as e:
            logger.error(f"Failed to generate relationship schema: {e}")
            return False
    
    def generate_topology_config(self) -> bool:
        """Generate topology configuration for relationship networks."""
        try:
            if not self.state["agent_types"] or not self.state["relationship_schema"]:
                logger.error("Agent types and relationship schema must be generated first")
                return False
            
            logger.info("Generating topology configuration...")
            profile_data_path = self.env_path / "profile" / "schema"
            self.create_directory(profile_data_path)
            
            topology_config = self.profile_agent.generate_topology_config(
                self.state["odd_protocol"],
                self.state["agent_types"],
                self.state["relationship_schema"]
            )
            
            with open(profile_data_path / "RelationshipTopology.json", "w") as f:
                json.dump(topology_config, f, ensure_ascii=False, indent=4)
            
            self.state["topology_config"] = topology_config
            logger.info("✓ Topology configuration generated and saved")
            return True
        except Exception as e:
            logger.error(f"Failed to generate topology configuration: {e}")
            return False
    
    def generate_env_data(self) -> bool:
        """Generate environment data."""
        try:
            if not self.state["system_data_model"]:
                logger.error("System data model must be generated first")
                return False
            
            logger.info("Generating environment data...")
            env_data_model = self.state["system_data_model"].get('environment', {})
            if self.state["scenario_config"]:
                env_config = self.state["scenario_config"].get("environment_config", {})
                if env_config and "data" in env_config and "variables" in env_data_model:
                    env_data_model["variables"].update(env_config["data"])
            env_data = self.profile_agent.generate_env_data(env_data_model, self.state['odd_protocol'])
           
            with open(self.env_path / "env_data.json", "w") as f:
                json.dump(env_data, f, ensure_ascii=False, indent=4)
            
            self.state["env_data"] = env_data
            logger.info("✓ Environment data generated and saved")
            return True
        except Exception as e:
            logger.error(f"Failed to generate environment data: {e}")
            return False
    
    def generate_profiles(self, num_profiles: int = 5) -> bool:
        """Generate profiles for all agent types."""
        try:
            if not self.state["agent_types"]:
                logger.error("Agent types must be generated first")
                return False
            
            logger.info("Generating profiles...")
            index = self.state.get("index", 0)
            all_profiles = {}
            all_agent_ids = {}
            scenario_agent_types = self.state["scenario_config"].get("agent_config", {}).get("types", {})
            
            for agent_type in self.state["agent_types"]:
                agent_config = scenario_agent_types.get(agent_type, {})
                if "count" in agent_config and agent_config["count"] > 0:
                    num_profiles = agent_config["count"]
                profiles = self.generate_profiles_for_agent_type(
                    agent_type, self.model, index, num_profiles
                )
                index += len(profiles)
                all_profiles[agent_type] = profiles
                all_agent_ids[agent_type] = [profile.get_agent_profile_id() for profile in profiles]
            
            self.state["all_profiles"] = all_profiles
            self.state["all_agent_ids"] = all_agent_ids
            self.state["index"] = index
            
            logger.info("✓ Profiles generated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to generate profiles: {e}")
            return False
    
    def generate_profiles_for_agent_type(self, agent_type: str, model, index: int = 0, num_profiles: int = 5):
        """Generate profiles for a specific agent type."""
        config_path = self.env_path / "profile" / "schema" / f"{agent_type}.json"
        output_path = self.env_path / "profile" / "data" / f"{agent_type}.json"
        self.create_directory(output_path.parent)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        profiles = AgentFactory.generate_profiles(
            agent_type=agent_type,
            schema=AgentSchema(config_data),
            model=model,
            num_profiles=num_profiles,
            output_path=str(output_path),
            index=index
        )
        return profiles
    
    def generate_relationships(self) -> bool:
        """Generate relationships between agents using topology generation."""
        try:
            if not self.state["relationship_schema"] and not self.state["all_agent_ids"]:
                logger.error("Relationship schema and agent IDs must be generated first")
                return False
            
            logger.info("Generating relationships...")
            from onesim.utils.relationship_utils import save_relationships
            
            # Load relationship schema if not in state
            if not self.state["relationship_schema"]:
                relationship_path = self.env_path / "profile" / "schema" / "Relationship.json"
                with open(relationship_path) as f:
                    self.state["relationship_schema"] = json.load(f)
            
            # Load actions and events if not in state
            if not self.state["actions"] or not self.state["events"]:
                with open(self.env_path / "actions.json", "r") as f:
                    self.state["actions"] = json.load(f)
                with open(self.env_path / "events.json", "r") as f:
                    self.state["events"] = json.load(f)
            
            # Generate or load topology configuration
            topology_config = None
            topology_path = self.env_path / "profile" / "schema" / "RelationshipTopology.json"
            if topology_path.exists():
                with open(topology_path, 'r') as f:
                    topology_config = json.load(f)
            else:
                # Generate topology configuration
                topology_config = self.profile_agent.generate_topology_config(
                    self.state["odd_protocol"], 
                    self.state["agent_types"], 
                    self.state["relationship_schema"]
                )
                with open(topology_path, 'w') as f:
                    json.dump(topology_config, f, ensure_ascii=False, indent=4)
            
            # Generate relationships using the new topology generation
            output_csv_path = self.env_path / "profile" / "data" / "Relationship.csv"
            self.create_directory(output_csv_path.parent)
            
            min_relationships = self.state["config"].get("min_relationships_per_agent", 1)
            max_relationships = self.state["config"].get("max_relationships_per_agent", 3)
            
            # Use ProfileAgent's enhanced relationship generation with topology support
            relationships_list = self.profile_agent.generate_relationships(
                agent_ids=self.state["all_agent_ids"],
                relationship_schema=self.state["relationship_schema"],
                topology_config=topology_config,
                actions=self.state["actions"],
                events=self.state["events"],
                min_relationships=min_relationships,
                max_relationships=max_relationships
            )
            
            # Validate the generated relationships
            validation_results = self.profile_agent.validate_relationship_generation(
                relationship_schema=self.state["relationship_schema"],
                topology_config=topology_config,
                relationships=relationships_list,
                agent_ids=self.state["all_agent_ids"],
                actions=self.state["actions"],
                events=self.state["events"]
            )
            
            # Log validation results
            if validation_results['success']:
                logger.info("✅ Relationship generation validation passed")
                if validation_results['warnings']:
                    logger.warning(f"⚠️ {len(validation_results['warnings'])} warnings: {validation_results['warnings'][:3]}")
            else:
                logger.error(f"❌ Relationship generation validation failed: {validation_results['errors'][:3]}")
            
            # Log statistics
            stats = validation_results['statistics']
            logger.info(f"📊 Generated {stats['total_relationships']} relationships for {stats['total_agents']} agents using {stats['topology_used']} topology")
            logger.info(f"📊 Connectivity ratio: {stats['connectivity_ratio']:.2%}")
            
            save_relationships(relationships_list, str(output_csv_path))
            logger.info("✓ Relationships generated and saved")
            return True
        except Exception as e:
            logger.error(f"Failed to generate relationships: {e}")
            return False
    
    def debug_environment(self) -> bool:
        """Debug the environment using available debugging methods."""
        try:
            if not self.model:
                logger.error("Model must be initialized first")
                return False
            
            logger.info(f"Starting to debug environment '{self.scene_name}'...")
            
            if DEBUGGING_AGENT_AVAILABLE:
                logger.info("Using DebuggingAgent as debugging method...")
                try:
                    config = DebuggingConfig(
                        model_config_name=self.model.config_name,
                        max_iterations=10,
                        timeout_per_iteration=600,
                        verbose_logging=True,
                        save_history=True
                    )
                    
                    debugging_agent = DebuggingAgent(
                        model_config_name=self.model.config_name,
                        config=config
                    )
                    
                    result = asyncio.run(debugging_agent.auto_debug(self.scene_name))
                    success = self._handle_debug_result(result, "DebuggingAgent")
                    if success:
                        return True
                    logger.error("DebuggingAgent also failed")
                except Exception as e:
                    logger.error(f"DebuggingAgent failed with error: {e}")
            
            # If no debugging methods are available
            if not DEBUGGING_AGENT_AVAILABLE:
                logger.warning("No debugging functionality available - scenario created but not debugged")
                logger.warning("Consider installing debugging dependencies or running manual tests")
                return True  # Don't fail the entire pipeline
            
            logger.error("All debugging methods failed")
            return False
            
        except Exception as e:
            logger.error(f"Failed to debug environment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _handle_debug_result(self, result, method_name: str) -> bool:
        """Handle and format debug results from different debugging methods."""
        try:
            # Handle DebuggingResult object (from both methods)
            if hasattr(result, 'success'):
                if result.success:
                    output_message = f"✅ Environment '{self.scene_name}' debug finished successfully using {method_name}.\n"
                    output_message += f"Total iterations: {result.iterations}\n"
                    output_message += f"Final status: 调试成功，环境运行正常"
                    logger.info(output_message)
                    
                    # Store debug results in statistics
                    self.state["statistics"]["debug_results"] = {
                        "method": method_name,
                        "success": True,
                        "iterations": result.iterations,
                        "final_status": "调试成功，环境运行正常"
                    }
                    return True
                else:
                    output_message = f"❌ Environment '{self.scene_name}' debug FAILED using {method_name}.\n"
                    output_message += f"Error: {result.final_error}\n"
                    output_message += f"Iterations completed: {result.iterations}\n"
                    output_message += f"Final status: 调试失败"
                    logger.error(output_message)
                    
                    # Store debug results in statistics
                    self.state["statistics"]["debug_results"] = {
                        "method": method_name,
                        "success": False,
                        "iterations": result.iterations,
                        "error": str(result.final_error),
                        "final_status": "调试失败"
                    }
                    return False
            
            # Handle dictionary format (fallback compatibility)
            elif isinstance(result, dict):
                if result.get("success"):
                    output_message = f"✅ Environment '{self.scene_name}' debug finished successfully using {method_name}.\n"
                    output_message += f"Total iterations: {result.get('total_iterations', 'N/A')}\n"
                    output_message += f"Final status: {result.get('final_status', 'Success')}"
                    logger.info(output_message)
                    
                    self.state["statistics"]["debug_results"] = {
                        "method": method_name,
                        "success": True,
                        "iterations": result.get('total_iterations', 0),
                        "final_status": result.get('final_status', 'Success')
                    }
                    return True
                else:
                    error_msg = result.get('error', result.get('final_error', 'No specific error message.'))
                    output_message = f"❌ Environment '{self.scene_name}' debug FAILED using {method_name}.\n"
                    output_message += f"Error: {error_msg}\n"
                    output_message += f"Final status: {result.get('final_status', '调试失败')}"
                    logger.error(output_message)
                    
                    self.state["statistics"]["debug_results"] = {
                        "method": method_name,
                        "success": False,
                        "error": error_msg,
                        "final_status": result.get('final_status', '调试失败')
                    }
                    return False
            
            else:
                logger.error(f"Unknown result format from {method_name}: {type(result)}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling debug result from {method_name}: {e}")
            return False
    
    def count_lines_of_code(self, code_path: Path) -> int:
        """Count total lines of code in all Python files."""
        total_lines = 0
        if code_path.exists():
            for file_path in code_path.rglob("*.py"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    continue
        return total_lines
    
    def count_files_by_type(self, env_path: Path) -> Dict[str, int]:
        """Count files by type in the environment directory."""
        file_counts = {"code": 0, "profile": 0, "other": 0}
        
        # Count code files
        code_path = env_path / "code"
        if code_path.exists():
            file_counts["code"] = len(list(code_path.rglob("*.py")))
        
        # Count profile files
        profile_path = env_path / "profile"
        if profile_path.exists():
            file_counts["profile"] = len(list(profile_path.rglob("*.json"))) + len(list(profile_path.rglob("*.csv")))
        
        # Count other files
        file_counts["other"] += len(list(env_path.glob("*.json")))
        file_counts["other"] += len(list(env_path.glob("*.html")))
        
        return file_counts
    
    def save_generation_statistics(self):
        """Save generation statistics to a JSON file."""
        try:
            # Get token usage stats
            try:
                token_stats = get_token_usage_stats()
                token_cost = estimate_token_cost()
                
                logger.info(f"Total token usage: {token_stats['total_tokens']} tokens")
                logger.info(f"  - Prompt tokens: {token_stats['total_prompt_tokens']}")
                logger.info(f"  - Completion tokens: {token_stats['total_completion_tokens']}")
                logger.info(f"Estimated cost: ${token_cost['total_cost_usd']:.4f}")
                
                token_usage_data = {
                    "total_tokens": token_stats['total_tokens'],
                    "prompt_tokens": token_stats['total_prompt_tokens'],
                    "completion_tokens": token_stats['total_completion_tokens'],
                    "request_count": token_stats['request_count'],
                    "tokens_per_second": token_stats['tokens_per_second'],
                    "estimated_cost": token_cost['total_cost_usd'],
                    "model_breakdown": {}
                }
                
                for model_name, usage in token_stats['model_usage'].items():
                    token_usage_data["model_breakdown"][model_name] = {
                        "total_tokens": usage['total_tokens'],
                        "prompt_tokens": usage['prompt_tokens'],
                        "completion_tokens": usage['completion_tokens'],
                        "request_count": usage['request_count']
                    }
                    
                    if model_name in token_cost.get('model_costs', {}):
                        token_usage_data["model_breakdown"][model_name]["cost"] = token_cost['model_costs'][model_name]['total_cost']
                        
            except Exception as e:
                logger.warning(f"Error getting token usage statistics: {e}")
                token_usage_data = {}
            
            stats_to_save = {
                "generation_date": datetime.now().isoformat(),
                "environment_name": self.scene_name,
                "research_topic": self.research_topic,
                "total_generation_time_seconds": self.state["statistics"].get("total_generation_time", 0),
                "workflow_generation_time_seconds": self.state["statistics"].get("workflow_generation_time", 0),
                "code_generation_time_seconds": self.state["statistics"].get("code_generation_time", 0),
                "code_verification_issues": self.state["statistics"].get("code_verification_issues", {}),
                "workflow_validation_issues": self.state["statistics"].get("workflow_validation_issues", []),
                "token_usage": token_usage_data,
                "files_generated": self.state["statistics"].get("files_generated", {}),
                "total_lines_of_code": self.state["statistics"].get("total_lines_of_code", 0),
                "agent_types": self.state.get("agent_types", []),
                "errors": self.state["statistics"].get("errors", [])
            }
            
            # Save to file
            stats_file = self.env_path / "generation_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, indent=2)
            logger.info(f"✓ Generation statistics saved to {stats_file}")
        except Exception as e:
            logger.error(f"Error saving generation statistics: {e}")
    
    def run_full_pipeline(self) -> bool:
        """Execute the complete scenario creation pipeline."""
        try:
            logger.info("=" * 100)
            logger.info("ONESIM SCENARIO CREATION PIPELINE")
            logger.info("=" * 100)
            logger.info(f"Scene: {self.scene_name}")
            logger.info(f"Topic: {self.research_topic}")
            logger.info("=" * 100)
            
            start_time = time.time()
            
            # Create environment directory
            self.create_directory(self.env_path)
            
            # Setup logging
            self.setup_logging()
            
            # Initialize LLM
            self.initialize_llm()

            # Load scenario config
            self.state["scenario_config"] = self.load_scenario_config()
            

            # Execute pipeline steps
            steps = [
                ("Generate Agent Types", self.generate_agent_types),
                ("Generate Workflow", self.generate_workflow),
                ("Generate Metrics", self.generate_metrics),
                ("Generate Environment Data", self.generate_env_data),
                ("Generate Code", self.generate_code),
                ("Save Profile Schemas", self.save_profile_schemas),
                ("Generate Relationship Schema", self.generate_relationship_schema),
                ("Generate Topology Configuration", self.generate_topology_config),
                ("Generate Profiles", self.generate_profiles),
                ("Generate Relationships", self.generate_relationships),
                ("Debug Environment", self.debug_environment)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"\n--- {step_name} ---")
                if not step_func():
                    logger.error(f"Pipeline failed at step: {step_name}")
                    return False
                
                # Log token usage after each step
                try:
                    log_token_usage()
                except Exception as e:
                    logger.warning(f"Error logging token usage: {e}")
            
            # Calculate total time and save statistics
            total_time = time.time() - start_time
            self.state["statistics"]["total_generation_time"] = total_time
            self.state["statistics"]["end_time"] = time.time()
            
            self.save_generation_statistics()
            
            logger.info("\n" + "=" * 100)
            logger.info("SCENARIO CREATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 100)
            logger.info(f"✓ Scene created: {self.env_path}")
            logger.info(f"✓ Total time: {total_time:.2f} seconds")
            logger.info(f"✓ Agent types: {len(self.state.get('agent_types', []))}")
            logger.info(f"✓ Lines of code: {self.state['statistics'].get('total_lines_of_code', 0)}")
            logger.info("=" * 100)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OneSim Complete Scenario Creation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create scenario from research topic
  python src/researcher/scenario_creation.py --topic "social network dynamics" --scene-name "social_dynamics"
  
  # Create scenario with specific model
  python src/researcher/scenario_creation.py --topic "cultural diffusion" --scene-name "culture_sim" --model "gpt-4"
  
  # Create scenario with custom config
  python src/researcher/scenario_creation.py --config config/my_scenario.json
        """
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic for scenario generation"
    )
    
    parser.add_argument(
        "--scene_name",
        type=str,
        help="Name of the scenario to create"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.json",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model configuration name to use"
    )
    
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base directory path (defaults to project root)"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        help="Project name"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to scenario configuration JSON file"
    )
    
    parser.add_argument(
        "--num_profiles",
        type=int,
        default=5,
        help="Number of profiles to generate per agent type"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_args()
        
        # Load from config file if provided
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            topic = config.get("topic", args.topic)
            scene_name = config.get("scene_name", args.scene_name)
            model_config = config.get("model_config", args.model_config)
            model = config.get("model", args.model_name)
            base_path = config.get("base_path", args.base_path)
            project_name = config.get("project_name", args.project_name)
        else:
            topic = args.topic
            scene_name = args.scene_name
            model_config = args.model_config
            model = args.model_name
            base_path = args.base_path
            project_name = None
        # Validate required arguments
        if not topic:
            logger.error("Research topic is required (--topic or config file)")
            return 1
        
        if not scene_name:
            logger.error("Scene name is required (--scene-name or config file)")
            return 1
        
        # Create and run pipeline
        pipeline = ScenarioCreationPipeline(
            scene_name=scene_name,
            research_topic=topic,
            model_config_path=model_config,
            selected_model=model,
            base_path=base_path,
            project_name=project_name
        )
        
        success = pipeline.run_full_pipeline()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())