#!/usr/bin/env python3
"""
Experiment Project Coordinator

This module provides the main coordination class for experimental projects,
integrating profile modifications, interventions, and configuration generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from onesim.profile import AgentProfile, AgentSchema
from onesim.simulator import AgentFactory

from .profile_modifier import ProfileModifier, ModificationResult
from .intervention_engine import InterventionEngine, InterventionResult
from .config_generator import ConfigGenerator, ExperimentGroupConfig
from loguru import logger


@dataclass
class ProjectConfig:
    """Configuration for an experimental project."""
    project_name: str
    project_description: str
    base_environment: str
    base_config_path: str
    intervention_specs_path: str
    output_directory: str
    model_name: Optional[str] = None
    model_config_path: Optional[str] = None


class ExperimentProject:
    """
    Main coordinator for experimental projects.
    
    This class orchestrates the entire experimental workflow:
    1. Load base profiles and configurations
    2. Apply interventions to create modified profiles
    3. Generate experiment-specific configurations
    4. Coordinate experiment execution
    """
    
    def __init__(self, project_config: ProjectConfig):
        """
        Initialize the ExperimentProject.
        
        Args:
            project_config: Configuration for this experimental project
        """
        self.config = project_config
        self.project_path = Path(project_config.output_directory)
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.profile_modifier = ProfileModifier(project_config.model_name)
        self.intervention_engine = InterventionEngine(project_config.model_name)
        self.config_generator = ConfigGenerator(project_config.base_config_path)
        
        # Load intervention specifications
        self.intervention_engine.load_intervention_specifications(
            project_config.intervention_specs_path
        )
        
        # Project state
        self.base_profiles: Dict[str, List[AgentProfile]] = {}
        self.experiment_results: Dict[str, InterventionResult] = {}
        self.generated_configs: Dict[str, str] = {}
        self.current_run_id: Optional[str] = None
        self.environment_path: Optional[str] = None  # Store actual environment path
        
        logger.info(f"Initialized experimental project: {project_config.project_name}")
    
    def load_experiment_config(self, experiment_config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and parse experiment configuration from file.
        
        Args:
            experiment_config_path: Path to experiment configuration file
            
        Returns:
            Parsed experiment configuration dictionary
        """
        config_path = Path(experiment_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            experiment_config = json.load(f)
        
        logger.info(f"Loaded experiment configuration from {config_path}")
        return experiment_config
    
    def parse_intervention_assignments(self, experiment_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse intervention assignments from unified configuration structure.
        
        In the new unified structure, intervention specifications are fully defined
        in intervention_specifications.json organized by treatment groups.
        The experiment_config.json only contains treatment group references.
        
        Args:
            experiment_config: Experiment configuration dictionary
            
        Returns:
            List of intervention configurations for apply_interventions method
        """
        intervention_configs = []
        
        # In unified structure, we create intervention configs directly from treatment groups
        experimental_groups = experiment_config.get('experimental_groups', {})
        # Process treatment groups - each maps to a treatment group in intervention specs
        for treatment_group in experimental_groups.get('treatment_groups', []):
            group_name = treatment_group['name']
            
            # Create intervention config that references the treatment group in intervention specs
            intervention_config = {
                'treatment_group_name': group_name,  # Maps to treatment_groups key in intervention_specs
                'target_groups': [group_name],
                'timing': 'pre_simulation',
                'llm_enhancement': False
            }
            intervention_configs.append(intervention_config)
        
        # Process replicates - they inherit interventions from their base treatment
        for replicate in experimental_groups.get('replicates', []):
            base_treatment = replicate.get('base_treatment')
            if base_treatment:
                replicate_name = replicate['name']
                
                # Create intervention config for replicate using base treatment specs
                intervention_config = {
                    'treatment_group_name': base_treatment,  # Use base treatment's intervention spec
                    'target_groups': [replicate_name],  # But target the replicate group
                    'timing': 'pre_simulation',
                    'llm_enhancement': False
                }
                intervention_configs.append(intervention_config)
        
        logger.info(f"Parsed {len(intervention_configs)} intervention assignments from unified structure")
        return intervention_configs
    
    def load_base_profiles(self, environment_path: Union[str, Path]) -> Dict[str, List[AgentProfile]]:
        """
        Load base profiles from the specified environment.
        
        For simplified operation, this method still creates AgentProfile objects
        but the ProfileModifier will convert them to dictionaries as needed.
        
        Args:
            environment_path: Path to the environment directory
            
        Returns:
            Dictionary mapping agent types to lists of profiles
        """
        env_path = Path(environment_path)
        profile_data_path = env_path / "profile" / "data"
        profile_schema_path = env_path / "profile" / "schema"
        
        if not profile_data_path.exists() or not profile_schema_path.exists():
            raise FileNotFoundError(f"Profile directories not found in {env_path}")
        
        profiles_by_type = {}
        
        # Load each agent type
        for schema_file in profile_schema_path.glob("*.json"):
            agent_type = schema_file.stem
            
            # Load schema
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            schema = AgentSchema(schema_data)
            
            # Load profile data
            profile_data_file = profile_data_path / f"{agent_type}.json"
            if profile_data_file.exists():
                with open(profile_data_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                
                # Create AgentProfile objects (ProfileModifier will convert to dict as needed)
                profiles = []
                for profile_data in profiles_data:
                    # Ensure agent_type is in profile data
                    if 'agent_type' not in profile_data:
                        profile_data['agent_type'] = agent_type
                    profile = AgentProfile(agent_type, schema, profile_data)
                    profiles.append(profile)
                
                profiles_by_type[agent_type] = profiles
                logger.info(f"Loaded {len(profiles)} {agent_type} profiles")
        
        self.base_profiles = profiles_by_type
        self.environment_path = str(environment_path)  # Store the actual environment path
        return profiles_by_type
    
    def design_experiment(
        self,
        experiment_design: Dict[str, Any]
    ) -> List[ExperimentGroupConfig]:
        """
        Design experimental groups based on specification.
        
        Args:
            experiment_design: Experiment design specification
            
        Returns:
            List of experiment group configurations
        """
        groups = []
        
        # Create control group
        if 'control_group' in experiment_design:
            control_spec = experiment_design['control_group']
            control_group = ExperimentGroupConfig(
                group_id=control_spec['name'],
                group_type='control',
                description=control_spec['description'],
                profile_modifications=[],
                random_seed=control_spec.get('parameters', {}).get('random_seed')
            )
            groups.append(control_group)
        
        # Create treatment groups
        if 'treatment_groups' in experiment_design:
            for treatment_spec in experiment_design['treatment_groups']:
                # Use the treatment group name as the intervention identifier
                intervention_id = treatment_spec['name']
                treatment_group = ExperimentGroupConfig(
                    group_id=treatment_spec['name'],
                    group_type='treatment',
                    description=treatment_spec['description'],
                    profile_modifications=[intervention_id],
                    custom_config_overrides=treatment_spec.get('parameters', {}),
                    random_seed=treatment_spec.get('parameters', {}).get('random_seed')
                )
                groups.append(treatment_group)
        
        # Create replicate groups
        if 'replicates' in experiment_design:
            for replicate_spec in experiment_design['replicates']:
                replicate_group = ExperimentGroupConfig(
                    group_id=replicate_spec['name'],
                    group_type='replicate',
                    description=replicate_spec['description'],
                    profile_modifications=[replicate_spec.get('base_treatment', '')],
                    random_seed=replicate_spec.get('parameters', {}).get('random_seed')
                )
                groups.append(replicate_group)
        
        logger.info(f"Designed experiment with {len(groups)} groups")
        return groups
    
    def apply_interventions(
        self,
        experiment_groups: List[ExperimentGroupConfig],
        intervention_configs: List[Dict[str, Any]],
        environment_path: Union[str, Path]
    ) -> Dict[str, InterventionResult]:
        """
        Apply interventions to create modified profiles for experimental groups.
        
        Args:
            experiment_groups: List of experimental group configurations
            intervention_configs: List of intervention application configurations
            environment_path: Path to base environment for copying control group profiles
            
        Returns:
            Dictionary mapping group IDs to intervention results
        """
        results = {}
        intervention_target_groups = set()
        
        # Track which groups have interventions applied
        for intervention_config in intervention_configs:
            intervention_target_groups.update(intervention_config['target_groups'])
        
        for intervention_config in intervention_configs:
            treatment_group_name = intervention_config['treatment_group_name']
            target_groups = intervention_config['target_groups']
            
            for group_config in experiment_groups:
                if group_config.group_id in target_groups:
                    # Get profiles for the agent types this group uses
                    all_profiles = []
                    for agent_type, profiles in self.base_profiles.items():
                        all_profiles.extend(profiles)
                    
                    # Apply profile modifications using the treatment group specifications
                    # The InterventionEngine will use the treatment group data from intervention specs
                    intervention_result = self.intervention_engine.apply_intervention(
                        treatment_group_name=treatment_group_name,
                        profiles=all_profiles,
                        target_selection={},  # Empty - target selection is embedded in intervention specs
                        llm_enhancement=intervention_config.get('llm_enhancement', False)
                    )
                    
                    results[group_config.group_id] = intervention_result
                    
                    # Save modified profiles
                    if intervention_result.success:
                        self._save_group_profiles(group_config, intervention_result)
        
        # Handle control groups and other groups without interventions
        for group_config in experiment_groups:
            if group_config.group_id not in intervention_target_groups:
                # This is a control group or group without interventions
                self._save_control_group_profiles(group_config, environment_path)
        
        self.experiment_results = results
        return results
    
    def _save_group_profiles(
        self,
        group_config: ExperimentGroupConfig,
        intervention_result: InterventionResult
    ) -> None:
        """Save modified profiles for an experimental group in envs-like structure."""
        
        # Create fixed profile directory structure for this group (not in runs)
        group_dir = self.project_path / "groups" / group_config.group_id / "profile"
        profile_data_dir = group_dir / "data"
        profile_schema_dir = group_dir / "schema"
        
        profile_data_dir.mkdir(parents=True, exist_ok=True)
        profile_schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the actual environment path for copying files
        env_path = Path(self.environment_path) if self.environment_path else Path(self.config.base_environment)
        original_data_dir = env_path / "profile" / "data"
        original_schema_dir = env_path / "profile" / "schema"
        
        # First, copy ALL original files (complete profile directory)
        if original_data_dir.exists():
            import shutil
            # Copy all files from data directory (including .json and .csv files)
            for data_file in original_data_dir.iterdir():
                if data_file.is_file():
                    shutil.copy2(data_file, profile_data_dir / data_file.name)
        
        if original_schema_dir.exists():
            import shutil
            # Copy all schema files
            for schema_file in original_schema_dir.iterdir():
                if schema_file.is_file():
                    shutil.copy2(schema_file, profile_schema_dir / schema_file.name)
        
        # Group modified profiles by agent type
        profiles_by_type = {}
        modified_agent_types = set()
        for result in intervention_result.modification_results:
            if result.success:
                agent_type = result.modified_profile.get('agent_type', 'unknown')
                if agent_type not in profiles_by_type:
                    profiles_by_type[agent_type] = []
                profiles_by_type[agent_type].append(result.modified_profile)
                modified_agent_types.add(agent_type)
        
        # Save modified profile data (overwriting the copied original files)
        for agent_type, modified_profiles in profiles_by_type.items():
            # modified_profiles are already dictionaries, so save them directly
            profile_data_file = profile_data_dir / f"{agent_type}.json"
            with open(profile_data_file, 'w', encoding='utf-8') as f:
                json.dump(modified_profiles, f, indent=2, ensure_ascii=False)
        
        # Save modified schemas if there are schema changes
        if intervention_result.modification_results:
            self._save_modified_schemas_for_group(intervention_result, profile_schema_dir, modified_agent_types)
        
        logger.info(f"Saved complete profiles and schemas for {group_config.group_id}")
    
    def _save_control_group_profiles(
        self,
        group_config: ExperimentGroupConfig,
        environment_path: Union[str, Path]
    ) -> None:
        """Copy complete original profiles for control group in envs-like structure."""
        
        # Create fixed profile directory structure for this group (not in runs)
        group_dir = self.project_path / "groups" / group_config.group_id / "profile"
        profile_data_dir = group_dir / "data"
        profile_schema_dir = group_dir / "schema"
        
        profile_data_dir.mkdir(parents=True, exist_ok=True)
        profile_schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy complete original profile data and schema files from environment
        env_path = Path(environment_path)
        original_data_dir = env_path / "profile" / "data"
        original_schema_dir = env_path / "profile" / "schema"
        
        if original_data_dir.exists():
            import shutil
            # Copy ALL files from data directory (including .json and .csv files)
            for data_file in original_data_dir.iterdir():
                if data_file.is_file():
                    shutil.copy2(data_file, profile_data_dir / data_file.name)
        
        if original_schema_dir.exists():
            import shutil
            # Copy ALL files from schema directory
            for schema_file in original_schema_dir.iterdir():
                if schema_file.is_file():
                    shutil.copy2(schema_file, profile_schema_dir / schema_file.name)
        
        logger.info(f"Copied complete profiles and schemas for control group {group_config.group_id}")
    
    def _save_modified_schemas_for_group(
        self,
        intervention_result: InterventionResult,
        profile_schema_dir: Path,
        modified_agent_types: set
    ) -> None:
        """Save modified schemas for agent types that had schema changes."""
        
        # Collect schema changes by agent type
        schema_changes_by_type = {}
        for result in intervention_result.modification_results:
            if result.success and result.schema_changes:
                agent_type = result.modified_profile.get('agent_type', 'unknown')
                if agent_type not in schema_changes_by_type:
                    schema_changes_by_type[agent_type] = {}
                
                # Merge schema changes for this agent type
                for field, change in result.schema_changes.items():
                    if field not in schema_changes_by_type[agent_type]:
                        schema_changes_by_type[agent_type][field] = change

        # Apply schema modifications to copied schema files
        for agent_type, changes in schema_changes_by_type.items():
            schema_file = profile_schema_dir / f"{agent_type}.json"
            
            if schema_file.exists():
                # Load existing schema
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                # Apply schema changes
                modified_schema = self._apply_schema_changes(schema, changes)
                
                # Save modified schema
                with open(schema_file, 'w', encoding='utf-8') as f:
                    json.dump(modified_schema, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Updated schema for {agent_type} with {len(changes)} changes")
            else:
                logger.warning(f"Schema file not found for {agent_type}: {schema_file}")

    def _apply_schema_changes(self, base_schema: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply schema changes to a base schema."""
        import copy
        
        modified_schema = copy.deepcopy(base_schema)
        
        for field, change in changes.items():
            if change['operation'] == 'add':
                # Add new field to schema
                if 'fields' not in modified_schema:
                    modified_schema['fields'] = {}
                modified_schema['fields'][field] = change['field_config']
                logger.debug(f"Added field {field} to schema")
                
            elif change['operation'] == 'remove':
                # Remove field from schema
                if 'fields' in modified_schema and field in modified_schema['fields']:
                    del modified_schema['fields'][field]
                    logger.debug(f"Removed field {field} from schema")
        
        return modified_schema
    
    def generate_configurations(
        self,
        experiment_groups: List[ExperimentGroupConfig]
    ) -> Dict[str, str]:
        """
        Generate simulation configurations for all experimental groups.
        
        Args:
            experiment_groups: List of experimental group configurations
            
        Returns:
            Dictionary mapping group IDs to configuration file paths
        """
        # Collect profile and schema paths for all groups (both treatment and control)
        profile_paths_by_group = {}
        schema_paths_by_group = {}
        
        for group_config in experiment_groups:
            # Check fixed profile structure for all groups
            group_profile_dir = self.project_path / "groups" / group_config.group_id / "profile"
            if not group_profile_dir.exists():
                continue
            profile_data_dir = group_profile_dir / "data"
            profile_schema_dir = group_profile_dir / "schema"
            
            if profile_data_dir.exists() and profile_schema_dir.exists():
                profile_paths = {}
                schema_paths = {}
                
                # Collect profile data paths
                for profile_file in profile_data_dir.glob("*.json"):
                    agent_type = profile_file.stem
                    profile_paths[agent_type] = str(profile_file.absolute())
                
                # Collect schema paths
                for schema_file in profile_schema_dir.glob("*.json"):
                    agent_type = schema_file.stem
                    schema_paths[agent_type] = str(schema_file.absolute())
                
                if profile_paths:
                    profile_paths_by_group[group_config.group_id] = profile_paths
                if schema_paths:
                    schema_paths_by_group[group_config.group_id] = schema_paths
        
        # First, generate intervention configs for runtime interventions
        self._generate_intervention_configs(experiment_groups)
        
        # Then generate main configurations in each group's directory
        generated_configs = {}
        
        for group_config in experiment_groups:
            # Save config.json to group root directory (fixed location)
            group_config_dir = self.project_path / "groups" / group_config.group_id
            group_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this group has runtime interventions
            intervention_config_file = group_config_dir / "intervention_config.json"
            has_runtime_interventions = intervention_config_file.exists()
            
            config_file_path = self.config_generator.generate_single_experiment_config(
                group_config=group_config,
                output_dir=group_config_dir,
                modified_profile_paths=profile_paths_by_group.get(group_config.group_id),
                modified_schema_paths=schema_paths_by_group.get(group_config.group_id),
                environment_name=self.environment_path,
                has_runtime_interventions=has_runtime_interventions
            )
            
            generated_configs[group_config.group_id] = config_file_path
        
        self.generated_configs = generated_configs
        
        return generated_configs
    
    def _generate_intervention_configs(self, experiment_groups: List[ExperimentGroupConfig]) -> None:
        """Generate intervention_config.json files for groups with runtime interventions."""
        
        treatment_group_data = self._get_treatment_group_data()
        
        for group_config in experiment_groups:
            runtime_interventions = []
            
            # Check if this group corresponds to a treatment group with runtime modifications
            base_group_name = group_config.group_id.replace('_group_', '_group_').split('_group_')[0] + '_group_' + group_config.group_id.split('_')[-1] if 'treatment_group_' in group_config.group_id else group_config.group_id
            
            # Look for runtime modifications in the original intervention specs
            for treatment_group_id, treatment_data in treatment_group_data.items():
                if (treatment_group_id in group_config.group_id or 
                    group_config.group_id in treatment_group_id or
                    treatment_group_id == base_group_name):
                    
                    if 'runtime_modifications' in treatment_data:
                        runtime_interventions.extend(treatment_data['runtime_modifications'])
            
            # Also check for replicate groups that inherit from base treatment
            if group_config.group_type == 'replicate' and hasattr(group_config, 'profile_modifications'):
                base_treatment = group_config.profile_modifications[0] if group_config.profile_modifications else None
                if base_treatment and base_treatment in treatment_group_data:
                    treatment_data = treatment_group_data[base_treatment]
                    if 'runtime_modifications' in treatment_data:
                        runtime_interventions.extend(treatment_data['runtime_modifications'])
            
            # Create intervention_config.json if there are runtime interventions
            if runtime_interventions:
                # Save intervention config to group root directory (fixed location)
                group_dir = self.project_path / "groups" / group_config.group_id
                intervention_config_file = group_dir / "intervention_config.json"
                
                # Add indexed IDs to runtime interventions for clarity
                indexed_runtime_interventions = []
                for i, intervention in enumerate(runtime_interventions):
                    intervention_with_id = intervention.copy()
                    intervention_with_id['intervention_id'] = f"intervention_runtime_{i}"
                    indexed_runtime_interventions.append(intervention_with_id)
                
                intervention_config = {
                    'runtime_interventions': indexed_runtime_interventions,
                    'metadata': {
                        'group_id': group_config.group_id,
                        'group_type': group_config.group_type,
                        'created_timestamp': self.config_generator._get_timestamp()
                    }
                }
                
                with open(intervention_config_file, 'w', encoding='utf-8') as f:
                    json.dump(intervention_config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Generated intervention_config.json for {group_config.group_id} with {len(runtime_interventions)} interventions")
    
    def _get_treatment_group_data(self) -> Dict[str, Any]:
        """Get the original treatment group data from intervention specs."""
        return self.intervention_engine.get_all_treatment_group_data()
    
    def _execute_simulations(self, generated_configs: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute OneSim simulations for all generated configurations.
        
        Args:
            generated_configs: Dictionary mapping group IDs to config file paths
            
        Returns:
            Dictionary with simulation execution results
        """
        simulation_results = {
            'total_simulations': len(generated_configs),
            'successful_simulations': 0,
            'failed_simulations': 0,
            'simulation_details': {}
        }
        
        # Generate a single run_id for all simulations in this batch
        batch_run_id = self._generate_run_id()
        logger.info(f"Using batch run ID for all simulations: {batch_run_id}")
        
        for group_id, config_path in generated_configs.items():
            logger.info(f"Executing simulation for {group_id}...")
            
            try:
                # Execute OneSim simulation using the generated config with shared run_id
                result = self._run_onesim_simulation(config_path, group_id, batch_run_id)
                
                simulation_results['simulation_details'][group_id] = {
                    'status': 'completed' if result['success'] else 'failed',
                    'config_path': config_path,
                    'simulation_output': result.get('output_path'),
                    'execution_time': result.get('execution_time'),
                    'error': result.get('error'),
                    'run_id': batch_run_id,
                    'log_files': result.get('log_files', [])
                }
                
                if result['success']:
                    simulation_results['successful_simulations'] += 1
                    log_files_info = f" (logs: {len(result.get('log_files', []))} files)" if result.get('log_files') else ""
                    logger.info(f"✅ Simulation completed for {group_id}{log_files_info}")
                else:
                    simulation_results['failed_simulations'] += 1
                    log_files_info = f" (logs: {len(result.get('log_files', []))} files)" if result.get('log_files') else ""
                    logger.error(f"❌ Simulation failed for {group_id}: {result.get('error')}{log_files_info}")
                    
            except Exception as e:
                simulation_results['failed_simulations'] += 1
                
                # Try to check for log files even when exception occurred
                try:
                    run_dir = self.project_path / "groups" / group_id / "runs" / batch_run_id
                    log_files = list(run_dir.glob("*.log")) if run_dir.exists() else []
                    log_info = [str(log_file) for log_file in log_files]
                except Exception:
                    log_info = []
                
                simulation_results['simulation_details'][group_id] = {
                    'status': 'failed',
                    'config_path': config_path,
                    'error': str(e),
                    'run_id': batch_run_id,
                    'log_files': log_info
                }
                logger.error(f"❌ Simulation exception for {group_id}: {e}")
        
        logger.info(f"Simulations completed: {simulation_results['successful_simulations']}/{simulation_results['total_simulations']} successful")
        return simulation_results
    
    def _run_onesim_simulation(self, config_path: str, group_id: str, run_id: str = None) -> Dict[str, Any]:
        """
        Run a single OneSim simulation.
        
        Args:
            config_path: Path to the simulation config file
            group_id: ID of the experimental group
            run_id: Optional run ID to use; if None, generates a new one
            
        Returns:
            Dictionary with simulation results
        """
        import subprocess
        import time
        from pathlib import Path
        
        start_time = time.time()
        
        try:
            # Use provided run_id or generate a new one
            if run_id is None:
                run_id = self._generate_run_id()
            run_dir = self.project_path / "groups" / group_id / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created run directory for {group_id}: {run_dir}")
            
            # Note: Working directory is set to run_dir for output to be saved there
            
            # Execute OneSim simulation using main.py
            # Get project root (this script is in src/researcher/experiment_platform/)
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            main_py_path = project_root / "src" / "main.py"
            
            # Use configured model_config_path or default
            if self.config.model_config_path:
                model_config_path = Path(self.config.model_config_path)
                if not model_config_path.is_absolute():
                    model_config_path = project_root / self.config.model_config_path
            else:
                model_config_path = project_root / "config" / "model_config.json"
            
            # Check if model config exists
            if not model_config_path.exists():
                logger.warning(f"Model config file not found: {model_config_path}")
                logger.warning("Simulation may fail if model configuration is required")
            
            cmd = [
                "python", str(main_py_path),
                "--config", str(config_path),
                "--model_config", str(model_config_path),
                "--mode", "single",
                "--output_dir", str(run_dir),
                "--log_dir", str(run_dir)  # Add log directory parameter for run-specific logging
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            logger.info(f"Working directory: {run_dir}")
            logger.info(f"Run logs will be stored in: {run_dir}")
            
            # Run the simulation
            result = subprocess.run(
                cmd,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=36000  # 10 hour timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Check if log files were created
                log_files = list(run_dir.glob("*.log"))
                log_info = [str(log_file) for log_file in log_files] if log_files else []
                
                return {
                    'success': True,
                    'output_path': str(run_dir),
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'log_files': log_info
                }
            else:
                # Check if log files were created even for failed runs
                log_files = list(run_dir.glob("*.log"))
                log_info = [str(log_file) for log_file in log_files] if log_files else []
                
                return {
                    'success': False,
                    'error': f"Simulation failed with return code {result.returncode}",
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'log_files': log_info
                }
                
        except subprocess.TimeoutExpired:
            # Check for log files even on timeout
            log_files = list(run_dir.glob("*.log"))
            log_info = [str(log_file) for log_file in log_files] if log_files else []
            
            return {
                'success': False,
                'error': "Simulation timed out after 10 hour",
                'execution_time': time.time() - start_time,
                'log_files': log_info
            }
        except Exception as e:
            # Check for log files even on exception
            log_files = list(run_dir.glob("*.log"))
            log_info = [str(log_file) for log_file in log_files] if log_files else []
            
            return {
                'success': False,
                'error': f"Simulation execution error: {str(e)}",
                'execution_time': time.time() - start_time,
                'log_files': log_info
            }
    
    def run_full_workflow_from_config(
        self,
        environment_path: Union[str, Path],
        experiment_config_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Execute the complete experimental workflow from configuration files.
        
        Args:
            environment_path: Path to base environment
            experiment_config_path: Path to experiment configuration file
            
        Returns:
            Complete workflow results
        """
        # Load experiment configuration
        experiment_config = self.load_experiment_config(experiment_config_path)
        
        # Parse intervention assignments
        intervention_configs = self.parse_intervention_assignments(experiment_config)
        
        # Design experiment from experimental_groups section
        experiment_design = experiment_config.get('experimental_groups', {})
        
        # Execute common workflow with config file path for tracking
        result = self._execute_common_workflow(
            environment_path=environment_path,
            experiment_design=experiment_design,
            intervention_configs=intervention_configs,
            experiment_config_path=experiment_config_path
        )
        
        return result
    
    def run_full_workflow(
        self,
        environment_path: Union[str, Path],
        experiment_design: Dict[str, Any],
        intervention_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the complete experimental workflow.
        
        Args:
            environment_path: Path to base environment
            experiment_design: Experiment design specification  
            intervention_configs: Intervention configurations
            
        Returns:
            Complete workflow results
        """
        return self._execute_common_workflow(
            environment_path=environment_path,
            experiment_design=experiment_design,
            intervention_configs=intervention_configs
        )
    
    def _execute_common_workflow(
        self,
        environment_path: Union[str, Path],
        experiment_design: Dict[str, Any],
        intervention_configs: List[Dict[str, Any]],
        experiment_config_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Execute the common workflow steps shared by both workflow methods.
        
        Args:
            environment_path: Path to base environment
            experiment_design: Experiment design specification
            intervention_configs: Intervention configurations  
            experiment_config_path: Optional path to config file for tracking
            
        Returns:
            Complete workflow results
        """
        workflow_results = {
            'project_name': self.config.project_name,
            'status': 'running',
            'steps_completed': []
        }
        
        try:
            # Initialize run ID for this workflow
            self.current_run_id = self._generate_run_id()
            logger.info(f"Starting experimental workflow with run ID: {self.current_run_id}")
            
            # Step 1: Load base profiles
            logger.info("Step 1: Loading base profiles...")
            base_profiles = self.load_base_profiles(environment_path)
            workflow_results['steps_completed'].append('load_profiles')
            workflow_results['base_profiles_count'] = sum(len(profiles) for profiles in base_profiles.values())
            
            # Step 2: Design experiment
            logger.info("Step 2: Designing experiment...")
            experiment_groups = self.design_experiment(experiment_design)
            workflow_results['steps_completed'].append('design_experiment')
            workflow_results['experiment_groups_count'] = len(experiment_groups)
            
            # Step 3: Apply interventions and setup profiles
            logger.info("Step 3: Applying interventions and setting up profiles...")
            # intervention_results = self.apply_interventions(experiment_groups, intervention_configs, environment_path)
            intervention_results={}
            workflow_results['steps_completed'].append('apply_interventions')
            workflow_results['interventions_applied'] = len(intervention_results)
           
            
            # Step 4: Generate configurations
            logger.info("Step 4: Generating configurations...")
            generated_configs = self.generate_configurations(experiment_groups)
            workflow_results['steps_completed'].append('generate_configurations')
            workflow_results['configurations_generated'] = len(generated_configs)
            
            # Step 5: Execute simulations
            logger.info("Step 5: Executing simulations...")
            simulation_results = self._execute_simulations(generated_configs)
            workflow_results['steps_completed'].append('execute_simulations')
            workflow_results['simulation_results'] = simulation_results
            
            # Step 6: Save project summary
            logger.info("Step 6: Saving project summary...")
            self._save_project_summary(workflow_results, experiment_groups, intervention_results)
            workflow_results['steps_completed'].append('save_summary')
            
            workflow_results['status'] = 'completed'
            workflow_results['output_directory'] = str(self.project_path)
            
            # Add experiment config path if provided
            if experiment_config_path:
                workflow_results['experiment_config_used'] = str(experiment_config_path)
            
            logger.info(f"Experimental workflow completed successfully for {self.config.project_name}")
            
        except Exception as e:
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            logger.error(f"Experimental workflow failed: {e}")
            raise
        
        return workflow_results
    
    def _save_project_summary(
        self,
        workflow_results: Dict[str, Any],
        experiment_groups: List[ExperimentGroupConfig],
        intervention_results: Dict[str, InterventionResult]
    ) -> None:
        """Save a comprehensive project summary."""
        
        summary = {
            'project_info': {
                'name': self.config.project_name,
                'description': self.config.project_description,
                'base_environment': self.config.base_environment,
                'created_timestamp': self.config_generator._get_timestamp()
            },
            'workflow_results': workflow_results,
            'experiment_design': {
                'total_groups': len(experiment_groups),
                'group_details': [
                    {
                        'group_id': group.group_id,
                        'group_type': group.group_type,
                        'description': group.description,
                        'interventions': group.profile_modifications
                    }
                    for group in experiment_groups
                ]
            },
            'intervention_summary': {
                intervention_id: {
                    'treatment_group': intervention_id,
                    'success': result.success,
                    'targets': len(result.target_agents),
                    'modifications': len(result.modification_results),
                    'profile_interventions': result.summary.get('profile_interventions', {})
                }
                for intervention_id, result in intervention_results.items()
            },
            'generated_files': {
                'configurations': self.generated_configs,
                'project_directory': str(self.project_path)
            }
        }
        
        summary_file = self.project_path / "project_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Project summary saved to {summary_file}")
    
    def validate_project(self) -> Dict[str, Any]:
        """
        Validate the entire experimental project setup.
        
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'components_checked': []
        }
        
        # Check base configuration
        try:
            base_config_validation = self.config_generator.validate_config(self.config.base_config_path)
            validation_results['components_checked'].append('base_config')
            if not base_config_validation['valid']:
                validation_results['errors'].extend(base_config_validation['errors'])
                validation_results['valid'] = False
        except Exception as e:
            validation_results['errors'].append(f"Base config validation failed: {e}")
            validation_results['valid'] = False
        
        # Check intervention specifications
        try:
            available_interventions = self.intervention_engine.get_available_interventions()
            validation_results['components_checked'].append('interventions')
            validation_results['available_interventions'] = len(available_interventions)
        except Exception as e:
            validation_results['errors'].append(f"Intervention specs validation failed: {e}")
            validation_results['valid'] = False
        
        # Check generated configurations
        for group_id, config_path in self.generated_configs.items():
            try:
                config_validation = self.config_generator.validate_config(config_path)
                validation_results['components_checked'].append(f'config_{group_id}')
                if not config_validation['valid']:
                    validation_results['errors'].extend([
                        f"{group_id}: {error}" for error in config_validation['errors']
                    ])
                    validation_results['valid'] = False
                validation_results['warnings'].extend([
                    f"{group_id}: {warning}" for warning in config_validation.get('warnings', [])
                ])
            except Exception as e:
                validation_results['errors'].append(f"Config validation failed for {group_id}: {e}")
                validation_results['valid'] = False
        
        return validation_results
    
    def _generate_run_id(self) -> str:
        """Generate a human-readable timestamp-based run ID."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")