#!/usr/bin/env python3
"""
Configuration Generator for Experimental Platform

This module provides functionality to generate experiment-specific
configurations for different experimental groups.
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from loguru import logger


@dataclass
class ExperimentGroupConfig:
    """Configuration for a single experimental group."""
    group_id: str
    group_type: str  # 'control', 'treatment', 'replicate'
    description: str
    profile_modifications: List[str]  # List of intervention IDs applied
    custom_config_overrides: Optional[Dict[str, Any]] = None
    random_seed: Optional[int] = None


class ConfigGenerator:
    """
    Generates experiment-specific configurations for OneSim.
    
    This class creates customized configuration files for different
    experimental groups, ensuring each group uses the correct profile
    files and settings.
    """
    
    def __init__(self, base_config_path: Union[str, Path]):
        """
        Initialize the ConfigGenerator.
        
        Args:
            base_config_path: Path to the base configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.base_config = self._load_base_config()
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base configuration file not found: {self.base_config_path}")
        
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            if self.base_config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # For YAML files (if needed in the future)
                import yaml
                config = yaml.safe_load(f)
        
        logger.info(f"Loaded base configuration from {self.base_config_path}")
        return config
    
    def generate_experiment_configs(
        self,
        experiment_groups: List[ExperimentGroupConfig],
        output_dir: Union[str, Path],
        modified_profile_paths: Dict[str, Dict[str, str]] = None,
        environment_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate configuration files for all experimental groups.
        
        Args:
            experiment_groups: List of experimental group configurations
            output_dir: Directory to save generated configurations
            modified_profile_paths: Dict mapping group_id -> agent_type -> profile_path
            environment_name: Override environment name if different from base
            
        Returns:
            Dictionary mapping group_id to generated config file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_configs = {}
        
        for group_config in experiment_groups:
            try:
                config_file_path = self._generate_single_config(
                    group_config=group_config,
                    output_dir=output_path,
                    modified_profile_paths=modified_profile_paths,
                    environment_name=environment_name
                )
                generated_configs[group_config.group_id] = config_file_path
                
            except Exception as e:
                logger.error(f"Failed to generate config for group {group_config.group_id}: {e}")
                raise
        
        logger.info(f"Generated {len(generated_configs)} configuration files in {output_dir}")
        return generated_configs
    
    def _generate_single_config(
        self,
        group_config: ExperimentGroupConfig,
        output_dir: Path,
        modified_profile_paths: Dict[str, Dict[str, str]] = None,
        environment_name: Optional[str] = None
    ) -> str:
        """Generate configuration for a single experimental group."""
        
        # Create a deep copy of the base configuration
        experiment_config = copy.deepcopy(self.base_config)
        
        # Update environment name if specified
        if environment_name:
            if 'simulator' not in experiment_config:
                experiment_config['simulator'] = {}
            if 'environment' not in experiment_config['simulator']:
                experiment_config['simulator']['environment'] = {}
            experiment_config['simulator']['environment']['name'] = environment_name
        
        # Set random seed if specified
        if group_config.random_seed is not None:
            # Add random seed to environment config
            if 'additional_config' not in experiment_config['simulator']['environment']:
                experiment_config['simulator']['environment']['additional_config'] = {}
            experiment_config['simulator']['environment']['additional_config']['random_seed'] = group_config.random_seed
        
        # Apply custom configuration overrides
        if group_config.custom_config_overrides:
            experiment_config = self._apply_config_overrides(
                experiment_config, group_config.custom_config_overrides
            )
        
        # Add experiment metadata
        experiment_config['experiment_metadata'] = {
            'group_id': group_config.group_id,
            'group_type': group_config.group_type,
            'description': group_config.description,
            'profile_modifications': group_config.profile_modifications,
            'generation_timestamp': self._get_timestamp()
        }
        
        # Create profile path configuration if modified profiles exist
        if modified_profile_paths and group_config.group_id in modified_profile_paths:
            profile_config = self._create_profile_config(
                modified_profile_paths[group_config.group_id]
            )
            if 'agent' not in experiment_config:
                experiment_config['agent'] = {}
            experiment_config['agent']['profile_config'] = profile_config
        
        # Save the configuration file
        config_filename = f"{group_config.group_id}_config.json"
        config_file_path = output_dir / config_filename
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated configuration for {group_config.group_id}: {config_file_path}")
        return str(config_file_path)
    
    def _apply_config_overrides(
        self,
        config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply configuration overrides using dot notation paths."""
        
        def set_nested_value(dictionary: Dict, path: str, value: Any):
            """Set a nested dictionary value using dot notation path."""
            keys = path.split('.')
            for key in keys[:-1]:
                if key not in dictionary:
                    dictionary[key] = {}
                dictionary = dictionary[key]
            dictionary[keys[-1]] = value
        
        for path, value in overrides.items():
            set_nested_value(config, path, value)
            logger.debug(f"Applied config override: {path} = {value}")
        
        return config
    
    def _create_profile_config(self, profile_paths: Dict[str, str]) -> Dict[str, Any]:
        """Create profile configuration section."""
        return {
            'custom_profile_paths': profile_paths,
            'use_custom_profiles': True,
            'profile_loading_strategy': 'custom_paths'
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_experiment_run_configs(
        self,
        experiment_config_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Create individual run configurations from experiment configuration.
        
        Args:
            experiment_config_path: Path to experiment configuration JSON
            output_dir: Directory for individual run configs
            
        Returns:
            Dictionary mapping run_id to config file path
        """
        with open(experiment_config_path, 'r', encoding='utf-8') as f:
            experiment_config = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_configs = {}
        
        # Get experimental groups from the configuration
        experimental_groups = experiment_config.get('experimental_groups', {})
        
        # Generate configs for each group
        for group_type, group_data in experimental_groups.items():
            if group_type == 'control_group':
                group_config = ExperimentGroupConfig(
                    group_id=group_data['name'],
                    group_type='control',
                    description=group_data['description'],
                    profile_modifications=[],
                    random_seed=group_data.get('parameters', {}).get('random_seed')
                )
                
            elif group_type == 'treatment_groups':
                for treatment in group_data:
                    # Use the treatment group name as the intervention identifier
                    intervention_id = treatment['name']
                    group_config = ExperimentGroupConfig(
                        group_id=treatment['name'],
                        group_type='treatment',
                        description=treatment['description'],
                        profile_modifications=[intervention_id],
                        custom_config_overrides=treatment.get('parameters', {}),
                        random_seed=treatment.get('parameters', {}).get('random_seed')
                    )
                    
                    config_path = self._generate_single_config(
                        group_config, output_path
                    )
                    run_configs[group_config.group_id] = config_path
            
            elif group_type == 'replicates':
                for replicate in group_data:
                    group_config = ExperimentGroupConfig(
                        group_id=replicate['name'],
                        group_type='replicate',
                        description=replicate['description'],
                        profile_modifications=[replicate.get('base_treatment', '')],
                        random_seed=replicate.get('parameters', {}).get('random_seed')
                    )
                    
                    config_path = self._generate_single_config(
                        group_config, output_path
                    )
                    run_configs[group_config.group_id] = config_path
        
        return run_configs
    
    def validate_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a generated configuration file.
        
        Args:
            config_path: Path to configuration file to validate
            
        Returns:
            Validation result dictionary
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            return {'valid': False, 'errors': [f'Configuration file not found: {config_path}']}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            errors = []
            warnings = []
            
            # Check required sections
            required_sections = ['simulator', 'agent']
            for section in required_sections:
                if section not in config:
                    errors.append(f'Missing required section: {section}')
            
            # Check simulator environment configuration
            if 'simulator' in config and 'environment' in config['simulator']:
                env_config = config['simulator']['environment']
                if 'name' not in env_config:
                    errors.append('Missing environment name in simulator config')
            
            # Check agent configuration
            if 'agent' in config:
                agent_config = config['agent']
                if 'profile_config' in agent_config:
                    profile_config = agent_config['profile_config']
                    if 'custom_profile_paths' in profile_config:
                        # Validate profile paths exist
                        for agent_type, path in profile_config['custom_profile_paths'].items():
                            if not Path(path).exists():
                                warnings.append(f'Profile path does not exist: {path}')
            
            # Check experiment metadata
            if 'experiment_metadata' not in config:
                warnings.append('Missing experiment metadata')
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'config_sections': list(config.keys())
            }
            
        except json.JSONDecodeError as e:
            return {'valid': False, 'errors': [f'Invalid JSON format: {str(e)}']}
        except Exception as e:
            return {'valid': False, 'errors': [f'Validation error: {str(e)}']}
    
    def generate_batch_configs(
        self,
        batch_specification: Dict[str, Any],
        output_dir: Union[str, Path]
    ) -> Dict[str, List[str]]:
        """
        Generate multiple configuration files for batch experiments.
        
        Args:
            batch_specification: Specification for batch experiment generation
            output_dir: Output directory for generated configurations
            
        Returns:
            Dictionary mapping experiment names to list of config files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        batch_configs = {}
        
        for experiment_name, experiment_spec in batch_specification.items():
            experiment_dir = output_path / experiment_name
            experiment_dir.mkdir(exist_ok=True)
            
            # Create experimental groups from specification
            groups = []
            for group_spec in experiment_spec.get('groups', []):
                group_config = ExperimentGroupConfig(
                    group_id=group_spec['group_id'],
                    group_type=group_spec['group_type'],
                    description=group_spec['description'],
                    profile_modifications=group_spec.get('profile_modifications', []),
                    custom_config_overrides=group_spec.get('config_overrides'),
                    random_seed=group_spec.get('random_seed')
                )
                groups.append(group_config)
            
            # Generate configurations for this experiment
            generated_files = self.generate_experiment_configs(
                experiment_groups=groups,
                output_dir=experiment_dir,
                environment_name=experiment_spec.get('environment_name')
            )
            
            batch_configs[experiment_name] = list(generated_files.values())
        
        logger.info(f"Generated batch configurations for {len(batch_configs)} experiments")
        return batch_configs
    
    def generate_single_experiment_config(
        self,
        group_config: ExperimentGroupConfig,
        output_dir: Path,
        modified_profile_paths: Dict[str, str] = None,
        modified_schema_paths: Dict[str, str] = None,
        environment_name: Optional[str] = None,
        has_runtime_interventions: bool = False
    ) -> str:
        """
        Generate configuration for a single experimental group with full base config.
        
        Args:
            group_config: Configuration for the experimental group
            output_dir: Directory to save the configuration
            modified_profile_paths: Dict mapping agent_type -> profile_path
            modified_schema_paths: Dict mapping agent_type -> schema_path
            environment_name: Override environment name if different from base
            has_runtime_interventions: Whether this group has runtime interventions
            
        Returns:
            Path to generated configuration file
        """
        # Create a deep copy of the base configuration
        experiment_config = copy.deepcopy(self.base_config)
        
        # Update environment name if specified - use the actual environment, not workspace
        if environment_name:
            if 'simulator' not in experiment_config:
                experiment_config['simulator'] = {}
            if 'environment' not in experiment_config['simulator']:
                experiment_config['simulator']['environment'] = {}
            # Extract environment name from path if it's a full path
            env_name = Path(environment_name).name if Path(environment_name).exists() else environment_name
            experiment_config['simulator']['environment']['name'] = env_name
        
        # Set random seed if specified
        if group_config.random_seed is not None:
            # Add random seed to environment config
            if 'additional_config' not in experiment_config['simulator']['environment']:
                experiment_config['simulator']['environment']['additional_config'] = {}
            experiment_config['simulator']['environment']['additional_config']['random_seed'] = group_config.random_seed
        
        # Apply custom configuration overrides
        if group_config.custom_config_overrides:
            experiment_config = self._apply_config_overrides(
                experiment_config, group_config.custom_config_overrides
            )
        
        # Note: We don't add experiment_metadata to the config as it's not part of standard OneSim config
        # Experiment metadata is tracked separately by the experiment platform
        
        # Configure agent profiles with custom paths if modified profiles exist
        if modified_profile_paths or modified_schema_paths:
            if 'agent' not in experiment_config:
                experiment_config['agent'] = {}
            
            # Create profile section with agent-specific paths in the requested format
            if 'profile' not in experiment_config['agent']:
                experiment_config['agent']['profile'] = {}
            
            # Add profile and schema paths for each agent type
            if modified_profile_paths:
                for agent_type, profile_path in modified_profile_paths.items():
                    if agent_type not in experiment_config['agent']['profile']:
                        experiment_config['agent']['profile'][agent_type] = {}
                    
                    # Add count (this should be extracted from the actual profile data)
                    # For now, we'll load the profile file to count agents
                    try:
                        with open(profile_path, 'r', encoding='utf-8') as f:
                            profile_data = json.load(f)
                            agent_count = len(profile_data) if isinstance(profile_data, list) else 1
                        experiment_config['agent']['profile'][agent_type]['count'] = agent_count
                    except Exception as e:
                        logger.warning(f"Could not determine agent count for {agent_type}: {e}")
                        experiment_config['agent']['profile'][agent_type]['count'] = 10  # Default
                    
                    # Use absolute paths
                    experiment_config['agent']['profile'][agent_type]['profile_path'] = profile_path
                    
                    # Add schema path if available
                    if modified_schema_paths and agent_type in modified_schema_paths:
                        experiment_config['agent']['profile'][agent_type]['schema_path'] = modified_schema_paths[agent_type]
        
        # Ensure agent section has planning and memory configuration from base config
        if 'agent' in self.base_config:
            base_agent_config = self.base_config['agent']
            
            # Preserve planning configuration
            if 'planning' in base_agent_config:
                experiment_config['agent']['planning'] = base_agent_config['planning']
            
            # Preserve complete memory configuration
            if 'memory' in base_agent_config:
                experiment_config['agent']['memory'] = copy.deepcopy(base_agent_config['memory'])
        
        # Add intervention_config section if runtime interventions exist
        if has_runtime_interventions:
            intervention_config_path = output_dir / "intervention_config.json"
            experiment_config['intervention_config'] = {
                'enabled': True,
                'intervention_specs_path': str(intervention_config_path.absolute())
            }
        
        # Save the configuration file
        config_filename = f"config.json"
        config_file_path = output_dir / config_filename
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated complete configuration for {group_config.group_id}: {config_file_path}")
        return str(config_file_path)