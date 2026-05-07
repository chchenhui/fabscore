#!/usr/bin/env python3
"""
Profile Modifier for Experimental Platform

Simplified version that works with dictionaries instead of AgentProfile objects.
This is more suitable for experimental profile modifications since:
1. We only need to modify data, not use full AgentProfile functionality
2. Final output is JSON anyway
3. Much simpler code with fewer edge cases
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from onesim.models import get_model
from loguru import logger


@dataclass
class ModificationSpec:
    """Specification for a single profile field modification."""
    field: str
    modification_type: str  # 'set_value', 'add_value', 'multiply_by', 'llm_generate', 'add_field', 'remove_field'
    value: Any = None
    validation: Optional[Dict[str, Any]] = None
    llm_prompt: Optional[str] = None
    field_config: Optional[Dict[str, Any]] = None  # For add_field operations


@dataclass
class ModificationResult:
    """Result of profile modification operation."""
    agent_id: str
    original_profile: Dict[str, Any]
    modified_profile: Dict[str, Any]
    applied_modifications: List[ModificationSpec]
    schema_changes: Dict[str, Any]  # Record schema modifications
    success: bool
    errors: List[str] = None


class ProfileModifier:
    """
    Simplified profile modifier that works with dictionaries.
    
    This class provides the same functionality as the original but works
    directly with dictionaries, making it much simpler and more suitable
    for experimental modifications.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the ProfileModifier.
        
        Args:
            model_name: Optional model name for LLM-based modifications
        """
        self.model_name = model_name
        self.modification_history: List[ModificationResult] = []
        
    def modify_profiles(
        self,
        profiles: Union[List[Any], List[Dict[str, Any]]],  # Support both AgentProfile objects and dicts
        modifications: List[ModificationSpec],
        target_agents: Optional[List[str]] = None,
        target_percentage: Optional[float] = None,
        selection_strategy: str = "random",
        llm_enhancement: bool = False
    ) -> List[ModificationResult]:
        """
        Modify profiles according to specifications.
        
        Args:
            profiles: List of profiles (AgentProfile objects or dictionaries)
            modifications: List of modification specifications
            target_agents: Specific agent IDs to target (optional)
            target_percentage: Percentage of agents to target (optional)
            selection_strategy: Strategy for selecting agents ('random', 'all')
            llm_enhancement: Whether to use LLM for coherent profile enhancement
            
        Returns:
            List of modification results
        """
        logger.info(f"Starting profile modification for {len(profiles)} profiles")
        
        # Convert AgentProfile objects to dictionaries if needed
        profile_data_list = []
        for profile in profiles:
            if hasattr(profile, 'get_data'):  # AgentProfile object
                profile_dict = self._agent_profile_to_dict(profile)
            else:  # Already a dictionary
                profile_dict = profile
            profile_data_list.append(profile_dict)
        
        # Select target profiles
        selected_profiles = self._select_target_profiles(
            profile_data_list, target_agents, target_percentage, selection_strategy
        )
        
        results = []
        
        for profile_data in selected_profiles:
            try:
                result = self._modify_single_profile(
                    profile_data, modifications, llm_enhancement
                )
                results.append(result)
                self.modification_history.append(result)
                
            except Exception as e:
                agent_id = profile_data.get('id', 'unknown')
                logger.error(f"Failed to modify profile {agent_id}: {e}")
                error_result = ModificationResult(
                    agent_id=agent_id,
                    original_profile=copy.deepcopy(profile_data),
                    modified_profile={},
                    applied_modifications=[],
                    schema_changes={},
                    success=False,
                    errors=[str(e)]
                )
                results.append(error_result)
        
        logger.info(f"Profile modification completed. Success: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def _agent_profile_to_dict(self, profile) -> Dict[str, Any]:
        """Convert AgentProfile object to dictionary."""
        try:
            # Try to get profile data
            if hasattr(profile, 'get_profile'):
                profile_dict = profile.get_profile()
            else:
                profile_dict = {}
            
            # Ensure basic fields
            if 'id' not in profile_dict and hasattr(profile, 'get_data'):
                profile_dict['id'] = profile.get_data('id', 'unknown')
            
            if 'agent_type' not in profile_dict and hasattr(profile, 'agent_type'):
                profile_dict['agent_type'] = profile.agent_type
            
            # Add public and private fields if they exist
            if hasattr(profile, '_public_fields') and profile._public_fields:
                profile_dict.update(profile._public_fields)
            if hasattr(profile, '_private_fields') and profile._private_fields:
                profile_dict.update(profile._private_fields)
                
            return profile_dict
            
        except Exception as e:
            logger.warning(f"Error converting profile to dict: {e}")
            # Fallback to minimal representation
            return {
                'id': getattr(profile, 'id', 'unknown'),
                'agent_type': getattr(profile, 'agent_type', 'unknown'),
                'error': f"Conversion failed: {str(e)}"
            }
    
    def _select_target_profiles(
        self,
        profile_data_list: List[Dict[str, Any]],
        target_agents: Optional[List[str]],
        target_percentage: Optional[float],
        selection_strategy: str
    ) -> List[Dict[str, Any]]:
        """Select target profiles based on selection criteria."""
        
        if target_agents:
            # Select specific agents by ID
            agent_id_map = {p.get('id', ''): p for p in profile_data_list}
            return [agent_id_map[aid] for aid in target_agents if aid in agent_id_map]
        
        elif target_percentage:
            # Select percentage of agents
            import random
            target_count = max(1, int(len(profile_data_list) * target_percentage))
            
            if selection_strategy == "random":
                return random.sample(profile_data_list, target_count)
            else:
                return profile_data_list[:target_count]
        
        else:
            # Select all agents
            return profile_data_list
    
    def _modify_single_profile(
        self,
        profile_data: Dict[str, Any],
        modifications: List[ModificationSpec],
        llm_enhancement: bool
    ) -> ModificationResult:
        """Modify a single profile dictionary."""
        
        agent_id = profile_data.get('id', 'unknown')
        applied_modifications = []
        schema_changes = {}
        errors = []
        
        # Create deep copy for modification
        original_data = copy.deepcopy(profile_data)
        modified_data = copy.deepcopy(profile_data)
        
        # Apply each modification
        for mod_spec in modifications:
            try:
                success = self._apply_modification(modified_data, mod_spec)
                
                if success:
                    applied_modifications.append(mod_spec)
                    
                    # Record schema changes for add_field/remove_field
                    if mod_spec.modification_type == 'add_field':
                        schema_changes[mod_spec.field] = {
                            "operation": "add",
                            "field_config": mod_spec.field_config,
                            "value": mod_spec.value
                        }
                    elif mod_spec.modification_type == 'remove_field':
                        schema_changes[mod_spec.field] = {
                            "operation": "remove",
                            "original_value": original_data.get(mod_spec.field)
                        }
                else:
                    errors.append(f"Failed to apply modification to field: {mod_spec.field}")
            except Exception as e:
                errors.append(f"Error modifying field {mod_spec.field}: {str(e)}")
        
        # Optional LLM enhancement for profile coherence
        if llm_enhancement and self.model_name:
            try:
                self._enhance_profile_coherence(modified_data, applied_modifications)
            except Exception as e:
                logger.warning(f"LLM enhancement failed for agent {agent_id}: {e}")
                errors.append(f"LLM enhancement failed: {str(e)}")
        
        return ModificationResult(
            agent_id=agent_id,
            original_profile=original_data,
            modified_profile=modified_data,
            applied_modifications=applied_modifications,
            schema_changes=schema_changes,
            success=len(errors) == 0,
            errors=errors if errors else None
        )
    
    def _apply_modification(self, profile_data: Dict[str, Any], mod_spec: ModificationSpec) -> bool:
        """Apply a single modification to a profile dictionary."""
        
        try:
            if mod_spec.modification_type == "set_value":
                profile_data[mod_spec.field] = mod_spec.value
                
            elif mod_spec.modification_type == "add_value":
                current_value = profile_data.get(mod_spec.field, 0)
                if isinstance(current_value, (int, float)) and isinstance(mod_spec.value, (int, float)):
                    profile_data[mod_spec.field] = current_value + mod_spec.value
                else:
                    return False
                    
            elif mod_spec.modification_type == "multiply_by":
                current_value = profile_data.get(mod_spec.field, 1)
                if isinstance(current_value, (int, float)) and isinstance(mod_spec.value, (int, float)):
                    profile_data[mod_spec.field] = current_value * mod_spec.value
                else:
                    return False
                    
            elif mod_spec.modification_type == "add_field":
                # Simply add the new field to the profile
                if mod_spec.value is not None:
                    profile_data[mod_spec.field] = mod_spec.value
                elif mod_spec.field_config and 'default' in mod_spec.field_config:
                    profile_data[mod_spec.field] = mod_spec.field_config['default']
                else:
                    profile_data[mod_spec.field] = None
                    
            elif mod_spec.modification_type == "remove_field":
                if mod_spec.field in profile_data:
                    del profile_data[mod_spec.field]
                else:
                    logger.warning(f"Field {mod_spec.field} not found in profile")
                    return False
                    
            elif mod_spec.modification_type == "llm_generate":
                new_value = self._llm_generate_value(profile_data, mod_spec)
                profile_data[mod_spec.field] = new_value
                
            else:
                logger.error(f"Unknown modification type: {mod_spec.modification_type}")
                return False
            
            # Validate the new value if validation rules are provided
            if mod_spec.validation and mod_spec.field in profile_data:
                if not self._validate_value(profile_data[mod_spec.field], mod_spec.validation):
                    logger.error(f"Value validation failed for field {mod_spec.field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying modification: {e}")
            return False
    
    def _validate_value(self, value: Any, validation: Dict[str, Any]) -> bool:
        """Validate a value against validation rules."""
        
        # Type validation
        if 'type' in validation:
            expected_type = validation['type']
            if expected_type == 'float' and not isinstance(value, (int, float)):
                return False
            elif expected_type == 'int' and not isinstance(value, int):
                return False
            elif expected_type == 'str' and not isinstance(value, str):
                return False
        
        # Range validation for numeric values
        if isinstance(value, (int, float)):
            if 'min' in validation and value < validation['min']:
                return False
            if 'max' in validation and value > validation['max']:
                return False
        
        return True
    
    def _llm_generate_value(self, profile_data: Dict[str, Any], mod_spec: ModificationSpec) -> Any:
        """Generate a new value using LLM."""
        
        if not self.model_name or not mod_spec.llm_prompt:
            raise ValueError("LLM generation requires model_name and llm_prompt")
        
        try:
            model = get_model(config_name=self.model_name)
            
            # Create context about the profile
            profile_context = json.dumps(profile_data, indent=2)
            
            # Generate prompt
            prompt = f"""
You are modifying an agent profile for a simulation experiment. 

Current profile context:
{profile_context}

Modification task: {mod_spec.llm_prompt}

Generate a new value for the field '{mod_spec.field}'. 
Ensure the new value is coherent with the agent's existing characteristics.

Respond with only the new value, no additional text.
"""
            
            response = model.generate(prompt, temperature=0.7)
            
            # Try to parse the response based on expected type
            if mod_spec.validation and 'type' in mod_spec.validation:
                value_type = mod_spec.validation['type']
                if value_type == 'float':
                    return float(response.strip())
                elif value_type == 'int':
                    return int(response.strip())
                else:
                    return response.strip()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _enhance_profile_coherence(self, profile_data: Dict[str, Any], modifications: List[ModificationSpec]):
        """Use LLM to enhance profile coherence after modifications."""
        
        if not self.model_name:
            return
        
        try:
            model = get_model(config_name=self.model_name)
            
            modification_summary = [f"{m.field}: {m.value}" for m in modifications]
            
            prompt = f"""
You are helping to maintain coherence in an agent profile after experimental modifications.

Current profile: {json.dumps(profile_data, indent=2)}

Applied modifications: {', '.join(modification_summary)}

Review the profile for coherence and suggest any additional adjustments to maintain 
personality and behavioral consistency. Focus on fields that might need adjustment
to align with the experimental modifications.

Respond with a JSON object containing only the fields that need adjustment:
{{"field_name": "new_value", ...}}

If no adjustments are needed, respond with an empty object: {{}}
"""
            
            response = model.generate(prompt, temperature=0.3)
            
            # Parse and apply suggested adjustments
            try:
                adjustments = json.loads(response.strip())
                for field, value in adjustments.items():
                    if field in profile_data:  # Only adjust existing fields
                        profile_data[field] = value
                        logger.info(f"Applied coherence adjustment: {field} = {value}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM coherence suggestions")
                
        except Exception as e:
            logger.error(f"Profile coherence enhancement failed: {e}")
    
    def save_modified_profiles(
        self,
        results: List[ModificationResult],
        output_dir: Union[str, Path],
        create_originals: bool = True
    ) -> Dict[str, str]:
        """
        Save modified profiles to files.
        
        Args:
            results: List of modification results
            output_dir: Directory to save profiles
            create_originals: Whether to save original profiles for comparison
            
        Returns:
            Dictionary mapping file types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save modified profiles
        modified_profiles = []
        for result in results:
            if result.success:
                modified_profiles.append(result.modified_profile)
        
        if modified_profiles:
            modified_file = output_path / "modified_profiles.json"
            with open(modified_file, 'w', encoding='utf-8') as f:
                json.dump(modified_profiles, f, indent=2, ensure_ascii=False)
            saved_files['modified'] = str(modified_file)
        
        # Save modification log
        modification_log = {
            'total_profiles': len(results),
            'successful_modifications': sum(1 for r in results if r.success),
            'failed_modifications': sum(1 for r in results if not r.success),
            'modifications': [
                {
                    'agent_id': r.agent_id,
                    'success': r.success,
                    'applied_modifications': [
                        {
                            'field': m.field,
                            'type': m.modification_type,
                            'value': m.value
                        } for m in r.applied_modifications
                    ],
                    'errors': r.errors
                } for r in results
            ]
        }
        
        logger.info(f"Saved modified profiles to {output_dir}")
        return saved_files
    
    def create_profile_data_files(
        self,
        results: List[ModificationResult],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Create agent type-specific profile data files compatible with OneSim.
        
        Args:
            results: List of modification results
            output_dir: Directory to save profile data files
            
        Returns:
            Dictionary mapping agent types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Group profiles by agent type
        profiles_by_type = {}
        for result in results:
            if result.success:
                agent_type = result.modified_profile.get('agent_type', 'unknown')
                if agent_type not in profiles_by_type:
                    profiles_by_type[agent_type] = []
                profiles_by_type[agent_type].append(result.modified_profile)
        
        # Save separate files for each agent type
        for agent_type, profiles in profiles_by_type.items():
            type_file = output_path / f"{agent_type}.json"
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False)
            saved_files[agent_type] = str(type_file)
            logger.info(f"Saved {len(profiles)} {agent_type} profiles to {type_file}")
        
        return saved_files
    
    def save_modified_schemas(
        self,
        results: List[ModificationResult],
        base_schemas: Dict[str, Dict[str, Any]],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Save modified schemas for agent types that had schema changes.
        
        Args:
            results: List of modification results
            base_schemas: Dictionary mapping agent types to their base schemas
            output_dir: Directory to save modified schemas
            
        Returns:
            Dictionary mapping agent types to schema file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_schemas = {}
        
        # Collect schema changes by agent type
        schema_changes_by_type = {}
        for result in results:
            if result.success and result.schema_changes:
                agent_type = result.modified_profile.get('agent_type', 'unknown')
                if agent_type not in schema_changes_by_type:
                    schema_changes_by_type[agent_type] = {}
                
                # Merge schema changes for this agent type
                for field, change in result.schema_changes.items():
                    if field not in schema_changes_by_type[agent_type]:
                        schema_changes_by_type[agent_type][field] = change
        
        # Generate modified schemas
        for agent_type, changes in schema_changes_by_type.items():
            if agent_type in base_schemas:
                base_schema = base_schemas[agent_type]
                modified_schema = self._apply_schema_changes(base_schema, changes)
                
                # Save modified schema
                schema_file = output_path / f"{agent_type}.json"
                with open(schema_file, 'w', encoding='utf-8') as f:
                    json.dump(modified_schema, f, indent=2, ensure_ascii=False)
                
                saved_schemas[agent_type] = str(schema_file)
                logger.info(f"Saved modified schema for {agent_type}: {schema_file}")
        
        return saved_schemas
    
    def _apply_schema_changes(self, base_schema: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply schema changes to a base schema."""
        
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