#!/usr/bin/env python3
"""
Intervention Engine for Experimental Platform

This module provides functionality to parse intervention specifications
and coordinate profile modifications for controlled experiments.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from onesim.profile import AgentProfile
from .profile_modifier import ProfileModifier, ModificationSpec, ModificationResult
from loguru import logger


@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    intervention_id: str
    intervention_type: str
    target_agents: List[str]
    modification_results: List[ModificationResult]
    success: bool
    summary: Dict[str, Any]


class InterventionEngine:
    """
    Manages and executes experimental interventions.
    
    This class handles parsing intervention specifications, selecting target
    agents, and coordinating profile modifications through the ProfileModifier.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the InterventionEngine.
        
        Args:
            model_name: Optional model name for LLM-based modifications
        """
        self.model_name = model_name
        self.profile_modifier = ProfileModifier(model_name)
        self.target_strategies: Dict[str, Dict] = {}
        self.treatment_group_data: Dict[str, Any] = {}
        
    def load_intervention_specifications(self, spec_file: Union[str, Path]) -> None:
        """
        Load intervention specifications from JSON file.
        
        Args:
            spec_file: Path to intervention specifications file
        """
        spec_path = Path(spec_file)
        if not spec_path.exists():
            raise FileNotFoundError(f"Intervention specification file not found: {spec_path}")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec_data = json.load(f)
        
        self._parse_specs(spec_data)
        logger.info(f"Loaded intervention specifications for {len(self.treatment_group_data)} treatment groups.")
    
    def _parse_specs(self, spec_data: Dict[str, Any]) -> None:
        """Parse intervention specifications from loaded JSON data."""
        
        # Parse target selection strategies, which are defined globally
        if 'target_selection_strategies' in spec_data:
            self.target_strategies = spec_data['target_selection_strategies']
        
        # In the new design, we work directly with the treatment_groups structure
        if 'treatment_groups' in spec_data:
            self.treatment_group_data = spec_data['treatment_groups']
        else:
            raise ValueError("Invalid specification file: 'treatment_groups' key is missing.")
    
    def apply_intervention(
        self,
        treatment_group_name: str,
        profiles: List[AgentProfile],
        target_selection: Dict[str, Any], # Note: this is unused in the new flow
        llm_enhancement: bool = False
    ) -> InterventionResult:
        """
        Apply all profile modifications for a given treatment group.
        
        Args:
            treatment_group_name: The treatment group name (e.g., 'treatment_group_001').
            profiles: List of all agent profiles for the simulation.
            target_selection: Unused, kept for signature compatibility. Targeting is now
                              defined within each modification block in the spec file.
            llm_enhancement: Whether to use LLM for coherent profile enhancement.
            
        Returns:
            A single InterventionResult aggregating all modifications for the group.
            Individual modifications within the group get intervention_profile_index IDs.
        """
        if treatment_group_name not in self.treatment_group_data:
            raise ValueError(f"Unknown treatment group name: {treatment_group_name}")

        treatment_group = self.treatment_group_data[treatment_group_name]
        profile_modifications = treatment_group.get('profile_modifications', [])

        if not profile_modifications:
            logger.warning(f"No 'profile_modifications' found for treatment group '{treatment_group_name}'. No action taken.")
            return InterventionResult(
                intervention_id=treatment_group_name,
                intervention_type='profile_modification',
                target_agents=[],
                modification_results=[],
                success=True,
                summary={'message': 'No profile modifications specified.'}
            )

        all_modification_results: List[ModificationResult] = []
        all_target_agent_ids = set()

        # Each 'profile_modification' block is a separate action with its own target and modifications
        for i, mod_block in enumerate(profile_modifications):
            # Generate standardized intervention ID for this profile modification block
            profile_intervention_id = f"intervention_profile_{i}"
            
            block_target_selection = mod_block.get('target_selection')
            if not block_target_selection:
                logger.error(f"Skipping profile modification {profile_intervention_id} in treatment group '{treatment_group_name}' due to missing 'target_selection'.")
                continue

            # 1. Select target agents for this specific block
            _, target_profiles = self._select_target_agents(profiles, block_target_selection)
            target_agent_ids = {p.get_data('id') for p in target_profiles}
            all_target_agent_ids.update(target_agent_ids)
            
            logger.info(f"Profile modification {profile_intervention_id} in treatment group '{treatment_group_name}': Found {len(target_profiles)} target agents for modification.")

            # 2. Create ModificationSpec objects for this block's modifications
            mod_specs = []
            for j, spec_data in enumerate(mod_block.get('modifications', [])):
                modification_id = f"{profile_intervention_id}_mod_{j}"
                logger.debug(f"Creating modification {modification_id}: {spec_data['field']} -> {spec_data['modification_type']}")
                
                mod_specs.append(ModificationSpec(
                    field=spec_data['field'],
                    modification_type=spec_data['modification_type'],
                    value=spec_data.get('value'),
                    validation=spec_data.get('validation'),
                    field_config=spec_data.get('field_config'),  # Crucial for 'add_field'
                    llm_prompt=mod_block.get('llm_prompt_template')
                ))

            # 3. Apply the modifications to the selected agents
            logger.info(f"Applying {len(mod_specs)} modifications for {profile_intervention_id}")
            modification_results = self.profile_modifier.modify_profiles(
                profiles=target_profiles,
                modifications=mod_specs,
                selection_strategy='all', # Apply to all profiles passed in
                llm_enhancement=llm_enhancement
            )
            
            # Add intervention ID to each modification result for tracking
            for result in modification_results:
                if hasattr(result, 'intervention_id'):
                    result.intervention_id = profile_intervention_id
                elif hasattr(result, '__dict__'):
                    result.__dict__['intervention_id'] = profile_intervention_id
            
            all_modification_results.extend(modification_results)

        total_success = all(res.success for res in all_modification_results)
        
        # Create detailed summary with individual intervention IDs
        profile_interventions_summary = {}
        for i in range(len(profile_modifications)):
            profile_intervention_id = f"intervention_profile_{i}"
            relevant_results = [r for r in all_modification_results if getattr(r, 'intervention_id', None) == profile_intervention_id]
            profile_interventions_summary[profile_intervention_id] = {
                'modifications_count': len(relevant_results),
                'success_count': len([r for r in relevant_results if r.success])
            }
        
        return InterventionResult(
            intervention_id=treatment_group_name,  # Keep treatment group name as main ID
            intervention_type='profile_modification',
            target_agents=list(all_target_agent_ids),
            modification_results=all_modification_results,
            success=total_success,
            summary={
                'total_agents_targeted': len(all_target_agent_ids),
                'total_modifications_applied': len(all_modification_results),
                'profile_interventions': profile_interventions_summary
            }
        )

    def _select_target_agents(
        self,
        profiles: List[AgentProfile],
        target_selection: Dict[str, Any]
    ) -> Tuple[List[str], List[AgentProfile]]:
        """
        Select target agents based on a variety of selection strategies.
        
        Args:
            profiles: List of all agent profiles
            target_selection: The selection criteria dictionary from the spec
            
        Returns:
            A tuple containing a list of selected agent IDs and a list of selected agent profiles.
        """
        selection_method = target_selection.get('method')
        if not selection_method:
            raise ValueError("Target selection 'method' not specified.")

        selected_profiles = []
        
        if selection_method == 'all':
            selected_profiles = profiles
        
        elif selection_method == 'specific_agents':
            target_ids = set(target_selection.get('agent_ids', []))
            selected_profiles = [p for p in profiles if p.get_data('id') in target_ids]
            
        elif selection_method == 'by_agent_type':
            target_types = set(target_selection.get('agent_types', []))
            candidate_profiles = [p for p in profiles if p.agent_type in target_types]
            
            percentage = target_selection.get('percentage', 1.0)
            target_count = int(len(candidate_profiles) * percentage)
            
            selected_profiles = random.sample(candidate_profiles, target_count)
            
        elif selection_method == 'random_sample':
            percentage = target_selection.get('percentage')
            if percentage is None:
                raise ValueError("random_sample requires 'percentage'.")
                
            target_count = int(len(profiles) * percentage)
            selected_profiles = random.sample(profiles, target_count)
            
        elif selection_method == 'by_profile_criteria':
            criteria = target_selection.get('criteria')
            if not criteria:
                raise ValueError("by_profile_criteria requires 'criteria'.")
                
            candidate_profiles = self._filter_by_profile_criteria(profiles, criteria)
            
            percentage = target_selection.get('percentage', 1.0)
            target_count = int(len(candidate_profiles) * percentage)
            
            selected_profiles = random.sample(candidate_profiles, target_count)
            
        else:
            # Placeholder for future strategies like network-based selection
            logger.warning(f"Unsupported selection method: {selection_method}")

        selected_agent_ids = [p.get_data('id', 'unknown_id') for p in selected_profiles]
        logger.info(f"Selected {len(selected_profiles)} agents using method '{selection_method}'.")
        return selected_agent_ids, selected_profiles
    
    def _filter_by_profile_criteria(self, profiles: List[AgentProfile], criteria: Dict[str, Any]) -> List[AgentProfile]:
        """Filter profiles that match all specified criteria."""
        
        filtered_profiles = []
        for profile in profiles:
            matches_all = True
            for field, condition in criteria.items():
                # Traverse nested fields if necessary (e.g., 'personality.openness')
                field_path = field.split('.')
                try:
                    value = profile.get_profile()
                    for key in field_path:
                        value = value[key]
                except (KeyError, TypeError):
                    value = None # Field does not exist in this profile
                
                if not self._evaluate_condition(value, condition):
                    matches_all = False
                    break
            
            if matches_all:
                filtered_profiles.append(profile)
                
        return filtered_profiles

    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate if a value satisfies a given condition."""
        
        cond_type = condition.get('type')
        cond_value = condition.get('value')

        if value is None:
            return False

        if cond_type == 'equals':
            return value == cond_value
        elif cond_type == 'not_equals':
            return value != cond_value
        elif cond_type in ('greater_than', 'gt'):
            return isinstance(value, (int, float)) and value > cond_value
        elif cond_type in ('less_than', 'lt'):
            return isinstance(value, (int, float)) and value < cond_value
        elif cond_type in ('greater_equal', 'gte'):
            return isinstance(value, (int, float)) and value >= cond_value
        elif cond_type in ('less_equal', 'lte'):
            return isinstance(value, (int, float)) and value <= cond_value
        elif cond_type == 'in_range':
            return isinstance(value, (int, float)) and cond_value[0] <= value <= cond_value[1]
        elif cond_type == 'contains':
            return isinstance(value, str) and cond_value in value
        elif cond_type == 'in_list':
            return value in cond_value
        elif cond_type == 'regex':
            import re
            return isinstance(value, str) and re.match(cond_value, value)
        
        return False

    def apply_multiple_interventions(
        self,
        intervention_configs: List[Dict[str, Any]],
        profiles: List[AgentProfile],
        llm_enhancement: bool = False
    ) -> List[InterventionResult]:
        """
        Apply multiple interventions sequentially.
        
        Args:
            intervention_configs: List of intervention configurations to apply
            profiles: List of agent profiles
            llm_enhancement: Whether to use LLM enhancement
            
        Returns:
            List of InterventionResult objects
        """
        results = []
        for config in intervention_configs:
            treatment_group_name = config['treatment_group_name']  # Use consistent naming
            # Target selection is now handled within apply_intervention
            target_selection = config.get('target_selection', {}) 
            
            result = self.apply_intervention(
                treatment_group_name,
                profiles,
                target_selection,
                llm_enhancement
            )
            results.append(result)
            
        return results
    
    def get_available_interventions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available interventions, keyed by treatment group ID.
        """
        return self.treatment_group_data
    
    def get_all_treatment_group_data(self) -> Dict[str, Any]:
        """Returns all loaded treatment group data from the intervention specifications."""
        return self.treatment_group_data

    def get_target_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available target selection strategies.
        """
        return self.target_strategies

    def create_intervention_summary(
        self,
        results: List[InterventionResult],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary of intervention results.
        
        Args:
            results: List of intervention results
            output_file: Optional file to save the summary
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_interventions': len(results),
            'successful_interventions': sum(1 for r in results if r.success),
            'failed_interventions': sum(1 for r in results if not r.success),
            'intervention_details': []
        }
        
        for result in results:
            detail = {
                'treatment_group_name': result.intervention_id,  # This is actually treatment group name
                'intervention_type': result.intervention_type,
                'success': result.success,
                'target_count': len(result.target_agents),
                'modification_summary': result.summary
            }
            summary['intervention_details'].append(detail)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Intervention summary saved to {output_file}")
        
        return summary