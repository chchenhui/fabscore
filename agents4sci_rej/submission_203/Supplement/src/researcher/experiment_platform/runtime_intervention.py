#!/usr/bin/env python3
"""
Runtime Intervention System for Experimental Platform

This module provides functionality for runtime interventions during simulation,
including agent attribute modifications, environment changes, and message broadcasting.
"""

import json

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class InterventionTarget(Enum):
    """Types of intervention targets."""
    AGENT = "agent"
    ENVIRONMENT = "environment"
    BROADCAST = "broadcast"


class InterventionTiming(Enum):
    """Types of intervention timing."""
    SCHEDULED = "scheduled"  # Execute at specific time steps
    CONDITIONAL = "conditional"  # Execute when conditions are met


@dataclass
class RuntimeInterventionSpec:
    """Specification for a runtime intervention."""
    intervention_id: str
    description: str
    target_type: InterventionTarget
    timing: InterventionTiming
    timing_config: Dict[str, Any]
    intervention_config: Dict[str, Any]
    target_selection: Optional[Dict[str, Any]] = None


class RuntimeInterventionEngine:
    """
    Parses and schedules runtime interventions for simulation execution.
    
    This class only handles configuration parsing and scheduling.
    Actual intervention execution is delegated to the simulation environment.
    """
    
    def __init__(self):
        """Initialize the RuntimeInterventionEngine."""
        self.interventions: Dict[str, RuntimeInterventionSpec] = {}
        self.scheduled_interventions: List[Dict[str, Any]] = []
        self.conditional_interventions: List[Dict[str, Any]] = []
        
    def load_runtime_interventions(self, spec_file: Union[str, Path]) -> None:
        """
        Load runtime intervention specifications from JSON file.
        
        Args:
            spec_file: Path to runtime intervention specifications file
        """
        spec_path = Path(spec_file)
        if not spec_path.exists():
            raise FileNotFoundError(f"Runtime intervention specification file not found: {spec_path}")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec_data = json.load(f)
        
        self._parse_runtime_specs(spec_data)
        logger.info(f"Loaded {len(self.interventions)} runtime intervention specifications")
    
    def _parse_runtime_specs(self, spec_data: Dict[str, Any]) -> None:
        """Parse runtime intervention specifications from loaded JSON data."""
        
        # Handle new format: treatment_groups with runtime_modifications
        if 'treatment_groups' in spec_data:
            for group_id, group_data in spec_data['treatment_groups'].items():
                runtime_modifications = group_data.get('runtime_modifications', [])
                for i, intervention_data in enumerate(runtime_modifications):
                    intervention_id = f"{group_id}_runtime_{i}"
                    self._parse_single_runtime_intervention(intervention_id, intervention_data)
        
        # Handle legacy format: direct runtime_interventions
        elif 'runtime_interventions' in spec_data:
            runtime_specs = spec_data['runtime_interventions']
            
            # Handle both dict and list formats
            if isinstance(runtime_specs, dict):
                # Old format: dict with intervention_id as keys
                for intervention_id, intervention_data in runtime_specs.items():
                    self._parse_single_runtime_intervention(intervention_id, intervention_data)
            elif isinstance(runtime_specs, list):
                # New format: array with index-based IDs
                for i, intervention_data in enumerate(runtime_specs):
                    intervention_id = f"intervention_runtime_{i}"
                    self._parse_single_runtime_intervention(intervention_id, intervention_data)
            else:
                logger.error("Invalid runtime_interventions format: must be dict or list")
        else:
            logger.warning("No runtime intervention specifications found in file")
    
    def _parse_single_runtime_intervention(self, intervention_id: str, intervention_data: Dict[str, Any]) -> None:
        """Parse a single runtime intervention."""
        try:
            # Parse intervention type and target type
            intervention_type = intervention_data.get('intervention_type', 'attribute_modification')
            target_type_str = intervention_data.get('target_type', 'agent')
            
            # Map intervention_type to our target enum
            if intervention_type == 'broadcast':
                target_type = InterventionTarget.BROADCAST
            elif target_type_str == 'environment':
                target_type = InterventionTarget.ENVIRONMENT
            else:
                target_type = InterventionTarget.AGENT
            
            # Parse timing
            timing_config = intervention_data.get('timing', {})
            timing_type = InterventionTiming(timing_config.get('type', 'scheduled'))
            
            # Create intervention spec
            intervention_spec = RuntimeInterventionSpec(
                intervention_id=intervention_id,
                description=intervention_data.get('description', f'Runtime intervention {intervention_id}'),
                target_type=target_type,
                timing=timing_type,
                timing_config=timing_config,
                intervention_config=intervention_data.get('config', {}),
                target_selection=intervention_data.get('target_selection')
            )
            
            self.interventions[intervention_id] = intervention_spec
            
            # Schedule interventions based on timing
            if timing_type == InterventionTiming.SCHEDULED:
                self._schedule_intervention(intervention_spec)
            elif timing_type == InterventionTiming.CONDITIONAL:
                self._register_conditional_intervention(intervention_spec)
            
        except Exception as e:
            logger.error(f"Error parsing runtime intervention {intervention_id}: {e}")
    
    def _schedule_intervention(self, intervention_spec: RuntimeInterventionSpec) -> None:
        """Schedule an intervention for specific time steps."""
        
        timing_config = intervention_spec.timing_config
        schedule = timing_config.get('schedule', [])
        
        for time_step in schedule:
            self.scheduled_interventions.append({
                'intervention_id': intervention_spec.intervention_id,
                'time_step': time_step,
                'spec': intervention_spec
            })
        
        logger.info(f"Scheduled intervention {intervention_spec.intervention_id} for steps: {schedule}")
    
    def _register_conditional_intervention(self, intervention_spec: RuntimeInterventionSpec) -> None:
        """Register a conditional intervention."""
        
        self.conditional_interventions.append({
            'intervention_id': intervention_spec.intervention_id,
            'spec': intervention_spec
        })
        
        logger.info(f"Registered conditional intervention {intervention_spec.intervention_id}")
    
    def get_scheduled_interventions(self, current_time_step: int) -> List[Dict[str, Any]]:
        """
        Get interventions scheduled for the current time step.
        
        Args:
            current_time_step: Current simulation time step
            
        Returns:
            List of intervention specifications scheduled for this step
        """
        due_interventions = [
            item for item in self.scheduled_interventions 
            if item['time_step'] == current_time_step
        ]
        
        logger.info(f"Found {len(due_interventions)} scheduled interventions for step {current_time_step}")
        return due_interventions
    
    def get_conditional_interventions(
        self,
        simulation_context: Dict[str, Any],
        current_time_step: int
    ) -> List[Dict[str, Any]]:
        """
        Get conditional interventions whose conditions are met.
        
        Args:
            simulation_context: Current simulation context
            current_time_step: Current simulation time step
            
        Returns:
            List of intervention specifications whose conditions are met
        """
        due_interventions = []
        
        for intervention_item in self.conditional_interventions:
            intervention_spec = intervention_item['spec']
            
            try:
                if self._check_intervention_condition(intervention_spec, simulation_context):
                    due_interventions.append(intervention_item)
                    
            except Exception as e:
                logger.error(f"Error checking conditional intervention {intervention_spec.intervention_id}: {e}")
        
        logger.info(f"Found {len(due_interventions)} conditional interventions ready for step {current_time_step}")
        return due_interventions
    
    def _check_intervention_condition(
        self,
        intervention_spec: RuntimeInterventionSpec,
        simulation_context: Dict[str, Any]
    ) -> bool:
        """Check if conditions for a conditional intervention are met."""
        
        timing_config = intervention_spec.timing_config
        condition = timing_config.get('condition', '')
        
        if not condition:
            return False
        
        try:
            # Get context variables
            agents_dict = simulation_context.get('agents', {})
            agent_count = sum(len(agent_dict) for agent_dict in agents_dict.values()) if agents_dict else 0
            
            context_vars = {
                'step': simulation_context.get('current_step', 0),
                'agent_count': agent_count,
                'environment': simulation_context.get('environment', {}),
                'metrics': simulation_context.get('metrics', {})
            }
            
            # Evaluate condition (basic implementation)
            return eval(condition, {"__builtins__": {}}, context_vars)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    def get_all_interventions(self) -> Dict[str, RuntimeInterventionSpec]:
        """Get all loaded intervention specifications."""
        return self.interventions.copy()
    
    def get_intervention_spec(self, intervention_id: str) -> Optional[RuntimeInterventionSpec]:
        """Get a specific intervention specification by ID."""
        return self.interventions.get(intervention_id)
    
    async def execute_intervention(self, intervention_spec: RuntimeInterventionSpec, environment):
        """Execute a single intervention using environment methods"""
        try:
            if intervention_spec.target_type == InterventionTarget.AGENT:
                await self._execute_agent_intervention(intervention_spec, environment)
            elif intervention_spec.target_type == InterventionTarget.ENVIRONMENT:
                await self._execute_environment_intervention(intervention_spec, environment)
            elif intervention_spec.target_type == InterventionTarget.BROADCAST:
                await self._execute_broadcast_intervention(intervention_spec, environment)
            else:
                logger.warning(f"Unknown intervention target type: {intervention_spec.target_type}")
                
        except Exception as e:
            logger.error(f"Error executing intervention {intervention_spec.intervention_id}: {e}")
    
    async def _execute_agent_intervention(self, intervention_spec: RuntimeInterventionSpec, environment):
        """Execute agent-targeted intervention using environment's agent management"""
        config = intervention_spec.intervention_config
        target_agents = self._select_target_agents(intervention_spec.target_selection or {}, environment)
        
        modifications = config.get('modifications', [])
        for modification in modifications:
            field = modification.get('field')
            value = modification.get('value')
            
            for agent in target_agents:
                agent_id = getattr(agent, 'id', 'unknown')
                try:
                    # Use environment's async update_agent_data method
                    success = await environment.update_agent_data(agent_id, field, value)
                    if success:
                        logger.info(f"Updated agent {agent_id} field {field} = {value}")
                    else:
                        logger.warning(f"Failed to update agent {agent_id} field {field}")
                except Exception as e:
                    logger.error(f"Error modifying agent {agent_id} field {field}: {e}")
    
    async def _execute_environment_intervention(self, intervention_spec: RuntimeInterventionSpec, environment):
        """Execute environment-targeted intervention using environment's data management"""
        config = intervention_spec.intervention_config
        
        modifications = config.get('modifications', [])
        for modification in modifications:
            field = modification.get('field')
            value = modification.get('value')
            
            try:
                # Use environment's async update_data method
                await environment.update_data(field, value)
                logger.info(f"Updated environment field {field} = {value}")
            except Exception as e:
                logger.error(f"Error modifying environment field {field}: {e}")
    
    async def _execute_broadcast_intervention(self, intervention_spec: RuntimeInterventionSpec, environment):
        """Execute broadcast intervention by adding memory to target agents"""
        config = intervention_spec.intervention_config
        target_agents = self._select_target_agents(intervention_spec.target_selection or {}, environment)
        
        
        for agent in target_agents:
            agent_id = getattr(agent, 'id', 'unknown')
            try:
                await agent.add_memory(config.get('message', ''))
            except Exception as e:
                logger.error(f"Error adding memory to agent {agent_id}: {e}")
        logger.info(f"Added broadcast memory to {len(target_agents)} agents")    

    def _select_target_agents(self, target_selection: Dict[str, Any], environment):
        """Select target agents based on selection criteria"""
        if not environment.agents:
            return []
        
        # Flatten all agents into a single list
        all_agents = []
        for agents_dict in environment.agents.values():
            all_agents.extend(agents_dict.values())
        
        method = target_selection.get('method', 'all')
        
        if method == 'all':
            return all_agents
        elif method == 'by_agent_type':
            target_types = target_selection.get('agent_types', [])
            return [agent for agent in all_agents if getattr(agent, 'agent_type', '') in target_types]
        elif method == 'random_sample':
            import random
            percentage = target_selection.get('percentage', 1.0)
            sample_size = max(1, int(len(all_agents) * percentage))
            return random.sample(all_agents, min(sample_size, len(all_agents)))
        elif method == 'by_profile_criteria':
            criteria = target_selection.get('criteria', {})
            return self._select_agents_by_criteria(all_agents, criteria)
        elif method == 'environment_global':
            return []  # Environment interventions don't target agents
        else:
            logger.warning(f"Unknown selection method: {method}, using all agents")
            return all_agents
    
    def _select_agents_by_criteria(self, agents, criteria: Dict[str, Any]):
        """Select agents based on profile criteria"""
        selected = []
        for agent in agents:
            try:
                profile = agent.get_profile() if hasattr(agent, 'get_profile') else None
                if profile and self._matches_criteria(profile, criteria):
                    selected.append(agent)
            except:
                continue
        return selected
    
    def _matches_criteria(self, profile, criteria: Dict[str, Any]) -> bool:
        """Check if profile matches all criteria"""
        for field, condition in criteria.items():
            field_value = getattr(profile, field, None)
            condition_type = condition.get('type')
            condition_value = condition.get('value')
            
            if condition_type == 'equals' and field_value != condition_value:
                return False
            elif condition_type == 'in_list' and field_value not in condition_value:
                return False
            # Add more criteria types as needed
        return True