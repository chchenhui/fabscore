# -*- coding: utf-8 -*-
"""A general dialog agent."""
from typing import Optional, Union, Sequence, Any

from loguru import logger

from onesim.models.core.message import Message
from .base import AgentBase
import json
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.relationship.topology_strategies import get_strategy, get_available_strategies
from onesim.utils.relationship_utils import analyze_topology

class ProfileAgent(AgentBase):
    """A simple agent used to perform a dialogue. Your can set its role by
    `sys_prompt`."""

    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str ='',
        **kwargs: Any,
    ) -> None:
        """Initialize the dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`):
                The name of the model config, which is used to load model from
                configuration.
        """
        super().__init__(
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        if sys_prompt == '':
            self.sys_prompt = """You are a helpful AI assistant specialized in analyzing and designing complex systems. Follow instructions carefully and provide accurate, well-structured responses."""
        if kwargs:
            logger.warning(
                f"Unused keyword arguments are provided: {kwargs}",
            )

    def assign_agent_portraits(self, agent_types_dict):
        """Assign portrait types to agent categories.
        
        Args:
            agent_types_dict: Dictionary with agent types as keys and descriptions as values.
            
        Returns:
            Dictionary mapping agent types to portrait IDs where:
            1 = Government Official
            2 = Researcher/Scholar
            3 = Worker/Laborer
            4 = Merchant/Business Person
            5 = Citizen
        """
        prompt = f"""
        You are assigning social role portraits to different agent types in a multi-agent system.
        
        The available portrait types are:
        1. Government Official - Represents authority, policy-making, regulation, and public administration
        2. Researcher/Scholar - Represents academia, expertise, analysis, and knowledge creation
        3. Worker/Laborer - Represents practical skills, manual labor, and production
        4. Merchant/Business Person - Represents commerce, trade, business acumen, and entrepreneurship
        5. Citizen - Represents the general public, consumers, and community members
        
        For the following agent types and their descriptions, assign ONE portrait type (1-5) that best matches each agent's role and characteristics:
        
        {json.dumps(agent_types_dict, indent=2)}
        
        Requirements:
        1. Each agent type should be assigned EXACTLY ONE portrait type (1-5)
        2. Try to distribute portrait types so that not all agents have the same portrait
        3. The assignment should be logical and reflect the agent's role in the system
        4. Return your answer as a JSON object with agent types as keys and portrait IDs (integers 1-5) as values
        
        Example output format:
        ```json
        {{
          "AgentType1": 2,
          "AgentType2": 4,
          "AgentType3": 1
        }}
        ```
        """
        
        parser = JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + parser.content_hint, role="user")
        )
        response = self.model(prompt)
        
        try:
            res = parser.parse(response)
            portrait_assignments = res.parsed
            return portrait_assignments
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def generate_agent_types(self,description):
        
        prompt = (
        f"Given the following description: {description}, identify or infer relevant agent types "
        "and return them as a JSON object where keys are PascalCase agent type names and values are short descriptions of each agent type.\n\n"
        "Requirements:\n"
        "1. Each agent type should be in English and formatted in PascalCase (capitalize each word, with no spaces or special characters).\n"
        "2. Ensure each agent name is concise, clearly reflects its role, and accurately represents the agent's primary function within a multi-agent system.\n"
        "3. Each agent type should include a brief description (1-2 sentences) explaining its role and responsibility.\n"
        "4. If no explicit agent types are present in the description, infer and include plausible agent types to establish a functional multi-agent system based on the described roles and actions.\n\n"
        "Return the agent types as a JSON object surrounded by ```json and ``` with the following format:\n"
        "```json"
        "{\n"
        "  \"AgentType1\": \"Brief description of AgentType1's role\",\n"
        "  \"AgentType2\": \"Brief description of AgentType2's role\"\n"
        "}\n"
        "```"
        )
        parser=JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.content_hint, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:
            
            res=parser.parse(response)
            agent_types = res.parsed
            return agent_types
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")
        

    def merge_relationships(self,relationships):
        """
        Merge and deduplicate relationships.
        """
        # Create a dictionary to store merged relationships
        merged = {}
        for rel in relationships:
            src = rel['source_agent']
            tgt = rel['target_agent']
            rel_type = rel['relationship_type']
            direction = rel['direction']

            key = (src, tgt)
            reciprocal_key = (tgt, src)

            if reciprocal_key in merged:
                existing_rel = merged[reciprocal_key]
                # Determine the new direction
                if existing_rel['direction'] == 'bidirectional' or direction == 'bidirectional':
                    new_direction = 'bidirectional'
                else:
                    new_direction = existing_rel['direction']  # Both unidirectional in the same direction

                # Combine relationship types if necessary
                if existing_rel['relationship_type'] != rel_type:
                    combined_type = f"{existing_rel['relationship_type']} & {rel_type}"
                else:
                    combined_type = existing_rel['relationship_type']

                merged[reciprocal_key] = {
                    'source_agent': tgt,
                    'target_agent': src,
                    'relationship_type': combined_type,
                    'direction': new_direction
                }
            else:
                merged[key] = rel

        # Convert the merged dict back to a list
        merged_relationships = list(merged.values())

        return merged_relationships
    
    def generate_relationship_schema(self, agent_types, actions, events):
        """Generate relationship schema defining what agent types CAN connect based on workflow events.
        
        Args:
            agent_types: List of agent types
            actions: Dictionary of actions
            events: Dictionary of events
            
        Returns:
            List of relationship schema objects defining allowed connections between agent types
        """
        # Extract allowed connections from events graph
        allowed_connections = self._extract_event_constraints(events)
        
        # Generate relationship schema based on workflow constraints
        return self._generate_workflow_relationship_schema(events, allowed_connections)
    
    def _extract_event_constraints(self, events):
        """Extract allowed agent connections from events graph."""
        connections = set()
        for event in events.values():
            from_agent = event.get('from_agent_type')
            to_agent = event.get('to_agent_type')
            
            if from_agent != 'EnvAgent' and to_agent != 'EnvAgent':
                connections.add((from_agent, to_agent))
                connections.add((to_agent, from_agent))  # Bidirectional by default
                
        return connections
    
    def _generate_workflow_relationship_schema(self, events, allowed_connections):
        """Generate relationship schema based on workflow constraints."""
        # Track connections between agent types
        connections = {}
        
        # Analyze events to build connection graph
        for event_id, event in events.items():
            from_agent = event['from_agent_type']
            to_agent = event['to_agent_type']
            
            # Skip environment agent connections
            if from_agent == 'EnvAgent' or to_agent == 'EnvAgent':
                continue
                
            # Initialize connection if not exists
            key = (from_agent, to_agent)
            reverse_key = (to_agent, from_agent)
            
            if key not in connections and reverse_key not in connections:
                connections[key] = {
                    'forward': False,  # Connection from source to target
                    'reverse': False   # Connection from target to source
                }
                
            # Mark connection direction
            if key in connections:
                connections[key]['forward'] = True
            else:
                connections[reverse_key]['reverse'] = True
                
        # Generate relationship schema based on connections
        relationships = []
        for (source, target), connection in connections.items():
            # Determine direction based on connections
            if connection['forward'] and connection['reverse']:
                direction = 'bidirectional'
            else:
                direction = 'unidirectional'
                
            # Create relationship schema object
            relationship = {
                'source_agent': source,
                'target_agent': target,
                'relationship_type': 'workflow_interaction',
                'direction': direction,
                'required': True,  # Required for workflow connectivity
                'constraints': {
                    'min_connections': 1,
                    'max_connections': None  # No limit by default
                }
            }
            relationships.append(relationship)
        
        return relationships

    def generate_topology_config(self, scenario_description, agent_types, relationship_schema):
        """Generate network topology configuration based on scenario and relationship schema.
        
        Args:
            scenario_description: Description of the simulation scenario
            agent_types: List of agent types in the simulation
            relationship_schema: List of allowed relationship types from schema generation
            
        Returns:
            Dictionary containing network topology configuration
        """
        available_strategies = get_available_strategies()
        
        # Build topology descriptions from available strategies
        topology_descriptions = []
        for strategy_name, strategy_class in available_strategies.items():
            description = strategy_class.get_description() if hasattr(strategy_class, 'get_description') else f"{strategy_name} topology"
            params_info = strategy_class.get_parameters_info() if hasattr(strategy_class, 'get_parameters_info') else "Parameters: see documentation"
            topology_descriptions.append(f"**{strategy_name}** - {description}\n   - {params_info}")
        
        topologies_text = "\n\n".join(topology_descriptions)
        
        # Extract allowed connections from relationship schema
        allowed_pairs = [(rel['source_agent'], rel['target_agent']) for rel in relationship_schema]
        
        prompt = f"""
        You are designing a social network topology for a multi-agent simulation. Analyze the scenario description and agent types to determine the most appropriate network structure and its parameters.

        **Scenario Description**: {scenario_description}

        **Agent Types**: {agent_types}
        
        **Allowed Agent Connections** (from workflow analysis): {allowed_pairs}

        **Available Network Topologies**:

        {topologies_text}

        **Parameter Generation Guidelines**:
        - **Grid dimensions**: Consider agent count and spatial context (e.g., 3x3 for 9 agents, 5x4 for 20 agents)
        - **Connection probability**: 0.1-0.3=sparse, 0.4-0.6=moderate, 0.7+=dense based on social closeness
        - **Boundary type**: "open" for bounded spaces, "wrap_around" for cyclical/global connections
        - **Branching factor**: 2-3 for strict hierarchies, 4-5 for broader management structures
        - **Central agent**: Choose most authoritative/coordinating agent type

        **Requirements**:
        1. Analyze the scenario's social dynamics and spatial constraints
        2. Choose the topology that best fits the described interactions
        3. Generate realistic parameters based on context (agent count, social density, spatial needs)
        4. Consider whether agents interact globally, locally, or hierarchically
        5. Only output the topology configuration, do not include any other text or comments

        Analyze the scenario and generate the optimal network configuration:
        **Output Format**:
        ```json
        {{
            "topology": "topology_name",
            "params": {{
                // only include relevant parameters for chosen topology
            }}
        }}
        ```
        """
        
        parser = JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        logger.info(f"Topology config prompt: {prompt}")
        response = self.model(prompt)
        logger.info(f"Topology config response: {response}")
        try:
            res = parser.parse(response)
            network_config = res.parsed
            return network_config
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for topology config, using default")
            return {"topology": "random", "params": {"connection_probability": 0.3}}

    def generate_relationships(self, agent_ids, relationship_schema, topology_config, actions, events, min_relationships=1, max_relationships=3, ensure_connectivity=True):
        """Generate concrete relationships between specific agent instances based on topology configuration.
        
        Args:
            agent_ids: Dictionary mapping agent types to lists of agent IDs
            relationship_schema: List of relationship schema definitions
            topology_config: Network topology configuration
            actions: Dictionary of actions
            events: Dictionary of events  
            min_relationships: Minimum relationships per agent
            max_relationships: Maximum relationships per agent
            ensure_connectivity: Whether to ensure all terminal agents are reachable from start agents
            
        Returns:
            List of relationship objects with source_id, target_id, and direction
        """
        from onesim.utils.relationship_utils import generate_relationships as generate_relationships_util
        from onesim.utils.work_graph import WorkGraph
        
        # Use the utility function to generate relationships with full connectivity guarantees
        relationships = generate_relationships_util(
            relationship_schema=relationship_schema,
            all_agent_ids=agent_ids,
            actions=actions,
            events=events,
            min_relationships=min_relationships,
            max_relationships=max_relationships
        )
        
        # Apply topology-specific relationship generation based on config
        topology_name = topology_config.get('topology', 'random')
        topology_params = topology_config.get('params', {})
        
        # Apply topology strategy to modify/enhance relationships
        relationships = self._apply_topology_to_relationships(
            relationships, agent_ids, topology_name, topology_params
        )
        
        # Additional connectivity verification and fixing if requested
        if ensure_connectivity:
            relationships = self._ensure_complete_connectivity(
                relationships, agent_ids, relationship_schema, actions, events
            )
        
        return relationships
    
    def _apply_topology_to_relationships(self, relationships, agent_ids, topology_name, topology_params):
        """Apply network topology strategy to enhance relationship generation.
        
        Args:
            relationships: Existing relationships from schema-based generation
            agent_ids: Dictionary mapping agent types to lists of agent IDs
            topology_name: Name of topology strategy (e.g., 'star', 'grid', 'complete')
            topology_params: Parameters for topology strategy
            
        Returns:
            Enhanced list of relationships following topology pattern
        """
        try:
            strategy = get_strategy(topology_name)
            if strategy is None:
                logger.warning(f"Unknown topology strategy: {topology_name}, keeping original relationships")
                return relationships
            
            # Create agent type lookup for constraint checking
            agent_type_map = {}
            for agent_type, ids in agent_ids.items():
                for agent_id in ids:
                    agent_type_map[str(agent_id)] = agent_type
            
            # Extract allowed connections from existing relationships (type-level constraints)
            allowed_connections = set()
            for rel in relationships:
                source_type = agent_type_map.get(str(rel['source_id']))
                target_type = agent_type_map.get(str(rel['target_id']))
                if source_type and target_type:
                    allowed_connections.add((source_type, target_type))
                    allowed_connections.add((target_type, source_type))  # Bidirectional
            
            # Generate topology connections using new interface (agent_ids instead of agent_types)
            topology_connections = strategy.generate_connections(agent_ids, allowed_connections, topology_params)
            
            # Convert topology connections to relationship format
            topology_relationships = []
            for source_id, target_id, rel_type in topology_connections:
                topology_relationships.append({
                    'source_id': str(source_id),
                    'target_id': str(target_id),
                    'direction': 'bidirectional'
                })
            
            # Merge with existing relationships, avoiding duplicates
            existing_pairs = {(str(rel['source_id']), str(rel['target_id'])) for rel in relationships}
            for rel in topology_relationships:
                pair = (rel['source_id'], rel['target_id'])
                reverse_pair = (rel['target_id'], rel['source_id'])
                if pair not in existing_pairs and reverse_pair not in existing_pairs:
                    relationships.append(rel)
                    existing_pairs.add(pair)
            
            logger.info(f"Applied {topology_name} topology, total relationships: {len(relationships)}")
            return relationships
            
        except Exception as e:
            logger.error(f"Error applying topology {topology_name}: {e}")
            return relationships
    
    def _ensure_complete_connectivity(self, relationships, agent_ids, relationship_schema, actions, events):
        """Ensure complete connectivity from start agents to terminal agents.
        
        Args:
            relationships: Current list of relationships
            agent_ids: Dictionary mapping agent types to lists of agent IDs
            relationship_schema: List of relationship definitions
            actions: Dictionary of actions
            events: Dictionary of events
            
        Returns:
            Enhanced relationships list with guaranteed connectivity
        """
        from onesim.utils.relationship_utils import ensure_connectivity, verify_connectivity
        from onesim.utils.work_graph import WorkGraph
        
        try:
            # Create work graph to identify start and end agents
            work_graph = WorkGraph()
            work_graph.load_workflow_data(actions, events)
            
            # Convert agent IDs to strings and create lookup
            string_agent_ids = {k: [str(id) for id in v] for k, v in agent_ids.items()}
            agent_type_lookup = {str(agent_id): agent_type 
                                for agent_type, agent_ids_list in string_agent_ids.items() 
                                for agent_id in agent_ids_list}
            
            # Track existing relationships to avoid duplicates
            existing_relationships = set()
            for rel in relationships:
                existing_relationships.add((str(rel['source_id']), str(rel['target_id'])))
            
            # Apply connectivity ensuring
            ensure_connectivity(
                work_graph=work_graph,
                relationships_list=relationships,
                all_agent_ids=string_agent_ids,
                agent_type_lookup=agent_type_lookup,
                relationship_schema=relationship_schema,
                existing_relationships=existing_relationships
            )
            
            # Connectivity verification will be done in validation step
            logger.info("✅ Connectivity verification passed - all terminal agents reachable")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error ensuring connectivity: {e}")
            return relationships
    
    
    def validate_relationship_generation(self, relationship_schema, topology_config, relationships, agent_ids, actions, events):
        """Comprehensive validation of the relationship generation process.
        
        Args:
            relationship_schema: List of relationship schema objects
            topology_config: Network topology configuration
            relationships: Generated relationship instances
            agent_ids: Dictionary mapping agent types to lists of agent IDs
            actions: Dictionary of actions
            events: Dictionary of events
            
        Returns:
            Dictionary with validation results and detailed feedback
        """
        validation_results = {
            'success': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'connectivity_verified': False
        }
        
        try:
            # 1. Validate relationship schema
            schema_validation = self._validate_relationship_schema(relationship_schema, actions, events)
            validation_results['schema_validation'] = schema_validation
            if not schema_validation['valid']:
                validation_results['success'] = False
                validation_results['errors'].extend(schema_validation['errors'])
            
            # 2. Validate topology configuration
            topology_validation = self._validate_topology_config(topology_config, relationship_schema)
            validation_results['topology_validation'] = topology_validation
            if not topology_validation['valid']:
                validation_results['success'] = False
                validation_results['errors'].extend(topology_validation['errors'])
            
            # 3. Validate generated relationships
            relationships_validation = self._validate_generated_relationships(relationships, agent_ids, relationship_schema)
            validation_results['relationships_validation'] = relationships_validation
            if not relationships_validation['valid']:
                validation_results['success'] = False
                validation_results['errors'].extend(relationships_validation['errors'])
            
            # 4. Verify connectivity
            from onesim.utils.relationship_utils import verify_connectivity
            from onesim.utils.work_graph import WorkGraph
            
            try:
                work_graph = WorkGraph()
                work_graph.load_workflow_data(actions, events)
                string_agent_ids = {k: [str(id) for id in v] for k, v in agent_ids.items()}
                agent_type_lookup = {str(agent_id): agent_type 
                                    for agent_type, agent_ids_list in string_agent_ids.items() 
                                    for agent_id in agent_ids_list}
                topology_paths = analyze_topology(work_graph, string_agent_ids)
                
                connectivity_result = verify_connectivity(
                    relationships=relationships,
                    all_agent_ids=string_agent_ids,
                    agent_type_lookup=agent_type_lookup,
                    topology_paths=topology_paths,
                    work_graph=work_graph
                )
                validation_results['connectivity_verified'] = connectivity_result
                if not connectivity_result:
                    validation_results['warnings'].append("Connectivity verification failed - some agents may not be reachable")
            except Exception as e:
                validation_results['connectivity_verified'] = False
                validation_results['warnings'].append(f"Connectivity verification error: {str(e)}")
            
            # 5. Generate statistics
            validation_results['statistics'] = self._generate_relationship_statistics(relationships, agent_ids, relationship_schema, topology_config)
            
            logger.info(f"Relationship validation completed: {'✅ SUCCESS' if validation_results['success'] else '❌ FAILED'}")
            
        except Exception as e:
            validation_results['success'] = False
            validation_results['errors'].append(f"Validation process failed: {str(e)}")
            logger.error(f"Error during relationship validation: {e}")
        
        return validation_results
    
    def _validate_relationship_schema(self, relationship_schema, actions, events):
        """Validate that relationship schema is consistent with workflow."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        if not relationship_schema:
            validation['valid'] = False
            validation['errors'].append("Relationship schema is empty")
            return validation
        
        # Check that all schema relationships are supported by workflow
        workflow_connections = self._extract_event_constraints(events)
        
        for rel in relationship_schema:
            source = rel.get('source_agent')
            target = rel.get('target_agent')
            
            if not source or not target:
                validation['errors'].append(f"Invalid relationship schema entry: missing source or target")
                validation['valid'] = False
                continue
            
            if (source, target) not in workflow_connections and (target, source) not in workflow_connections:
                validation['warnings'].append(f"Relationship {source} -> {target} not found in workflow events")
        
        return validation
    
    def _validate_topology_config(self, topology_config, relationship_schema):
        """Validate topology configuration."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        if not topology_config:
            validation['valid'] = False
            validation['errors'].append("Topology config is empty")
            return validation
        
        topology_name = topology_config.get('topology')
        if not topology_name:
            validation['valid'] = False
            validation['errors'].append("Topology name not specified")
            return validation
        
        # Check if topology strategy exists
        strategy = get_strategy(topology_name)
        if strategy is None:
            validation['valid'] = False
            validation['errors'].append(f"Unknown topology strategy: {topology_name}")
        
        return validation
    
    def _validate_generated_relationships(self, relationships, agent_ids, relationship_schema):
        """Validate generated relationship instances."""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        if not relationships:
            validation['valid'] = False
            validation['errors'].append("No relationships were generated")
            return validation
        
        # Check relationship format
        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                validation['errors'].append(f"Relationship {i} is not a dictionary")
                validation['valid'] = False
                continue
            
            required_fields = ['source_id', 'target_id', 'direction']
            for field in required_fields:
                if field not in rel:
                    validation['errors'].append(f"Relationship {i} missing required field: {field}")
                    validation['valid'] = False
        
        # Check that all agents have minimum connections
        agent_connection_counts = {}
        all_agents = [str(agent_id) for agent_ids_list in agent_ids.values() for agent_id in agent_ids_list]
        
        for agent_id in all_agents:
            agent_connection_counts[agent_id] = 0
        
        for rel in relationships:
            source_id = str(rel.get('source_id', ''))
            target_id = str(rel.get('target_id', ''))
            
            if source_id in agent_connection_counts:
                agent_connection_counts[source_id] += 1
            if target_id in agent_connection_counts:
                agent_connection_counts[target_id] += 1
        
        isolated_agents = [agent_id for agent_id, count in agent_connection_counts.items() if count == 0]
        if isolated_agents:
            validation['warnings'].append(f"{len(isolated_agents)} agents have no connections: {isolated_agents[:5]}")
        
        return validation
    
    def _generate_relationship_statistics(self, relationships, agent_ids, relationship_schema, topology_config):
        """Generate comprehensive statistics about the relationship generation."""
        stats = {
            'total_relationships': len(relationships),
            'total_agents': sum(len(ids) for ids in agent_ids.values()),
            'agent_types': list(agent_ids.keys()),
            'topology_used': topology_config.get('topology', 'unknown'),
            'schema_definitions': len(relationship_schema),
            'relationships_per_agent': {},
            'connectivity_ratio': 0.0
        }
        
        # Calculate relationships per agent
        agent_connection_counts = {}
        all_agents = [str(agent_id) for agent_ids_list in agent_ids.values() for agent_id in agent_ids_list]
        
        for agent_id in all_agents:
            agent_connection_counts[agent_id] = 0
        
        for rel in relationships:
            source_id = str(rel.get('source_id', ''))
            target_id = str(rel.get('target_id', ''))
            
            if source_id in agent_connection_counts:
                agent_connection_counts[source_id] += 1
            if target_id in agent_connection_counts:
                agent_connection_counts[target_id] += 1
        
        connected_agents = sum(1 for count in agent_connection_counts.values() if count > 0)
        stats['connectivity_ratio'] = connected_agents / len(all_agents) if all_agents else 0.0
        
        # Average connections per agent type
        for agent_type, agent_ids_list in agent_ids.items():
            type_connections = [agent_connection_counts.get(str(agent_id), 0) for agent_id in agent_ids_list]
            stats['relationships_per_agent'][agent_type] = {
                'average': sum(type_connections) / len(type_connections) if type_connections else 0,
                'min': min(type_connections) if type_connections else 0,
                'max': max(type_connections) if type_connections else 0
            }
        
        return stats


    def generate_profile_schema(self, scenario_description, agent_name, agent_data_model):
        prompt = """ Please generate a Profile Config Schema for an Agent in a multi-agent scenario, based on the following scenario description, Agent name, and existing data model. The Schema should adhere to the following rules:
    1. **Scenario Description**: {scenario_description}
    2. **Agent Name**: {agent_name}
    3. **Existing Data Model**: {agent_data_model}

    4. **Schema Format**:
    - IMPORTANT: Return a FLAT JSON structure. DO NOT use nested structures with "properties" or "required" fields.
    - IMPORTANT: Each field should be directly at the root level of the JSON object.
    - DO NOT use any JSON Schema validation format. Just provide the direct field definitions.

    5. **Schema Guidelines**:
    - The schema MUST contain a 'name' field as a required property.
    - IMPORTANT: Do NOT include any field with 'id' in its name (such as 'agent_id', 'worker_id', etc.), as unique identifiers will be assigned separately.
    - Incorporate all relevant attributes from the provided data model.
    - Extend the schema beyond the existing data model to create a comprehensive profile.
    - Each property in the Schema should represent either:
        a) A static profile attribute that defines the agent's characteristics
        b) A dynamic runtime variable that changes during simulation
    - Each property should include the following fields:
        - **"type"**: The data type of the property, which can be "int", "str", "list", or "float". ONLY use these four types.
        - **"default"**: The default value of the property (see improved guidelines below).
        - **"private"**: A boolean indicating whether the property is visible to others.
        - **"sampling"**: Specifies how the property's value is generated:
        - "llm": For static attributes needing language model generation
        - "random": For static attributes with defined choices/ranges
        - "default": For dynamic variables that change during simulation
        - **"range"** or **"choices"**: Used to limit possible values for static attributes when sampling is "random"
        - **"sample_size"**: When type is "list", specifies number of items to sample (for static attributes)
        - **"description"**: Required when sampling is "llm"; provides generation guidance

    6. **IMPROVED DEFAULT VALUE GUIDELINES**:
    - For properties with sampling="llm" or sampling="random": 
        - Provide sensible example values that might be generated
    
    - For properties with sampling="default" (dynamic variables):
        - AVOID empty strings, zeros, or empty lists unless absolutely necessary
        - Instead, provide realistic initial values that make sense for the simulation start:
        - For "str" type: Use a meaningful initial state (e.g., "awaiting_input", "ready", "inactive")
        - For "int"/"float" types: Use plausible starting values (e.g., 100 for points, 50 for percentage)
        - For "list" type: Include at least one sample item that demonstrates the expected structure
        - These meaningful defaults will:
        - Make the simulation behave more realistically from the start
        - Provide examples of the expected data structure
        - Avoid null/empty value errors during simulation

    7. **Additional Requirements**:
    - For ALL properties, ensure that default values are relevant to the scenario context
    - The Schema should be output in valid JSON format
    - Only return the JSON Schema, do not include other text/comments
    - CRITICAL: Return a FLAT structure directly at the root level. DO NOT include "properties" or "required" fields.

        **Examples of GOOD vs. POOR Default Values for Dynamic Variables**:

        *POOR (avoid this pattern):*
        ```json
        {{
            "transaction_history": {{
                "type": "list",
                "default": [],
                "private": true,
                "sampling": "default"
            }},
            "current_status": {{
                "type": "str",
                "default": "",
                "private": false,
                "sampling": "default"
            }},
            "customer_satisfaction": {{
                "type": "int",
                "default": 0,
                "private": false,
                "sampling": "default"
            }}
        }}
        ```

        *GOOD (use this pattern):*
        ```json
        {{
            "transaction_history": {{
                "type": "list",
                "default": [{{ "date": "2025-04-01", "amount": 125.50, "type": "initial" }}],
                "private": true,
                "sampling": "default"
            }},
            "current_status": {{
                "type": "str",
                "default": "available",
                "private": false,
                "sampling": "default"
            }},
            "customer_satisfaction": {{
                "type": "int",
                "default": 75,
                "private": false,
                "sampling": "default"
            }}
        }}
        ```

    8. **Expected Output Format**:
    - Your output should look exactly like the example below, with each field directly at the root level:
    
    ```json
    {{
        "name": {{
            "type": "str",
            "default": "John Smith",
            "private": false,
            "sampling": "llm",
            "description": "The agent's full name"
        }},
        "age": {{
            "type": "int",
            "default": 35,
            "private": false,
            "sampling": "random",
            "range": [25, 65]
        }},
        ... other fields directly at root level ...
    }}
    ```

    IMPORTANT: Do NOT add any nested "properties" or "required" fields. Each field should be a direct key-value pair at the root of the JSON object.

    Respond code in a markdown's fenced code block as follows:
    ```json
    Your Profile Config Schema here
    ```
    """.format(scenario_description=scenario_description, agent_name=agent_name,agent_data_model=agent_data_model)
        parser=JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:
            res=parser.parse(response)
            agent_schema = res.parsed
            for key, value in agent_schema.items():
                if value['type']=='array':
                    value['type']='list'
                elif value['type']=='string':
                    value['type']='str'

            agent_schema['id']={"type": "str",
            "default": "0",
            "private": True,
            "sampling": "default",
            "description": "Unique id of agent"}
            return agent_schema
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")
        


    def generate_env_data(self, env_data_schema, description):
        prompt = f""" Transform the provided environment data schema into a realistic JSON object with contextually appropriate values for simulation.

        **Simulation Context**: {description}

        **Input Schema**: {env_data_schema}

        **Transformation Requirements**:
        - Extract each variable from the "variables" array in the input schema
        - For each variable:
          - Use the "name" field as the key in the output
          - For all cases, generate a REALISTIC, SPECIFIC value based on:
            - The simulation context described above
            - The variable's name and type
            - Real-world plausibility for the scenario

        **Value Generation Guidelines**:
        - **"list"**: Generate 3-5 diverse, specific items relevant to the context
        - **"str"**: Create descriptive, specific strings (avoid generic terms like "value" or "item")
        - **"int"**: Use realistic numbers within expected ranges for the domain (avoid 0, 1, or round numbers)
        - **"float"**: Use precise decimal values with appropriate precision for the context
        - **"bool"**: Choose true/false based on what makes sense in the scenario

        **IMPORTANT**: 
        - NEVER use null, empty strings, zeros, or placeholder values
        - Make values varied and realistic as if from a real-world dataset
        - Ensure all generated values directly relate to the simulation context
        - Consider relationships between variables (e.g., if generating inventory and demand, make them proportional)
        
        **Examples of Good vs Poor Value Generation**:

        For a retail simulation context:
        
        POOR:
        ```json
        {{
            "product_price": "0",
            "customer_demand": "100",
            "store_location": "Location"
        }}
        ```
        
        GOOD:
        ```json
        {{
            "product_price": "24.99",
            "customer_demand": "843",
            "store_location": "Downtown Shopping Center"
        }}
        ```
        
        Return ONLY the transformed key-value JSON object without any additional explanations or comments.
        """

        parser = JsonBlockParser()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + parser.format_instruction, role="user")
        )
        response = self.model(prompt)
        # Parse the LLM's JSON response
        try:
            res = parser.parse(response)
            initialized_data = res.parsed
            return initialized_data
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")