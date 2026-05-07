# -*- coding: utf-8 -*-
"""Network topology strategies for social relationship generation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Set
import random
import math
from loguru import logger


class NetworkTopologyStrategy(ABC):
    """Abstract base class for network topology generation strategies."""
    
    @abstractmethod
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate connections based on topology strategy.
        
        Args:
            agent_ids: Dictionary mapping agent types to lists of agent IDs
            allowed_connections: Set of allowed (source_type, target_type) pairs from events
            params: Strategy-specific parameters
            
        Returns:
            List of (source_agent_id, target_agent_id, relationship_type) tuples
        """
        pass


class RandomTopologyStrategy(NetworkTopologyStrategy):
    """Generate random connections with specified probability."""
    
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate random relationships between agent instances."""
        connection_probability = params.get('connection_probability', 0.3)
        
        connections = []
        
        # Create agent type lookup for checking constraints
        agent_type_lookup = {}
        all_agent_instances = []
        
        for agent_type, ids in agent_ids.items():
            for agent_id in ids:
                agent_id_str = str(agent_id)
                agent_type_lookup[agent_id_str] = agent_type
                all_agent_instances.append(agent_id_str)
        
        # Generate random connections between all agent pairs
        for i, agent1_id in enumerate(all_agent_instances):
            for agent2_id in all_agent_instances[i+1:]:
                agent1_type = agent_type_lookup[agent1_id]
                agent2_type = agent_type_lookup[agent2_id]
                
                # Check event constraints (type-level)
                if ((agent1_type, agent2_type) not in allowed_connections and 
                    (agent2_type, agent1_type) not in allowed_connections):
                    continue
                
                # Random connection decision
                if random.random() < connection_probability:
                    connections.append((agent1_id, agent2_id, 'random_connection'))
        
        return connections


class GridTopologyStrategy(NetworkTopologyStrategy):
    """Generate N×N grid topology where agents connect only to adjacent neighbors."""
    
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate grid-based relationships between agent instances."""
        dimensions = params.get('dimensions', [3, 3])
        boundary_type = params.get('boundary_type', 'open')
        diagonal_connections = params.get('diagonal_connections', False)
        
        # Create agent type lookup and flatten all agents
        agent_type_lookup = {}
        all_agent_instances = []
        
        for agent_type, ids in agent_ids.items():
            for agent_id in ids:
                agent_id_str = str(agent_id)
                agent_type_lookup[agent_id_str] = agent_type
                all_agent_instances.append(agent_id_str)
        
        total_agents = len(all_agent_instances)
        
        # Determine grid dimensions
        rows, cols = dimensions
        total_grid_positions = rows * cols
        
        if total_agents > total_grid_positions:
            # Find minimum grid size that can accommodate all agents
            min_dimension = math.ceil(math.sqrt(total_agents))
            rows = cols = min_dimension
            total_grid_positions = rows * cols
            logger.info(f"Expanded grid to {rows}×{cols} to accommodate {total_agents} agents")
        elif total_agents < total_grid_positions:
            # Use existing grid size but only fill with available agents
            logger.info(f"Using {rows}×{cols} grid with {total_agents} agents (some positions empty)")
        
        # Use only the available agents (no duplication)
        grid_agents = all_agent_instances
        
        connections = []
        
        # Generate grid connections
        for i in range(rows):
            for j in range(cols):
                current_idx = i * cols + j
                if current_idx >= len(grid_agents):
                    # No more agents to place
                    break
                    
                current_agent_id = grid_agents[current_idx]
                current_agent_type = agent_type_lookup[current_agent_id]
                
                # Define neighbor offsets
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if diagonal_connections:
                    neighbors.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
                
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    # Handle boundary conditions
                    if boundary_type == 'wrap_around':
                        ni = ni % rows
                        nj = nj % cols
                    elif boundary_type == 'open':
                        if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                            continue
                    
                    neighbor_idx = ni * cols + nj
                    if 0 <= neighbor_idx < len(grid_agents):
                        neighbor_agent_id = grid_agents[neighbor_idx]
                        neighbor_agent_type = agent_type_lookup[neighbor_agent_id]
                        
                        # Check event constraints (type-level)
                        if ((current_agent_type, neighbor_agent_type) not in allowed_connections and 
                            (neighbor_agent_type, current_agent_type) not in allowed_connections):
                            continue
                        
                        # Avoid duplicate connections
                        reverse_exists = any(
                            conn[0] == neighbor_agent_id and conn[1] == current_agent_id
                            for conn in connections
                        )
                        
                        if not reverse_exists and current_agent_id != neighbor_agent_id:
                            connections.append((current_agent_id, neighbor_agent_id, 'spatial_adjacency'))
        
        return connections


class CompleteTopologyStrategy(NetworkTopologyStrategy):
    """Generate complete graph where every agent connects to every other agent."""
    
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate complete graph relationships between all agent instances."""
        connections = []
        
        # Create agent type lookup and flatten all agents
        agent_type_lookup = {}
        all_agent_instances = []
        
        for agent_type, ids in agent_ids.items():
            for agent_id in ids:
                agent_id_str = str(agent_id)
                agent_type_lookup[agent_id_str] = agent_type
                all_agent_instances.append(agent_id_str)
        
        # Generate complete graph: connect every agent to every other agent
        for i, agent1_id in enumerate(all_agent_instances):
            for agent2_id in all_agent_instances[i+1:]:
                agent1_type = agent_type_lookup[agent1_id]
                agent2_type = agent_type_lookup[agent2_id]
                
                # Check event constraints (type-level)
                if ((agent1_type, agent2_type) not in allowed_connections and 
                    (agent2_type, agent1_type) not in allowed_connections):
                    continue
                
                connections.append((agent1_id, agent2_id, 'complete_connection'))
        
        return connections


class StarTopologyStrategy(NetworkTopologyStrategy):
    """Generate star topology with central hub agent."""
    
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate star topology relationships with a central hub agent instance."""
        central_agent_type = params.get('central_agent')
        central_agent_id = params.get('central_agent_id')
        
        # Create agent type lookup and flatten all agents
        agent_type_lookup = {}
        all_agent_instances = []
        
        for agent_type, ids in agent_ids.items():
            for agent_id in ids:
                agent_id_str = str(agent_id)
                agent_type_lookup[agent_id_str] = agent_type
                all_agent_instances.append(agent_id_str)
        
        # Select central agent
        if central_agent_id and str(central_agent_id) in agent_type_lookup:
            # Use specific agent ID if provided
            central_agent = str(central_agent_id)
        elif central_agent_type and central_agent_type in agent_ids:
            # Use first agent of specified type
            central_agent = str(agent_ids[central_agent_type][0])
        else:
            # Auto-select first available agent
            central_agent = all_agent_instances[0]
            
        logger.info(f"Selected central agent: {central_agent} (type: {agent_type_lookup[central_agent]})")
        
        connections = []
        central_agent_type = agent_type_lookup[central_agent]
        
        # Connect central agent to all other agents
        for agent_id in all_agent_instances:
            if agent_id != central_agent:
                other_agent_type = agent_type_lookup[agent_id]
                
                # Check event constraints (type-level)
                if ((central_agent_type, other_agent_type) not in allowed_connections and 
                    (other_agent_type, central_agent_type) not in allowed_connections):
                    continue
                
                connections.append((central_agent, agent_id, 'hub_connection'))
        
        return connections


class HierarchicalTopologyStrategy(NetworkTopologyStrategy):
    """Generate hierarchical/tree topology."""
    
    def generate_connections(
        self, 
        agent_ids: Dict[str, List[str]], 
        allowed_connections: Set[Tuple[str, str]],
        params: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Generate hierarchical relationships between agent instances."""
        branching_factor = params.get('branching_factor', 2)
        
        # Create agent type lookup and flatten all agents
        agent_type_lookup = {}
        all_agent_instances = []
        
        for agent_type, ids in agent_ids.items():
            for agent_id in ids:
                agent_id_str = str(agent_id)
                agent_type_lookup[agent_id_str] = agent_type
                all_agent_instances.append(agent_id_str)
        
        connections = []
        
        # Create tree structure among agent instances
        for i, agent_id in enumerate(all_agent_instances):
            # Calculate parent index
            if i > 0:
                parent_idx = (i - 1) // branching_factor
                if parent_idx < len(all_agent_instances):
                    parent_agent_id = all_agent_instances[parent_idx]
                    
                    # Get agent types for constraint checking
                    parent_agent_type = agent_type_lookup[parent_agent_id]
                    child_agent_type = agent_type_lookup[agent_id]
                    
                    # Check event constraints (type-level)
                    if ((parent_agent_type, child_agent_type) in allowed_connections or 
                        (child_agent_type, parent_agent_type) in allowed_connections):
                        connections.append((parent_agent_id, agent_id, 'hierarchical_relation'))
        
        return connections


# Strategy registry
TOPOLOGY_STRATEGIES = {
    'random': RandomTopologyStrategy(),
    'grid': GridTopologyStrategy(),
    'complete': CompleteTopologyStrategy(),
    'star': StarTopologyStrategy(),
    'hierarchical': HierarchicalTopologyStrategy(),
}


def get_strategy(strategy_name: str) -> NetworkTopologyStrategy:
    """Get topology strategy by name."""
    return TOPOLOGY_STRATEGIES.get(strategy_name)


def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return TOPOLOGY_STRATEGIES