import random
import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from loguru import logger
from .work_graph import WorkGraph


def generate_relationships(relationship_schema: List[Dict[str, Any]], 
                          all_agent_ids: Dict[str, List[str]], 
                          actions: Dict[str, List[Dict[str, Any]]],
                          events: Dict[str, Dict[str, Any]],
                          output_csv_path: Optional[str] = None,
                          min_relationships: int = 1,
                          max_relationships: int = 1) -> List[Dict[str, Any]]:
    """
    Generate relationships based on the relationship schema and agent ids,
    ensuring path connectivity from start to terminate in the simulation topology.
    
    Args:
        relationship_schema: List of relationship definitions
        all_agent_ids: Dictionary mapping agent types to lists of agent IDs
        actions: Dictionary containing lists of actions for each agent type
        events: Dictionary containing events
        output_csv_path: Path to save the CSV file (optional)
        min_relationships: Minimum relationships per agent
        max_relationships: Maximum relationships per agent
        
    Returns:
        List of generated relationships
    """
    # Input validation
    if not isinstance(relationship_schema, list):
        raise ValueError("Invalid relationship_schema: must be a list")
    if not isinstance(all_agent_ids, dict):
        raise ValueError("Invalid all_agent_ids: must be a dictionary")
    if not isinstance(actions, dict):
        raise ValueError("Invalid actions: must be a dictionary")
    if not isinstance(events, dict):
        raise ValueError("Invalid events: must be a dictionary")
    
    # Create work graph
    work_graph = WorkGraph()
    work_graph.load_workflow_data(actions, events)
    
    # Convert all agent IDs to strings and create lookup tables     
    all_agent_ids = {k: [str(id) for id in v] for k, v in all_agent_ids.items()}
    agent_type_lookup = {str(agent_id): agent_type for agent_type, agent_ids in all_agent_ids.items() for agent_id in agent_ids}
    
    # Initialize relationship tracking
    relationships_list = []
    existing_relationships = set()
    
    # Analyze topology
    topology_paths = analyze_topology(work_graph, all_agent_ids)
    
    # Generate relationships based on topology
    if topology_paths:
        relationships_list.extend(generate_topology_relationships(
            topology_paths, relationship_schema, all_agent_ids, existing_relationships
        ))
    
    # Generate relationships based on schema
    relationships_list.extend(generate_schema_relationships(
        relationship_schema, all_agent_ids, existing_relationships
    ))
    
    # Add random relationships to meet minimum requirements
    relationships_list.extend(generate_random_relationships(
        min_relationships, max_relationships, relationship_schema, all_agent_ids, existing_relationships
    ))
    
    # Ensure connectivity
    ensure_connectivity(work_graph, relationships_list, all_agent_ids, agent_type_lookup, 
                        relationship_schema, existing_relationships)
    
    # Save relationships if path provided
    if output_csv_path:
        save_relationships(relationships_list, output_csv_path)
    
    
    return relationships_list


def analyze_topology(work_graph: WorkGraph, all_agent_ids: Dict[str, List[str]]) -> List[List[str]]:
    """Analyze the topology and return all possible paths from start to end agents."""
    topology_paths = []
    start_agent_types = work_graph.get_start_agent_types()
    end_agent_types = work_graph.get_end_agent_types()
    
    for start_type in start_agent_types:
        paths = find_paths_to_end(work_graph, start_type, end_agent_types)
        topology_paths.extend(paths)
    
    logger.info(f"Found {len(topology_paths)} distinct paths in topology")
    return topology_paths


def find_paths_to_end(work_graph: WorkGraph, start_type: str, end_types: List[str]) -> List[List[str]]:
    """
    Find all possible agent type paths from start_type to any end_type.
    """
    paths = []
    visited = set()
    
    def dfs(current_type: str, current_path: List[str]):
        if current_type in end_types:
            paths.append(current_path.copy())
            return
        
        visited.add(current_type)
        
        # Get all actions for this agent type
        actions = work_graph.get_actions_by_agent(current_type)
        for action_id in actions:
            # Get successor agent types for this action
            successor_types = work_graph.get_successor_agent_types(action_id)
            for next_type in successor_types:
                if next_type not in visited:
                    current_path.append(next_type)
                    dfs(next_type, current_path)
                    current_path.pop()  # Backtrack
                    
        visited.remove(current_type)
    
    dfs(start_type, [start_type])
    return paths


def generate_topology_relationships(
    topology_paths: List[List[str]], 
    relationship_schema: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]], 
    existing_relationships: Set[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Generate relationships based on topology paths."""
    relationships_list = []
    
    for path in topology_paths:
        for i in range(len(path) - 1):
            source_type = path[i]
            target_type = path[i+1]
            
            # Get relationship direction from schema
            direction = get_relationship_direction(relationship_schema, source_type, target_type)
            
            # Get agents of these types
            source_agents = all_agent_ids.get(source_type, [])
            target_agents = all_agent_ids.get(target_type, [])
            
            if not source_agents or not target_agents:
                logger.warning(f"Missing agents for {source_type}->{target_type} path segment")
                continue
            
            # Distribute connections evenly
            for source_id in source_agents:
                target_id = select_target_agent(source_id, target_agents, existing_relationships)
                if target_id:
                    relationship_key = (str(source_id), str(target_id))
                    existing_relationships.add(relationship_key)
                    relationships_list.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "direction": direction
                    })
    
    return relationships_list


def get_relationship_direction(
    relationship_schema: List[Dict[str, Any]], 
    source_type: str, 
    target_type: str
) -> str:
    """Get relationship direction from schema."""
    for rel in relationship_schema:
        if rel["source_agent"] == source_type and rel["target_agent"] == target_type:
            return rel["direction"]
        if rel["source_agent"] == target_type and rel["target_agent"] == source_type and rel["direction"] == "bidirectional":
            return "bidirectional"
    logger.warning(f"Could not find relationship direction for {source_type}->{target_type}")
    return "bidirectional"  # Default


def select_target_agent(
    source_id: str, 
    target_agents: List[str], 
    existing_relationships: Set[Tuple[str, str]]
) -> Optional[str]:
    """Select a target agent for connection, avoiding existing relationships."""
    for target_id in target_agents:
        if target_id != source_id and (source_id, target_id) not in existing_relationships:
            return target_id
    return None


def generate_schema_relationships(
    relationship_schema: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]], 
    existing_relationships: Set[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Generate relationships based on schema requirements."""
    relationships_list = []
    
    for relationship in relationship_schema:
        source_type = relationship["source_agent"]
        target_type = relationship["target_agent"]
        direction = relationship["direction"]
        
        source_ids = all_agent_ids.get(source_type, [])
        target_ids = all_agent_ids.get(target_type, [])
        
        if not source_ids or not target_ids:
            logger.warning(f"No agents found for {source_type}->{target_type} relationship")
            continue
        
        # Track connected agents
        sources_with_connections = set()
        targets_with_connections = set()
        
        # Check existing relationships
        for rel in relationships_list:
            if rel["source_id"] in source_ids and rel["target_id"] in target_ids:
                sources_with_connections.add(rel["source_id"])
                targets_with_connections.add(rel["target_id"])
        
        # Connect unconnected sources
        for source_id in source_ids:
            if source_id not in sources_with_connections:
                target_id = find_available_target(source_id, target_ids, targets_with_connections, existing_relationships)
                if target_id:
                    add_relationship(relationships_list, existing_relationships, source_id, target_id, direction)
        
        # Handle bidirectional relationships
        if direction == "bidirectional":
            for target_id in target_ids:
                if target_id not in targets_with_connections:
                    source_id = find_available_source(target_id, source_ids, sources_with_connections, existing_relationships)
                    if source_id:
                        add_relationship(relationships_list, existing_relationships, source_id, target_id, direction)
    
    return relationships_list


def find_available_target(
    source_id: str, 
    target_ids: List[str], 
    targets_with_connections: Set[str], 
    existing_relationships: Set[Tuple[str, str]]
) -> Optional[str]:
    """Find an available target agent for connection."""
    # Prefer unconnected targets
    available_targets = [t for t in target_ids if t not in targets_with_connections and t != source_id 
                        and (source_id, t) not in existing_relationships]
    
    if not available_targets:
        available_targets = [t for t in target_ids if t != source_id and (source_id, t) not in existing_relationships]
    
    return available_targets[0] if available_targets else None


def find_available_source(
    target_id: str, 
    source_ids: List[str], 
    sources_with_connections: Set[str], 
    existing_relationships: Set[Tuple[str, str]]
) -> Optional[str]:
    """Find an available source agent for connection."""
    available_sources = [s for s in source_ids if s not in sources_with_connections and s != target_id 
                        and (s, target_id) not in existing_relationships]
    
    if not available_sources:
        available_sources = [s for s in source_ids if s != target_id and (s, target_id) not in existing_relationships]
    
    return available_sources[0] if available_sources else None


def add_relationship(
    relationships_list: List[Dict[str, Any]], 
    existing_relationships: Set[Tuple[str, str]], 
    source_id: str, 
    target_id: str, 
    direction: str
) -> None:
    """Add a new relationship to the list and update existing relationships set."""
    relationship_key = (str(source_id), str(target_id))
    if relationship_key not in existing_relationships:
        existing_relationships.add(relationship_key)
        relationships_list.append({
            "source_id": source_id,
            "target_id": target_id,
            "direction": direction
        })


def generate_random_relationships(
    min_relationships: int,
    max_relationships: int,
    relationship_schema: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]], 
    existing_relationships: Set[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Generate random relationships to meet minimum requirements."""
    relationships_list = []
    
    for source_type, source_ids in all_agent_ids.items():
        for source_id in source_ids:
            current_rel_count = sum(1 for rel in relationships_list if rel["source_id"] == source_id)
            
            if current_rel_count >= min_relationships:
                continue
            
            # Find compatible targets
            compatible_targets = find_compatible_targets(source_id, source_type, relationship_schema, all_agent_ids, existing_relationships)
            
            # Add random relationships
            needed = min_relationships - current_rel_count
            if compatible_targets and needed > 0:
                sample_size = min(needed, len(compatible_targets))
                selected = random.sample(compatible_targets, sample_size)
                
                for target_id, direction in selected:
                    add_relationship(relationships_list, existing_relationships, source_id, target_id, direction)
    
    return relationships_list


def find_compatible_targets(
    source_id: str, 
    source_type: str, 
    relationship_schema: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]], 
    existing_relationships: Set[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Find compatible target agents for a given source agent."""
    compatible_targets = []
    for rel in relationship_schema:
        if rel["source_agent"] == source_type:
            target_type = rel["target_agent"]
            for target_id in all_agent_ids.get(target_type, []):
                if target_id != source_id and (source_id, target_id) not in existing_relationships:
                    compatible_targets.append((target_id, rel["direction"]))
    return compatible_targets


def ensure_connectivity(
    work_graph: WorkGraph, 
    relationships_list: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]],
    agent_type_lookup: Dict[str, str],
    relationship_schema: List[Dict[str, Any]], 
    existing_relationships: Set[Tuple[str, str]]
) -> None:
    """Ensure all terminal agents are reachable from starting agents."""
    # Build temporary graph
    temp_graph = nx.DiGraph()
    
    # Add all agents as nodes first to prevent missing node errors
    for agent_type, agent_ids in all_agent_ids.items():
        for agent_id in agent_ids:
            # Ensure agent_id is a string
            temp_graph.add_node(str(agent_id))
    
    # Now add edges
    for rel in relationships_list:
        source_id = str(rel["source_id"])
        target_id = str(rel["target_id"])
        temp_graph.add_edge(source_id, target_id)
        if rel["direction"] == "bidirectional":
            temp_graph.add_edge(target_id, source_id)
    
    # Get start and end agents
    start_agent_types = work_graph.get_start_agent_types()
    end_agent_types = work_graph.get_end_agent_types()
    
    start_agents = [str(agent_id) for agent_type in start_agent_types 
                   for agent_id in all_agent_ids.get(agent_type, [])]
    end_agents = [str(agent_id) for agent_type in end_agent_types 
                 for agent_id in all_agent_ids.get(agent_type, [])]
    
    # Check and fix connectivity
    for end_agent in end_agents:
        if not any(nx.has_path(temp_graph, start_agent, end_agent) 
                   for start_agent in start_agents 
                   if start_agent in temp_graph and end_agent in temp_graph):
            fix_connectivity(end_agent, start_agents, agent_type_lookup, relationship_schema, 
                           relationships_list, existing_relationships)


def fix_connectivity(
    end_agent: str, 
    start_agents: List[str], 
    agent_type_lookup: Dict[str, str],
    relationship_schema: List[Dict[str, Any]], 
    relationships_list: List[Dict[str, Any]], 
    existing_relationships: Set[Tuple[str, str]]
) -> None:
    """Fix connectivity by adding necessary relationships."""
    logger.warning(f"Terminal agent {end_agent} not reachable - adding connection")
    
    if not start_agents:
        logger.error("No start agents available for connectivity fix")
        return
        
    start_agent = random.choice(start_agents)
    direction = "bidirectional"  # Default
    
    # Find direction from schema if possible
    source_type = agent_type_lookup.get(start_agent)
    target_type = agent_type_lookup.get(end_agent)
    if source_type and target_type:
        for rel in relationship_schema:
            if rel["source_agent"] == source_type and rel["target_agent"] == target_type:
                direction = rel["direction"]
                break
    
    add_relationship(relationships_list, existing_relationships, start_agent, end_agent, direction)


def save_relationships(relationships_list: List[Dict[str, Any]], output_csv_path: str) -> None:
    """Save relationships to CSV file."""
    if relationships_list:
        try:
            relationships_df = pd.DataFrame(relationships_list)
            relationships_df.to_csv(output_csv_path, index=False)
            logger.info(f"Relationships generated successfully. Total: {len(relationships_list)}")
        except Exception as e:
            logger.error(f"Error saving relationships to CSV: {e}")
    else:
        logger.warning("No relationships were generated")


def verify_connectivity(
    relationships: List[Dict[str, Any]], 
    all_agent_ids: Dict[str, List[str]],
    agent_type_lookup: Dict[str, str],
    topology_paths: List[List[str]], 
    work_graph: WorkGraph
) -> bool:
    """
    Verify that the generated relationships maintain connectivity by checking:
    1. Each terminal Agent can be reached from some starting Agent
    2. Each starting Agent can reach some terminal Agent
    """
    # Build a graph of agent connections
    G = nx.DiGraph()
    
    # Add all agents as nodes
    for agent_type, agent_ids in all_agent_ids.items():
        for agent_id in agent_ids:
            # Ensure agent_id is a string
            G.add_node(str(agent_id), agent_type=agent_type)
    
    # Add relationships as edges
    for rel in relationships:
        source_id = str(rel["source_id"])
        target_id = str(rel["target_id"])
        G.add_edge(source_id, target_id)
        if rel["direction"] == "bidirectional":
            G.add_edge(target_id, source_id)
    
    # Get start and end agent types
    start_agent_types = work_graph.get_start_agent_types()
    end_agent_types = work_graph.get_end_agent_types()
    
    # Get all start and end agents
    start_agents = []
    for start_type in start_agent_types:
        if start_type in all_agent_ids:
            start_agents.extend([str(agent_id) for agent_id in all_agent_ids[start_type]])
    
    end_agents = []
    for end_type in end_agent_types:
        if end_type in all_agent_ids:
            end_agents.extend([str(agent_id) for agent_id in all_agent_ids[end_type]])
    
    # Check 1: Every terminal agent must be reachable from some starting agent
    unreachable_terminals = []
    for end_agent in end_agents:
        if end_agent not in G:
            logger.warning(f"Terminal agent {end_agent} not found in graph")
            unreachable_terminals.append(end_agent)
            continue
            
        reachable = False
        for start_agent in start_agents:
            if start_agent not in G:
                logger.warning(f"Start agent {start_agent} not found in graph")
                continue
                
            if nx.has_path(G, start_agent, end_agent):
                reachable = True
                break
        
        if not reachable:
            unreachable_terminals.append(end_agent)
    
    if unreachable_terminals:
        logger.warning(f"Connectivity issue: {len(unreachable_terminals)} terminal agents cannot be reached from any starting agent")
        logger.warning(f"Examples: {unreachable_terminals[:5]}")
    else:
        logger.info("All terminal agents can be reached from at least one starting agent")
    
    # Check 2: From every starting agent, at least one terminal agent must be reachable
    stranded_starts = []
    for start_agent in start_agents:
        if start_agent not in G:
            logger.warning(f"Start agent {start_agent} not found in graph")
            stranded_starts.append(start_agent)
            continue
            
        can_reach_end = False
        for end_agent in end_agents:
            if end_agent not in G:
                continue
                
            if nx.has_path(G, start_agent, end_agent):
                can_reach_end = True
                break
        
        if not can_reach_end:
            stranded_starts.append(start_agent)
    
    if stranded_starts:
        logger.warning(f"Connectivity issue: {len(stranded_starts)} starting agents cannot reach any terminal agent")
        logger.warning(f"Examples: {stranded_starts[:5]}")
    else:
        logger.info("All starting agents can reach at least one terminal agent")
    
    # Check 3: Verify all topology paths have at least one concrete instance
    if topology_paths:
        for path in topology_paths:
            path_instance_exists = False
            
            # Get agents for the first and last type in the path
            first_type = path[0]
            last_type = path[-1]
            first_agents = [str(agent_id) for agent_id in all_agent_ids.get(first_type, [])]
            last_agents = [str(agent_id) for agent_id in all_agent_ids.get(last_type, [])]
            
            # Check if any concrete path exists
            for first_agent in first_agents:
                if first_agent not in G:
                    continue
                    
                for last_agent in last_agents:
                    if last_agent not in G:
                        continue
                        
                    if nx.has_path(G, first_agent, last_agent):
                        # Verify that the path follows the topology
                        path_instance_exists = True
                        break
                if path_instance_exists:
                    break
            
            if not path_instance_exists:
                logger.warning(f"Topology path not instantiated: {' -> '.join(path)}")
    
    return len(unreachable_terminals) == 0 and len(stranded_starts) == 0 