import networkx as nx
import random
from typing import Any, Dict, List, Optional, Tuple
from pyvis.network import Network
from loguru import logger
import threading
import uuid

# Set up logging

class SingletonMeta(type):
    """
    A metaclass for implementing the Singleton pattern.
    """
    _instances: Dict[Any, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Ensure thread safety
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class WorkGraph(metaclass=SingletonMeta):
    def __init__(self):
        """
        Initialize WorkGraph.
        """
        self.graph = nx.DiGraph()
        self.actions: Dict[str, List[Dict[str, Any]]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()  # Add thread lock

    def load_workflow_data(self, actions: Dict[str, List[Dict[str, Any]]], events: Dict[str, Dict[str, Any]]):
        """
        Load new actions and events into the WorkGraph.
        Clears the existing graph and rebuilds it with the new data.
        """
        with self._lock:
            self.graph.clear()  # Clear all nodes and edges from the graph
            self.actions = actions
            self.events = events
            self._build_graph() # Rebuild with new actions and events

    def _ensure_unique_event_names(self):
        """
        Ensure all event names are unique. Raises an exception if duplicates are found.
        """
        event_names = set()
        for event_id, event in self.events.items():
            event_name = event.get('event_name')
            if event_name in event_names:
                raise ValueError(f"Duplicate event name detected: {event_name}")
            event_names.add(event_name)

    def _build_graph(self):
        """Build the graph by adding nodes and edges."""
        # Add actions as nodes, node IDs formatted as {agent_type}.{action_name}
        for agent_type, actions_list in self.actions.items():
            for action in actions_list:
                node_id = f"{agent_type}.{action['name']}"
                if self.graph.has_node(node_id):
                    raise ValueError(f"Duplicate node: {node_id}")
                self.graph.add_node(node_id, 
                                    type='action', 
                                    agent_type=agent_type,
                                    name=action['name'],
                                    description=action['description'],
                                    action_type=action['type'],
                                    required_variables=action.get('required_variables', []),
                                    output_updates=action.get('output_updates', []))
                #logger.info(f"Action node added: {node_id}")

        # Add special EnvAgent action nodes if necessary
        env_start_node_id = "EnvAgent.start"
        if env_start_node_id not in self.graph:
            self.graph.add_node(env_start_node_id, 
                                type='action', 
                                agent_type='EnvAgent',
                                name='start',
                                description='Environment start action',
                                action_type='OR',
                                required_variables=[],
                                output_updates=[])
            #logger.info(f"EnvAgent start node added: {env_start_node_id}")

        env_end_node_id = "EnvAgent.terminate"
        if env_end_node_id not in self.graph:
            self.graph.add_node(env_end_node_id, 
                                type='action', 
                                agent_type='EnvAgent',
                                name='terminate',
                                description='Environment end action',
                                action_type='OR',
                                required_variables=[],
                                output_updates=[])
            #logger.info(f"EnvAgent start node added: {env_start_node_id}")

        # Add events as edges, labeled with a unique event_name
        for event_id, event in self.events.items():
            from_agent_type = event['from_agent_type']
            to_agent_type = event['to_agent_type']
            from_action = event['from_action_name']
            to_action = event['to_action_name']
            event_name = event['event_name']

            from_node_id = f"{from_agent_type}.{from_action}"
            to_node_id = f"{to_agent_type}.{to_action}"

            if not self.graph.has_node(from_node_id):
                raise ValueError(f"Source node does not exist: {from_node_id}")
            if not self.graph.has_node(to_node_id):
                raise ValueError(f"Target node does not exist: {to_node_id}")

            self.graph.add_edge(from_node_id, to_node_id, 
                                type='event',
                                event_name=event_name,
                                event_info=event.get('event_info', ''),
                                fields=event.get('fields', []))

    # ===================== Node Operations =====================

    def add_action(self, agent_type: str, name: str, description: str,
                   action_type: str, required_variables: List[Dict[str, Any]] = None,
                   output_updates: List[Dict[str, Any]] = None) -> bool:
        """
        Add a new action node.

        :param agent_type: The agent type to which the action belongs
        :param name: The name of the action
        :param description: Description of the action
        :param action_type: The action type (e.g., "OR", "AND")
        :param required_variables: Variables required by the action
        :param output_updates: Output updates for the action
        :return: Whether the addition was successful
        """
        if required_variables is None:
            required_variables = []
        if output_updates is None:
            output_updates = []

        node_id = f"{agent_type}.{name}"
        with self._lock:
            if self.graph.has_node(node_id):
                logger.warning(f"Action node {node_id} already exists.")
                return False
            self.graph.add_node(node_id,
                                type='action',
                                agent_type=agent_type,
                                name=name,
                                description=description,
                                action_type=action_type,
                                required_variables=required_variables,
                                output_updates=output_updates)
            #logger.info(f"Action node added successfully: {node_id}")
            return True

    def remove_action(self, agent_type: str, name: str) -> bool:
        """
        Remove an action node.

        :param agent_type: The agent type to which the action belongs
        :param name: The name of the action
        :return: Whether the removal was successful
        """
        node_id = f"{agent_type}.{name}"
        with self._lock:
            if not self.graph.has_node(node_id):
                logger.warning(f"Action node {node_id} does not exist.")
                return False
            self.graph.remove_node(node_id)
            #logger.info(f"Action node removed successfully: {node_id}")
            return True

    def update_action(self, agent_type: str, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update the properties of an action node.

        :param agent_type: The agent type to which the action belongs
        :param name: The name of the action
        :param updates: A dictionary of properties to update
        :return: Whether the update was successful
        """
        node_id = f"{agent_type}.{name}"
        with self._lock:
            if not self.graph.has_node(node_id):
                logger.warning(f"Action node {node_id} does not exist.")
                return False
            for key, value in updates.items():
                if key in self.graph.nodes[node_id]:
                    self.graph.nodes[node_id][key] = value
                    #logger.info(f"Updated property {key} of action node {node_id} to {value}")
                else:
                    logger.warning(f"Property {key} does not exist on action node {node_id}.")
            return True

    # ===================== Edge Operations =====================

    def add_event(self, from_agent_type: str, from_action: str,
                 to_agent_type: str, to_action: str,
                 event_name: str, event_info: str,
                 fields: List[Dict[str, Any]] = []) -> bool:
        """
        Add a new event edge.

        :param from_agent_type: The source agent type
        :param from_action: The source action name
        :param to_agent_type: The target agent type
        :param to_action: The target action name
        :param event_name: The name of the event
        :param event_info: Information about the event
        :param fields: List of fields for the event
        :return: Whether the addition was successful
        """
        from_node_id = f"{from_agent_type}.{from_action}"
        to_node_id = f"{to_agent_type}.{to_action}"

        with self._lock:
            if not self.graph.has_node(from_node_id):
                logger.warning(f"Source node {from_node_id} does not exist. Cannot add event {event_name}.")
                return False
            if not self.graph.has_node(to_node_id):
                logger.warning(f"Target node {to_node_id} does not exist. Cannot add event {event_name}.")
                return False
            # Check if event_name is unique
            for _, _, edge_data in self.graph.edges(data=True):
                if edge_data.get('event_name') == event_name:
                    logger.warning(f"Event name {event_name} already exists. Cannot add duplicate.")
                    return False
            self.graph.add_edge(from_node_id, to_node_id,
                                type='event',
                                event_name=event_name,
                                event_info=event_info,
                                fields=fields)
            #logger.info(f"Event edge added successfully: {event_name} from {from_node_id} to {to_node_id}")
            return True

    def remove_event(self, event_name: str) -> bool:
        """
        Remove an event edge.

        :param event_name: The name of the event
        :return: Whether the removal was successful
        """
        with self._lock:
            for from_node, to_node, edge_data in list(self.graph.edges(data=True)):
                if edge_data.get('event_name') == event_name:
                    self.graph.remove_edge(from_node, to_node)
                    #logger.info(f"Event edge removed successfully: {event_name} from {from_node} to {to_node}")
                    return True
            logger.warning(f"Event name {event_name} does not exist.")
            return False

    def update_event(self, event_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update the properties of an event edge.

        :param event_name: The name of the event
        :param updates: A dictionary of properties to update
        :return: Whether the update was successful
        """
        with self._lock:
            for from_node, to_node, edge_data in self.graph.edges(data=True):
                if edge_data.get('event_name') == event_name:
                    for key, value in updates.items():
                        if key in edge_data:
                            edge_data[key] = value
                            #logger.info(f"Updated property {key} of event edge {event_name} to {value}")
                        else:
                            logger.warning(f"Property {key} does not exist on event edge {event_name}.")
                    return True
            logger.warning(f"Event name {event_name} does not exist.")
            return False

    # ===================== Query Functions =====================

    def get_predecessor_ids(self, node_id: str) -> List[str]:
        """Get the list of predecessor action node IDs for a given action."""
        with self._lock:
            return list(self.graph.predecessors(node_id))

    def get_successor_ids(self, node_id: str) -> List[str]:
        """Get the list of successor action node IDs for a given action."""
        with self._lock:
            return list(self.graph.successors(node_id))

    def get_successor_agent_types(self, node_id: str) -> List[str]:
        """Get the list of agent types for successor action nodes of a given action."""
        if node_id not in self.graph:
            return []
        with self._lock:
            return [self.graph.nodes[successor_id]['agent_type'] for successor_id in self.graph.successors(node_id)]

    def get_action_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific action."""
        with self._lock:
            if self.graph.has_node(node_id):
                return self.graph.nodes[node_id]
            return None

    def get_event_info(self, event_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific event by its name."""
        with self._lock:
            for _, _, edge_data in self.graph.edges(data=True):
                if edge_data.get('event_name') == event_name:
                    return edge_data
            return None

    def get_actions_by_agent(self, agent_type: str) -> List[str]:
        """Find all action node IDs associated with a specific agent type."""
        with self._lock:
            return [node for node, data in self.graph.nodes(data=True) if data.get('agent_type') == agent_type]

    def get_events_by_name(self, event_name: str) -> List[str]:
        """Find the list of event names by their name (since event_name is unique, return a single list element)."""
        with self._lock:
            return [edge_data['event_name'] for _, _, edge_data in self.graph.edges(data=True) if edge_data.get('event_name') == event_name]

    def get_start_agent_types(self) -> List[str]:
        """Find all starting agent types."""
        start_agent_types = []
        with self._lock:
            for event_id, event in self.events.items():
                if event.get('event_name') == 'StartEvent':
                    start_agent_types.append(event.get('to_agent_type'))
        return list(set(start_agent_types))

    def get_end_agent_types(self) -> List[str]:
        """Find all ending agent types."""
        end_agent_types = []
        with self._lock:
            for event_id, event in self.events.items():
                if event.get('to_action_name') == 'terminate':
                    end_agent_types.append(event.get('from_agent_type'))
        return list(set(end_agent_types))

    def get_end_events(self) -> List[str]:
        """Find all ending events."""
        end_events = []
        with self._lock:
            for event_id, event in self.events.items():
                if event.get('to_action_name') == 'terminate':
                    end_events.append(event['event_name'])
        return end_events
    # ===================== 可视化 =====================

    def visualize_interactive_graph(self, filename="workflow.html"):
        """使用 pyvis 交互式可视化图形。
        
        参数:
            filename (str): 保存的 HTML 文件名。
        """
        with self._lock:
            net = Network(height="750px", width="100%", directed=True)
            net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=150, spring_strength=0.05, damping=0.09)

            # 为每种 Agent 生成唯一颜色
            agent_colors = {}
            for node_id, node_data in self.graph.nodes(data=True):
                agent_type = node_data.get("agent_type", "UnknownAgent")
                if agent_type not in agent_colors:
                    agent_colors[agent_type] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                node_color = agent_colors[agent_type]
                node_label = node_data.get('name', '')
                node_title = f"Agent: {agent_type}\nAction: {node_data.get('name', '')}"

                net.add_node(
                    node_id,
                    label=node_label,
                    title=node_title,
                    color=node_color,
                    size=25,
                    borderWidth=2
                )

            # 添加边并确保 event_name 唯一
            for source_id, target_id, edge_data in self.graph.edges(data=True):
                edge_label = edge_data.get("event_name", "")
                edge_title = f"Event: {edge_label}\nInfo: {edge_data.get('event_info', '')}"

                net.add_edge(
                    source_id,
                    target_id,
                    label=edge_label,
                    title=edge_title,
                    color="gray",
                    arrows="to"
                )

            # 设置 pyvis 的图形选项
            net.set_options("""
            var options = {
                "nodes": {
                    "font": {
                        "size": 16,
                        "face": "Arial",
                        "color": "black",
                        "background": "rgba(255,255,255,0.8)"
                    },
                    "scaling": {
                        "min": 15,
                        "max": 40
                    },
                    "borderWidth": 2,
                    "borderWidthSelected": 4,
                    "color": {
                        "highlight": {
                            "border": "#2B7CE9",
                            "background": "#D2E5FF"
                        }
                    },
                    "shape": "dot"
                },
                "edges": {
                    "color": {
                        "color": "gray",
                        "highlight": "blue",
                        "hover": "blue"
                    },
                    "width": 1.5,
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 1.5
                        }
                    },
                    "font": {
                        "size": 14,
                        "face": "Arial",
                        "align": "horizontal",
                        "color": "darkred",
                        "strokeWidth": 1,
                        "strokeColor": "#ffffff",
                        "multi": true
                    },
                    "smooth": {
                        "type": "dynamic"
                    }
                },
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "iterations": 300
                    },
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.2,
                        "springLength": 150,
                        "springConstant": 0.05,
                        "damping": 0.09
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 200,
                    "hideEdgesOnDrag": false
                },
                "layout": {
                    "improvedLayout": true
                }
            }
            """)

            # 保存 HTML 文件
            net.save_graph(filename)
            logger.info(f"交互式图形已保存到 {filename}。请在浏览器中打开以查看。")
