from typing import Dict, List
from onesim.relationship import Relationship, RelationshipManager
from onesim.agent import GeneralAgent

class SocialNetworkManager:
    def __init__(self):
        self.agent_relationships: Dict[str, RelationshipManager] = {}

    def register_agent(self, agent_id: str, relationship_manager: RelationshipManager):
        self.agent_relationships[agent_id] = relationship_manager

    def register_agents(self, agents: List[GeneralAgent]):
        for agent in agents:
            self.register_agent(agent.agent_id, agent.relationship_manager)

    def get_network_density(self) -> float:
        total_agents = len(self.agent_relationships)
        if total_agents <= 1:
            return 0.0
        total_possible_relationships = total_agents * (total_agents - 1)
        total_actual_relationships = sum(len(rm.relationships) for rm in self.agent_relationships.values())
        density = total_actual_relationships / total_possible_relationships
        return density

    def get_all_agents(self) -> List[str]:
        return list(self.agent_relationships.keys())

    def get_relationships_of_agent(self, agent_id: str) -> List[Relationship]:
        if agent_id in self.agent_relationships:
            return self.agent_relationships[agent_id].get_all_relationships()
        else:
            return []
