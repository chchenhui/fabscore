"""
OneSim Data Storage Module

This module provides database storage for scenarios, trails, environment states,
agents, agent states, events, and agent decisions.
"""

from onesim.data.database import DatabaseManager
from onesim.data.scenario import ScenarioManager
from onesim.data.trail import TrailManager, TrailStatus
from onesim.data.environment import EnvironmentStateManager
from onesim.data.agent import AgentManager
from onesim.data.event import EventManager
from onesim.data.decision import DecisionManager

# Create singleton instances for easy access
def get_database_manager():
    """Get the database manager instance"""
    return DatabaseManager.get_instance()

def get_scenario_manager():
    """Get the scenario manager instance"""
    return ScenarioManager()

def get_trail_manager():
    """Get the trail manager instance"""
    return TrailManager()

def get_environment_state_manager():
    """Get the environment state manager instance"""
    return EnvironmentStateManager()

def get_agent_manager():
    """Get the agent manager instance"""
    return AgentManager()

def get_event_manager():
    """Get the event manager instance"""
    return EventManager()

def get_decision_manager():
    """Get the decision manager instance"""
    return DecisionManager()

__all__ = [
    'DatabaseManager', 'ScenarioManager', 'TrailManager', 'TrailStatus',
    'EnvironmentStateManager', 'AgentManager', 'EventManager', 'DecisionManager',
    'get_database_manager', 'get_scenario_manager', 'get_trail_manager',
    'get_environment_state_manager', 'get_agent_manager', 'get_event_manager',
    'get_decision_manager'
] 