"""
Experimental Design Package.

This package provides a framework for generating detailed simulation specifications
from vague social science research topics using a multi-agent collaboration approach.

The framework consists of four main agents:
1. InspirationAgent: Generates multiple potential research questions and scenarios
2. EvaluatorAgent: Evaluates and selects the most promising scenario
3. DetailerAgent: Elaborates the selected scenario into a detailed specification
4. AssessorAgent: Evaluates the quality of the conversion from brief description to ODD protocol

These agents are coordinated by the Coordinator class, which manages the workflow.
"""

from .agent_base import AgentBase
from .inspiration_agent import InspirationAgent
from .evaluator_agent import EvaluatorAgent
from .detailer_agent import DetailerAgent
from .assessor_agent import AssessorAgent
from .experiments_agent import Experimentagent
from .intervention_agent import InterventionAgent
from .scenario_agent import ScenarioAgent

__all__ = [
    'AgentBase',
    'InspirationAgent',
    'EvaluatorAgent',
    'DetailerAgent',
    'AssessorAgent',
    'Experimentagent',
    'InterventionAgent',
    'ScenarioAgent'
] 