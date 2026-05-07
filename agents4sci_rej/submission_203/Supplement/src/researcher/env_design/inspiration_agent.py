"""
Inspiration Agent for generating research questions and simulation scenarios.
This module provides the InspirationAgent class, which is responsible for
generating multiple potential research questions and simulation scenarios
from a given vague social science research topic.
"""

import json
from typing import Dict, Any, List

from .agent_base import AgentBase


class InspirationAgent(AgentBase):
    """
    Agent responsible for generating research questions and simulation scenarios.
    
    The inspiration agent takes a vague research topic and generates multiple
    specific research questions and corresponding simulation scenarios that
    could be explored using multi-agent simulations.
    """
    
    def __init__(self, model_config=None):
        """
        Initialize the inspiration agent.
        
        Args:
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        super().__init__("Inspiration Agent", model_config)
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt for the inspiration agent.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert social science researcher specializing in multi-agent simulations.
Your task is to generate multiple promising research questions and corresponding simulation scenarios
based on a vague social science research topic.

For each research question:
1. Focus on a specific aspect of the broader topic that would be suitable for agent-based modeling
2. Ensure the question is well-defined, specific, and can be addressed through simulation
3. Consider both theoretical relevance and practical feasibility

For each simulation scenario:
1. Outline the key agent types that would be involved
2. Describe basic interaction mechanisms between agents
3. Suggest what parameters or variables might be manipulated
4. Keep complexity manageable (2-4 agent types maximum)

Provide 3-5 distinct research questions and scenarios, each targeting different aspects of the topic.
Format your response as JSON with the following structure:
{
  "research_questions": [
    {
      "id": 1,
      "question": "Specific research question",
      "rationale": "Why this question is important and suitable for simulation",
      "simulation_scenario": {
        "description": "Brief description of the simulation scenario",
        "Independent_variable" : "What variable will be manipulated",
        "Independent_variable_range": "[List of the independent variable, detail numbers]",
        "dependent_variable": "What variable will be measured",
        "agent_types": ["List of agent types"],
        "interactions": "Description of how agents interact",
        "key_parameters": ["List of important parameters"],
        "expected_insights": "What insights might be gained",
      }
    },
    ... additional questions ...
  ]
}

Be creative but realistic. Focus on scenarios that would yield meaningful insights while being feasible to implement with language model-based agents."""
    
    def _construct_user_prompt(self, research_topic: str) -> str:
        """
        Construct the user prompt for the inspiration agent.
        
        Args:
            research_topic (str): The vague research topic.
            
        Returns:
            str: The user prompt.
        """
        return f"""Please generate 3-5 specific research questions and corresponding simulation scenarios based on this vague social science research topic:

RESEARCH TOPIC: {research_topic}

For each research question and scenario:
1. Ensure it is specific enough to be simulated with language model-based agents
2. Consider what kinds of interactions between agents would be most revealing
3. Think about how the simulation results could provide insights into the broader topic
4. Assess the complexity and feasibility of implementing the simulation

Remember to format your response as JSON with the structure specified in your instructions."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and generate research questions and scenarios.
        
        Args:
            input_data (Dict[str, Any]): Input data containing the research topic.
            
        Returns:
            Dict[str, Any]: Output data containing the generated research questions
                and scenarios.
        """
        # Extract research topic
        research_topic = input_data.get("research_topic", "")
        
        if not research_topic:
            raise ValueError("Research topic is required.")
        
        # Generate response
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(research_topic)

        print("=" * 20)
        print("System Prompt:")
        print(system_prompt)
        print("=" * 20)
        print("User Prompt:")
        print(user_prompt)
        print("=" * 20)
        
        response = self.generate_response(system_prompt, user_prompt)
        print("Response from Agent:")
        print(response)
        print("=" * 20)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response)
            return {
                "research_questions": parsed_response.get("research_questions", []),
                "original_topic": research_topic
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON content from text response
            try:
                # Look for JSON content between triple backticks
                if "```json" in response:
                    json_content = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_content = response.split("```")[1].strip()
                else:
                    json_content = response
                
                parsed_response = json.loads(json_content)
                return {
                    "research_questions": parsed_response.get("research_questions", []),
                    "original_topic": research_topic
                }
            except (json.JSONDecodeError, IndexError):
                # If still fails, return an error message
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...") 