"""
Inspiration Agent for generating observations and simulation scenarios.

This module provides the InspirationAgent class, which is responsible for
generating potential observations (O) and corresponding simulation scenarios
based on a given social science theory (T) and its conditions (C).
"""

import json
from typing import Dict, Any, List

from .agent_base import AgentBase


class InspirationAgent(AgentBase):
    """
    Agent responsible for generating research questions and simulation scenarios
    following the O = f(T, C) paradigm.

    The inspiration agent takes a social science theory and a specific condition
    as input. It then generates specific research questions (based on expected
    observations) and corresponding multi-agent simulation scenarios.
    """

    def __init__(self, model_config=None):
        """
        Initializes the inspiration agent.

        Args:
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        # Call the parent class's constructor to set up the agent name and model config.
        super().__init__("Inspiration Agent", model_config)

    def _construct_system_prompt(self) -> str:
        """
        Constructs the system prompt to guide the agent's behavior.

        Returns:
            str: The system prompt instructing the agent on its task and output format.
        """
        return """You are an expert social science researcher specializing in multi-agent simulations.
Your task is to generate corresponding simulation scenarios based on a social science theory and a specific condition of that theory.

For each simulation scenario:
1.  **Outline** the key agent types involved.
2.  **Describe** the basic interaction mechanisms between agents.
3.  **Suggest** the parameters or variables that can be manipulated.
4.  **Keep** the complexity manageable (maximum of 2-4 agent types).

Please write the and scenarios.
Format your response as JSON with the following structure:
{
      "simulation_scenario": {
        "description": "Brief description of the simulation scenario",
        "agent_types": ["List of agent types"],
        "interactions": "Description of how agents interact",
        "key_parameters": ["List of important parameters"],
        "expected_insights": "What insights might be gained",
      }
}

Be creative but realistic. Focus on scenarios that would yield meaningful insights while being feasible to implement with language model-based agents."""

    def _construct_user_prompt(self, theory: str, condition: str) -> str:
        """
        Constructs the user prompt containing the specific theory and condition.

        Args:
            theory (str): The social science theory.
            condition (str): The specific condition or boundary.
            
        Returns:
            str: The user prompt providing the context for the agent's generation task.
        """
        # The user prompt has been updated to provide the theory and condition directly.
        # This replaces the old "vague research topic" input.
        return f"""Please generate simulation scenarios based on the social science theory and condition provided:

Theory: {theory}
Condition: {condition}

For each research question and scenario:
1. Ensure it is specific enough to be simulated with language model-based agents.
2. Consider what kinds of interactions between agents would be most revealing.
3. Assess the complexity and feasibility of implementing the simulation.

Remember to format your response as JSON with the structure specified in your instructions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (Theory and Condition) and generates research questions and scenarios.

        Args:
            input_data (Dict[str, Any]): Input data containing 'theory' and 'condition'.
            
        Returns:
            Dict[str, Any]: Output data containing the generated research questions and scenarios.
        """
        # The input data extraction logic has been updated to look for 'theory' and 'condition'.
        theory = input_data.get("theory")
        condition = input_data.get("condition")

        if not theory or not condition:
            raise ValueError("Input data must contain 'theory' and 'condition' keys.")

        # Construct the prompts
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(theory, condition)

        print("=" * 20)
        print("System Prompt:")
        print(system_prompt)
        print("=" * 20)
        print("User Prompt:")
        print(user_prompt)
        print("=" * 20)
        
        # Generate the response using the constructed prompts
        response = self.generate_response(system_prompt, user_prompt)

        print("Response from Agent:")
        print(response)
        print("=" * 20)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response)
            return {
                "scenario": parsed_response.get("simulation_scenario", []),
                "input_theory": theory,
                "input_condition": condition
            }
        except json.JSONDecodeError:
            # Fallback for parsing JSON content if the model wraps it in markdown.
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
                    "scenario": parsed_response.get("research_questions", []),
                    "input_theory": theory,
                    "input_condition": condition
                }
            except (json.JSONDecodeError, IndexError) as e:
                # If JSON parsing still fails, raise an informative error.
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...") from e