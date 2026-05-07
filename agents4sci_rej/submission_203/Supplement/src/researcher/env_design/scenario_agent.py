"""
Scenario Agent for generating structured simulation experiment configurations.

This module provides the ScenarioAgent class, which is responsible for generating
complete experiment scenarios in JSON format. Each scenario includes configuration
details such as agent setup, environment settings, simulation parameters, and
evaluation metrics based on a given research question and contextual scene information.
"""

import json
from typing import Dict, Any
from .agent_base import AgentBase


class ScenarioAgent(AgentBase):
    """
    ScenarioAgent is responsible for generating simulation experiment scenarios
    based on user-provided research questions and contextual scene information.

    The generated output follows a standardized JSON schema and includes detailed
    configurations such as scenario description, simulation parameters, agent
    definitions, environment settings, and evaluation metrics.

    Attributes:
        model_config (str): Optional name of the underlying language model configuration.
    """

    def __init__(self, model_config=None):
        """
        Initializes the ScenarioAgent with an optional language model configuration.

        Args:
            model_config (str, optional): Model configuration to be used.
                                           Defaults to None (uses default model).
        """
        super().__init__("Scenario Agent", model_config)

    def _construct_system_prompt(self) -> str:
        """
        Constructs the system prompt used to define the agent's role and responsibilities.

        Returns:
            str: A system prompt containing detailed instructions and a strict JSON schema
                 for scenario generation.
        """
        return """You are an expert social science researcher specializing in multi-agent simulations.
Your task is to design a SIMULATION SCENARIO configuration based on the research question, scene information and experiment config.
Focus on SIMULATION CONFIGURATION DETAILS rather than experimental design.

1. **scenario_info**: Basic scenario information using consistent scene_name.
2. **simulation_config**: Simulation parameters (steps, duration, termination conditions).
3. **environment_config**: Environment-specific configuration using scene_name.
4. **agent_config**: Agent types, counts, profiles, and relationships.
5. **metrics_config**: Metrics collection configuration aligned with experiment needs.


Format your response as JSON with the following structure:
{
  "scenario_info": {
    "scene_name": "cross_department_collaboration",
    "description": "Multi-agent simulation of inter-department collaboration and resource allocation impact on project efficiency",
    "created_date": "2025-08-10"
  },
  "simulation_config": {
    "mode": "ROUND",
    "max_steps": 30
  },
  "environment_config": {
    "data": {
      "departments": [
        {
          "name": "department_A",
          "size": 25,
          "skill_focus": "development"
        },
        {
          "name": "department_B",
          "size": 25,
          "skill_focus": "design"
        }
      ],
      "projects": [
        {
          "name": "project_alpha",
          "required_skills": ["development", "design"],
          "deadline_rounds": 12
        },
        {
          "name": "project_beta",
          "required_skills": ["development", "quality_assurance"],
          "deadline_rounds": 15
        }
      ],
      "resources": [
        {
          "type": "budget",
          "total_amount": 100000
        },
        {
          "type": "equipment",
          "total_units": 50
        }
      ]
    }
  },
  "agent_config": {
    "total_count": 60, //no more than 100
    "types": {
      "Developer": {
        "count": 25,
        "profile_path": "profile/data/Developer.json",
        "schema_path": "profile/schema/Developer.json"
      },
      "Designer": {
        "count": 25,
        "profile_path": "profile/data/Designer.json",
        "schema_path": "profile/schema/Designer.json"
      },
      "QAEngineer": {
        "count": 10,
        "profile_path": "profile/data/QAEngineer.json",
        "schema_path": "profile/schema/QAEngineer.json"
      }
    }
  },
  "metric_config": {
    "metrics": [
      {
        "name": "project_completion_rate",
        "type": "percentage",
        "description": "Percentage of projects completed before deadline"
      },
      {
        "name": "average_resource_utilization",
        "type": "efficiency",
        "description": "Average percentage of allocated resources utilized"
      },
      {
        "name": "collaboration_intensity",
        "type": "network",
        "description": "Measures frequency and diversity of cross-department collaborations"
      }
    ]
  }
}
Please note that the above JSON structure is just an example and has no relation to the content you are to generate. You need to create the corresponding experimental configuration based on the research question and scenario information.
"""

    def _construct_user_prompt(self, research_question: str, scene_information: str, exp_information: str, scene_name: str) -> str:
        """
        Constructs the user-specific prompt based on the research question and scene description.

        Args:
            research_question (str): The research question to guide scenario generation.
            scene_information (str): Background or contextual information relevant to the scenario.

        Returns:
            str: A user prompt embedding both the research question and scene information.
        """
        return f"""Please design a simulation scenario configuration based on the research question, scene information and experiment config.

IMPORTANT: Use scene_name: "{scene_name}" consistently throughout the configuration.
Focus on simulation configuration details, NOT experimental design.

research question: {research_question}
scene information: {scene_information}
exp_config: {exp_information}
scene_name: {scene_name}

Ensure content alignment: metrics should correspond to those in exp_config, agent types should align with experiment needs.
Note that the JSON you generate must not contain any comments.
Remember to format your response as JSON with the structure specified in your instructions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to generate a structured simulation experiment scenario.

        Args:
            input_data (Dict[str, Any]): Input dictionary containing:
                - research_question (str): The specific research question
                - scene_information (str): The contextual information of the simulation scene

        Returns:
            Dict[str, Any]: A dictionary containing the generated 'experiments_config'

        Raises:
            ValueError: If required fields are missing or the response cannot be parsed as JSON.
        """
        research_question = input_data.get("research_question", "")
        scene_information = input_data.get("scene_information", "")
        exp_information = input_data.get('exp_config', "")
        scene_name = input_data.get('scene_name', "")
        print("*" * 20)
        print("exp_information:", exp_information)
        print("scene_name:", scene_name)

        if not research_question or not scene_information or not exp_information or not scene_name:
            raise ValueError("research_question, scene_information, exp_config, and scene_name are all required.")

        # Construct prompts for the language model
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(research_question, scene_information, exp_information, scene_name)

        # Debug: Print constructed prompts
        print("=" * 20)
        print("System Prompt:")
        print(system_prompt)
        print("=" * 20)
        print("User Prompt:")
        print(user_prompt)
        print("=" * 20)

        # Use the language model to generate the scenario
        response = self.generate_response(system_prompt, user_prompt)
        print("Response from Agent:")
        print(response)
        print("=" * 20)

        # Try to parse the JSON response
        try:
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            # Attempt to extract JSON content from common formatting blocks
            try:
                if "```json" in response:
                    json_content = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_content = response.split("```")[1].strip()
                else:
                    json_content = response

                parsed_response = json.loads(json_content)
                return parsed_response
            except (json.JSONDecodeError, IndexError):
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...")
