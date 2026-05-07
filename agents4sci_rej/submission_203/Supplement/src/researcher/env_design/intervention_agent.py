"""
Intervention Agent for generating detailed simulation experiment interventions.

This module defines the InterventionAgent class, which is responsible for
generating fine-grained intervention configurations in structured simulation experiments.
It leverages large language models (LLMs) to synthesize interventions based on
coarse-grained intervention descriptions and contextual scene information.

The output conforms to a predefined JSON schema, including both profile-level and runtime interventions.
"""

import json
from typing import Dict, Any

from .agent_base import AgentBase


class InterventionAgent(AgentBase):
    """
    Agent for generating detailed intervention configurations for simulation experiments.

    This agent takes as input high-level (coarse) intervention descriptions and rich
    scene context, and produces structured, fine-grained intervention configurations,
    including profile modifications, runtime interventions, and target selection strategies,
    following a standardized JSON schema.

    Attributes:
        model_config (str): Optional. The identifier for the LLM model configuration.
    """

    def __init__(self, model_config=None):
        """
        Initialize the intervention agent with an optional model configuration.

        Args:
            model_config (str, optional): Configuration name of the LLM to be used.
                If None, a default model configuration is applied.
        """
        super().__init__("Inspiration Agent", model_config)

    def _construct_system_prompt(self) -> str:
        """
        Build the system prompt that defines the agent's role and output schema.

        Returns:
            str: A system instruction string that guides the LLM in generating
                 structured intervention data in JSON format.
        """
        return """You are an expert social science researcher specializing in multi-agent simulations.
Your task is to generate detailed intervention information based on the scene information, research questions, experiments config.
"profile_modifications" first selects the agent profiles that need to be modified, and then adjusts the specific content according to the differences between the treatment group and the control group.
Format your response as JSON with the following structure:
{
  "treatment_groups": {
    "treatment_group_001": {
      "profile_modifications": [
        {
          "target_selection": {
            "method": "by_agent_type",
            "agent_types": [
              "CulturalAgent"
            ],
            "percentage": 0.6
          },
          "modifications": [
            {
              "field": "personality_trait",
              "modification_type": "set_value",
              "value": "open",
              "validation": {
                "type": "str"
              }
            },
            {
              "field": "new_cultural_openness",
              "modification_type": "add_field",
              "value": 0.8,
              "validation": {
                "min": 0.0,
                "max": 1.0,
                "type": "float"
              },
              "field_config": {
                "type": "float",
                "default": 0.5,
                "private": false,
                "sampling": "random",
                "range": [
                  0.0,
                  1.0
                ],
                "description": "Openness to adopting new cultural traits"
              }
            }
          ],
          "llm_prompt_template": "Modify this agent to be more culturally open and curious. Enhance their willingness to explore and adopt new cultural experiences while maintaining personality coherence."
        }
      ]
    },
    "treatment_group_002": {
      "profile_modifications": [
        {
          "target_selection": {
            "method": "by_profile_criteria",
            "criteria": {
              "personality_trait": {
                "type": "in_list",
                "value": [
                  "social",
                  "open"
                ]
              }
            },
            "percentage": 1.0
          },
          "modifications": [
            {
              "field": "social_influence_power",
              "modification_type": "add_field",
              "value": 0.9,
              "validation": {
                "min": 0.0,
                "max": 1.0,
                "type": "float"
              },
              "field_config": {
                "type": "float",
                "default": 0.5,
                "private": false,
                "sampling": "random",
                "range": [
                  0.0,
                  1.0
                ],
                "description": "Power to influence others' cultural choices"
              }
            },
            {
              "field": "communication_effectiveness",
              "modification_type": "add_field",
              "value": 0.85,
              "validation": {
                "min": 0.0,
                "max": 1.0,
                "type": "float"
              },
              "field_config": {
                "type": "float",
                "default": 0.5,
                "private": false,
                "sampling": "random",
                "range": [
                  0.0,
                  1.0
                ],
                "description": "Effectiveness in cultural communication"
              }
            }
          ],
          "llm_prompt_template": "Transform this agent into a cultural connector who can effectively influence others' cultural choices. Enhance their social influence and communication abilities while maintaining their core personality."
        }
      ]
    },
    "treatment_group_003": {
      "runtime_modifications": [
        {
          "intervention_type": "broadcast",
          "target_type": "agent",
          "timing": {
            "type": "scheduled",
            "schedule": [
              1,
              2
            ]
          },
          "target_selection": {
            "method": "all"
          },
          "config": {
            "message": "A major cultural festival is happening! Consider exploring new cultural experiences.",
            "message_type": "cultural_event",
            "message_data": {
              "event_type": "cultural_festival",
              "influence_strength": 0.3,
              "duration": 3
            }
          }
        },
        {
          "intervention_type": "attribute_modification",
          "target_type": "agent",
          "timing": {
            "type": "conditional",
            "condition": "step > 15"
          },
          "target_selection": {
            "method": "random_sample",
            "percentage": 0.4
          },
          "config": {
            "modifications": [
              {
                "field": "music_preference",
                "modification_type": "set_value",
                "value": "Electronic"
              }
            ]
          }
        },
        {
          "intervention_type": "attribute_modification",
          "target_type": "agent",
          "timing": {
            "type": "scheduled",
            "schedule": [
              12,
              22
            ]
          },
          "target_selection": {
            "method": "by_profile_criteria",
            "criteria": {
              "personality_trait": {
                "type": "equals",
                "value": "social"
              }
            }
          },
          "config": {
            "modifications": [
              {
                "field": "social_influence_power",
                "modification_type": "set_value",
                "value": 1.0
              }
            ]
          }
        },
        {
          "intervention_type": "attribute_modification",
          "target_type": "environment",
          "timing": {
            "type": "scheduled",
            "schedule": [
              25
            ]
          },
          "target_selection": {
            "method": "environment_global"
          },
          "config": {
            "modifications": [
              {
                "field": "cultural_climate",
                "modification_type": "set_value",
                "value": "progressive"
              },
              {
                "field": "openness_factor",
                "modification_type": "set_value",
                "value": 1.2
              }
            ]
          }
        }
      ]
    }
  }
}
Note that the JSON you generate must not contain any comments.
Please note that the above JSON structure is just an example and has no relation to the content you are to generate. You need to create the corresponding experimental configuration based on the research question and scenario information.
"""

    def _construct_user_prompt(self, research_question: str, scene_information: str, exp_config: str) -> str:
        """
        Construct the user-facing prompt by combining coarse interventions and scene details.

        Args:
            scene_information (str): Detailed context of the simulation scenario.

        Returns:
            str: A formatted prompt string for LLM input.
        """
        return f"""Please generate detailed intervention information based on the coarse-grained interventions and scene information.

        research question: {research_question}
        scene information: {scene_information}
        experiments_config: {exp_config}

        Remember to format your response as JSON with the structure specified in your instructions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured intervention configuration based on input context.

        Args:
            input_data (Dict[str, Any]): A dictionary containing:
                - interventions (str): High-level intervention descriptions.
                - scene_information (str): Contextual details of the simulation scene.

        Returns:
            Dict[str, Any]: A dictionary containing the key 'experiments_config',
                            which holds the parsed intervention configuration.

        Raises:
            ValueError: If required fields are missing or JSON parsing fails.
        """
        research_question = input_data.get("research_question", "")
        exp_config = input_data.get("research_question", "")
        scene_information = input_data.get("exp_config", "")

        if not scene_information or not research_question or not exp_config:
            raise ValueError("Both 'research_question' and 'exp_config' and 'scene_information' are required.")

        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(research_question, scene_information, exp_config)

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

        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            try:
                if "```json" in response:
                    json_content = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_content = response.split("```")[1].strip()
                else:
                    json_content = response
                parsed_response = json.loads(json_content)
            except (json.JSONDecodeError, IndexError):
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...")

        return parsed_response
        
