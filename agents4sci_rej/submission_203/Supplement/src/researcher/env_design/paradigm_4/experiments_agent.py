"""
Experiment Agent for generating structured simulation experiment configurations.

This module provides the ExperimentAgent class, which is responsible for generating
structured experiment configurations including scenarios, and analysis
plans based on research questions and scene information.
"""

import json
from typing import Dict, Any

from .agent_base import AgentBase


class Experimentagent(AgentBase):
    """
    Agent responsible for generating simulation experiment configurations.
    
    The experiment agent takes a research question and scene information to generate
    a structured experiment configuration that includes scenarios,
    metrics, and analysis settings following a predefined JSON schema.
    
    Attributes:
        model_config (str): Configuration name for the underlying language model
    """
    
    def __init__(self, model_config=None):
        """
        Initialize the experiment agent with specified model configuration.
        
        Args:
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        super().__init__("Experiment Agent", model_config)
        self.tool_list = [
                          "independent_t_test",
                          "paired_t_test",
                          "mann_whitney_u_test",
                          "wilcoxon_test",
                          "one_way_anova",
                          "kruskal_wallis_test",
                          "chi2_independence_test",
                          "fisher_exact_test",
                          "ks_2samp_test",
                          
                          "granger_causality",
                          "var_model",
                          "time_series_correlation",
                          "compute_trend",
                          
                          "bayes_factor_ttest",
                          "bayesian_simple_regression",
                          "bayesian_hierarchical_regression",
                          
                          "fit_mixedlm",
                          
                          "factorial_anova",
                          "manova",
                          
                          "repeated_measures_anova",
                          "friedman_test",
                          "ancova",
                          "multiple_comparisons"
                      ]
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt defining the agent's role and task requirements.
        
        Returns:
            str: System prompt with detailed instructions for experiment configuration generation
        """
        return """You are an expert social science researcher specializing in multi-agent simulations.
Your task is to design an experiment based on the scenario and corresponding scene information.
Focus on EXPERIMENTAL DESIGN rather than simulation configuration details.

1. **experiment_info**: Basic experiment information including name, title, description, research question, and creation date.
2. **scene_reference**: Reference to the scene (NOT detailed simulation configuration) - use scene_name consistently.
3. **experimental_design**: Design experimental groups including control groups, treatment groups, and replicates based on research variables.
4. **analysis_config**: Statistical analysis configuration including tests, confidence level, and comparison groups.

IMPORTANT: Do NOT include detailed simulation parameters (agent counts, steps, etc.) - these belong in scenario_config.json.
Format your response as JSON with the following structure:
{
  "experiment_info": {
    "name": "communication_effectiveness_study",
    "title": "Communication Effectiveness in Property Rights",
    "description": "Study on how communication affects property rights establishment",
    "created_date": "2025-08-10"
  },
  "scene_reference": {
    "scene_name": "mining_property_rights",
    "scene_path": "src/envs/mining_property_rights"
  },
  "experimental_design": {
    "control_group": {
      "name": "control_group_001",
      "description": "Baseline scenario without interventions",
      "parameters": {
        "random_seed": 12345
      }
    },
    "treatment_groups": [
      {
        "name": "treatment_group_001",
        "description": "Enhanced cultural openness intervention - adds new cultural traits",
        "parameters": {
          "random_seed": 12345
        }
      },
      {
        "name": "treatment_group_002",
        "description": "Social connector enhancement - targets social personality agents",
        "parameters": {
          "random_seed": 12345
        }
      },
      ...
    ],
    "replicates": [
      {
        "name": "replicate_001",
        "description": "Replicate of treatment_group_001 with different random seed",
        "base_treatment": "treatment_group_001",
        "parameters": {
          "random_seed": 54321
        }
      },
      ...
    ]
  },
  "analysis_config": {
    "statistical_tests": ["t_test", "anova",...],
    "confidence_level": 0.95,
    "comparison_groups": [
      {
        "name": "control_vs_treatment1",
        "groups": ["control_group_001", "treatment_group_001"]
      }
      ...
    ]
  }
}
Note that the JSON you generate must not contain any comments.
Please note that the above JSON structure is just an example and has no relation to the content you are to generate. You need to create the corresponding experimental configuration based on the research question and scenario information.
"""
    
    def _construct_user_prompt(self, scenario: str, scene_information : str, scene_name: str) -> str:
        """
        Construct the user prompt with specific research question and scene information.
        
        Args:
            research_question (str): Specific research question to base the experiment on
            scene_information (str): Contextual information about the simulation scenario
            
        Returns:
            str: Formatted user prompt with input parameters
        """
        return f"""Please generate an experiment configuration based on the research question and scene information.
For **statistical_tests**, the tools you can use are **{self.tool_list}**.

IMPORTANT: Use scene_name: "{scene_name}" in the scene_reference section.
Focus on experimental design, NOT simulation configuration details.

scenario: {scenario}
scene information: {scene_information}
scene_name: {scene_name}

Remember to format your response as JSON with the structure specified in your instructions."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to generate structured experiment configuration.
        
        Args:
            input_data (Dict[str, Any]): Dictionary containing:
                - research_question (str): Specific research question
                - scene_information (str): Contextual scene information
                
        Returns:
            Dict[str, Any]: Dictionary containing generated experiments_config
            
        Raises:
            ValueError: If required input fields are missing or JSON parsing fails
        """
        # Extract required parameters from input
        scenario = input_data.get("scenario", "")
        scene_information = input_data.get("scene_information", "")
        scene_name = input_data.get("scene_name", "")
        
            
        # Construct prompts for LLM interaction
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(scenario, scene_information, scene_name)

        # Debug output for prompt inspection
        
        # Generate response using configured model
        response = self.generate_response(system_prompt, user_prompt)
        print("Response from EXP Agent:")
        print(response)
        print("=" * 20)
        
        # Attempt to parse JSON response with fallback strategies
        try:
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            # Try alternative JSON extraction methods
            try:
                # Extract from code block formatting
                if "```json" in response:
                    json_content = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_content = response.split("```")[1].strip()
                else:
                    json_content = response
                
                parsed_response = json.loads(json_content)
                return parsed_response
            except (json.JSONDecodeError, IndexError):
                # Final failure case with error message
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...")