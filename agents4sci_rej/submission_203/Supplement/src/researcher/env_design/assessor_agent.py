"""
Assessor Agent for evaluating the quality of ODD protocol conversion.

This module provides the AssessorAgent class, which is responsible for
assessing the quality of the conversion from a brief description to a
complete ODD protocol based on four metrics.
"""

import json
from typing import Dict, Any, List

from .agent_base import AgentBase


class AssessorAgent(AgentBase):
    """
    Agent responsible for assessing ODD protocol quality.
    
    The assessor agent evaluates the quality of the conversion from
    a brief research description to a complete ODD protocol based on
    four metrics: relevance, fidelity, feasibility, and significance.
    """
    
    def __init__(self, model_config=None):
        """
        Initialize the assessor agent.
        
        Args:
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        super().__init__("Assessor Agent", model_config)
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt for the assessor agent.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert evaluator of agent-based social simulation designs.
Your task is to assess the quality of the conversion from a brief research topic to a complete ODD protocol.

Evaluate the ODD protocol based on the following metrics (score 1-5 for each):

1. Relevance (1-5): Assess whether the simulation design accurately corresponds to the original research topic.
   This includes the clarity of research question definition and alignment with user intent.
   - Score 1: Design completely misses the original research direction
   - Score 5: Design perfectly captures the research direction with clear questions

2. Fidelity (1-5): Assess whether the model design (agent types, attributes, interaction mechanisms) 
   accurately reflects the essential characteristics of the social phenomenon being studied.
   This includes integration of relevant domain theories and knowledge.
   - Score 1: Model design bears little resemblance to the phenomenon being studied
   - Score 5: Model design captures the essence of the phenomenon with strong theoretical basis

3. Feasibility (1-5): Assess the technical feasibility of implementing the design.
   This includes the reasonableness of parameter settings, computational resource requirements,
   and practicality of data requirements.
   - Score 1: Design would be extremely difficult or impossible to implement
   - Score 5: Design is straightforward to implement with reasonable resources

4. Significance (1-5): Assess the research value of the generated questions and simulation design.
   This includes whether the design can provide new insights or solve existing problems in the field.
   - Score 1: Research has minimal value or replicates existing knowledge
   - Score 5: Research addresses important questions with potential for significant insights

For each metric, provide:
1. A score (1-5)
2. A brief justification for the score
3. Suggestions for improvement (especially for lower scores)

Also calculate an overall score (sum of all four metrics, maximum 20).

Format your response as JSON with the following structure:
{
  "relevance": {
    "score": 4,
    "justification": "Detailed justification for the score",
    "suggestions": "Suggestions for improvement",
    "summary": "Brief one-line summary"
  },
  "fidelity": {
    "score": 3,
    "justification": "Detailed justification for the score",
    "suggestions": "Suggestions for improvement",
    "summary": "Brief one-line summary"
  },
  "feasibility": {
    "score": 5,
    "justification": "Detailed justification for the score",
    "suggestions": "Suggestions for improvement",
    "summary": "Brief one-line summary"
  },
  "significance": {
    "score": 4,
    "justification": "Detailed justification for the score",
    "suggestions": "Suggestions for improvement",
    "summary": "Brief one-line summary"
  },
  "overall_score": 4,
  "overall_assessment": "Brief overall assessment of the ODD protocol quality"
}

Be critical but constructive in your assessment. Focus on providing actionable feedback."""
    
    def _construct_user_prompt(self, original_topic: str, odd_protocol: Dict[str, Any], scene_name: str) -> str:
        """
        Construct the user prompt for the assessor agent.
        
        Args:
            original_topic (str): The original research topic.
            odd_protocol (Dict[str, Any]): The ODD protocol to assess.
            scene_name (str): The name of the scene.
            
        Returns:
            str: The user prompt.
        """
        # Convert ODD protocol to a formatted string
        odd_protocol_str = json.dumps(odd_protocol, indent=2)
        
        return f"""Please assess the quality of the conversion from the original research topic to the ODD protocol.

ORIGINAL RESEARCH TOPIC:
{original_topic}

SCENE NAME:
{scene_name}

ODD PROTOCOL:
{odd_protocol_str}

Evaluate the quality of the conversion based on the four metrics in your instructions:
1. Relevance (1-5)
2. Fidelity (1-5)
3. Feasibility (1-5)
4. Significance (1-5)

For each metric, provide a score, justification, suggestions for improvement, and a brief one-line summary.
Also calculate an overall score (average of all four metrics, maximum 5).

Remember to format your response as JSON with the structure specified in your instructions."""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and assess the ODD protocol quality.
        
        Args:
            input_data (Dict[str, Any]): Input data containing the original topic,
                ODD protocol, and scene name.
            
        Returns:
            Dict[str, Any]: Output data containing the assessment results.
        """
        # Extract input data
        original_topic = input_data.get("original_topic", "")
        odd_protocol = input_data.get("odd_protocol", {})
        scene_name = input_data.get("scene_name", "")
        
        if not odd_protocol:
            raise ValueError("ODD protocol is required for assessment.")
        
        # Generate response
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(original_topic, odd_protocol, scene_name)
        
        response = self.generate_response(system_prompt, user_prompt)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response)
            return parsed_response
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
                return parsed_response
            except (json.JSONDecodeError, IndexError):
                # If still fails, return an error message
                raise ValueError(f"Failed to parse agent response as JSON. Response: {response[:200]}...") 