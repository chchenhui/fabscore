"""
Detailer Agent for elaborating selected research questions and simulation scenarios.

This module provides the DetailerAgent class, which is responsible for
taking a selected research question and simulation scenario and developing it
into a comprehensive ODD protocol for implementation.
"""

import json
import re
from typing import Dict, Any, List, Optional

from .agent_base import AgentBase
from loguru import logger
import os 

class DetailerAgent(AgentBase):
    """
    Agent responsible for elaborating selected research scenarios into an ODD protocol.
    
    The detailer agent takes a selected research question and simulation scenario
    and generates a detailed ODD (Overview, Design concepts, Details) protocol
    compatible with the OneSim framework.
    """
    
    def __init__(self, model_config: Optional[str] = None):
        """
        Initialize the detailer agent.
        
        Args:
            model_config (Optional[str]): The model configuration name to use.
                If None, the default model will be used.
        """
        super().__init__("Detailer Agent", model_config)
        
        # Initialize the scene_info with a flexible structure for storing the ODD protocol
        self.scene_info: Dict[str, Any] = {
            "domain": "",
            "scene_name": "",
            "odd_protocol": {
                "overview": {},
                "design_concepts": {},
                "details": {}
            }
        }
        
        # Define recommended keys for each section to guide the LLM's output
        self.recommended_keys: Dict[str, List[str]] = {
            "overview": [
                "system_goal", 
                "agent_types", 
                "environment_description"
            ],
            "design_concepts": [
                "interaction_patterns", 
                "communication_protocols", 
                "decision_mechanisms"
            ],
            "details": [
                "agent_behaviors", 
                "decision_algorithms", 
                "specific_constraints"
            ]
        }
    
    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt that instructs the LLM on its role and output format.
        
        Returns:
            str: The system prompt.
        """
        return """You are an expert in designing multi-agent systems and generating ODD (Overview, Design concepts, Details) protocols.
Your task is to elaborate a selected research question and scenario into an ODD protocol for a multi-agent simulation.

Based on the research question and scenario, you need to:

1. Determine the most appropriate DOMAIN for this scenario from these options ONLY:
   - Economics
   - Sociology
   - Politics
   - Psychology
   - Organization
   - Demographics
   - Law
   - Communication

2. Generate a concise, descriptive SCENE_NAME in snake_case format (lowercase with underscores) that clearly identifies this scenario.
   The scene_name should be 3-5 words maximum, focusing on the core phenomenon or domain being studied.
   Examples: "social_influence_dynamics", "market_competition", "group_decision_making", "rumor_propagation".

3. Create a complete ODD protocol with these sections:
   
   a) OVERVIEW:
      - system_goal: What the multi-agent system aims to achieve
      - agent_types: Description of different agent types
      - environment_description: The environment agents operate in
      
   b) DESIGN_CONCEPTS:
      - interaction_patterns: How agents interact with each other
      - communication_protocols: Communication mechanisms between agents
      - decision_mechanisms: How decisions are made
      
   c) DETAILS:
      - agent_behaviors: Specific behaviors of each agent type
      - decision_algorithms: Algorithms used for decision-making
      - specific_constraints: Constraints affecting the system

CRITICAL FORMATTING INSTRUCTIONS:
1. ALL values in the odd_protocol must be simple strings only - no nested objects or arrays
2. For multiple agent types or behaviors, describe them all within a single comprehensive string
3. Create meaningful descriptive field names instead of generic names like "additional_field1"
4. Only include fields that contain actual information - don't create empty placeholders
5. You may add any number of fields to each section as appropriate for the specific scenario

Your output must be a valid JSON object with this exact structure:
{
    "domain": "ONE of the eight domains listed above",
    "scene_name": "descriptive_name_in_snake_case",
    "odd_protocol": {
        "overview": {
            "system_goal": "Comprehensive description as a single string",
            "agent_types": "All agent types described in a single string",
            "environment_description": "Complete environment description as a string",
            "meaningful_field_name": "Additional relevant information as a string"
        },
        "design_concepts": {
            "interaction_patterns": "Complete interaction patterns as a single string",
            "communication_protocols": "All communication protocols as a single string",
            "decision_mechanisms": "Decision mechanisms described in one string",
            "meaningful_field_name": "Additional relevant information as a string"
        },
        "details": {
            "agent_behaviors": "All agent behaviors described in a single string",
            "decision_algorithms": "All decision algorithms in a single string",
            "specific_constraints": "All constraints described in one string",
            "meaningful_field_name": "Additional relevant information as a string"
        }
    }
}

Be specific, detailed, and comprehensive in your descriptions while maintaining a clear structure.
All descriptions must be plain strings - not objects or arrays.
Ensure all aspects of the research question are addressed in the protocol.
"""
    
    def _construct_user_prompt(
        self, 
        scenario: Dict[str, Any], 
    ) -> str:
        """
        Construct the user prompt containing the research scenario to be processed.
        
        Args:
            scenario (Dict[str, Any]): The research scenario dictionary.
            
        Returns:
            str: The user prompt to be sent to the LLM.
        """
    
        # Format the selected scenario as a JSON string for the prompt
        scenario_str = json.dumps(scenario, indent=2)
        return f"""Please develop the following selected research question and simulation scenario
into an ODD protocol for a multi-agent simulation:

SELECTED RESEARCH QUESTION AND SCENARIO:
{scenario_str}

Based on this information, please create a comprehensive ODD protocol according to the structure 
specified in the instructions. The protocol should be detailed enough to guide the implementation
of a multi-agent simulation.

Make sure to:
1. Choose an appropriate domain from the provided options
2. Generate a descriptive scene_name in snake_case format that clearly identifies this scenario
3. Provide detailed information for all required sections of the ODD protocol
4. Add any additional fields necessary for this specific scenario

Return ONLY a valid JSON object with the specified structure.
"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and elaborate the selected scenario into an ODD protocol.
        
        Args:
            input_data (Dict[str, Any]): Input data containing the selected scenario and original topic.
            
        Returns:
            Dict[str, Any]: A dictionary containing the generated ODD protocol and related info.
        """
        # Extract the selected scenario and original topic from input data
        research_questions = input_data.get("research_questions", {})
        
        
        # Generate LLM response based on system and user prompts
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(research_questions)
        response = self.generate_response(system_prompt, user_prompt)

        # Log prompts and response for debugging purposes
        logger.info("=" * 20)
        logger.info("System Prompt:\n" + system_prompt)
        logger.info("=" * 20)
        logger.info("User Prompt:\n" + user_prompt)
        logger.info("=" * 20)
        logger.info("Response from Agent:\n" + response)
        logger.info("=" * 20)
        
        # Attempt to parse and validate the JSON from the LLM's response
        try:
            # Try to extract JSON from a markdown code block first
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
                scene_info = json.loads(json_str)
            else:
                # Fallback: Try to parse the entire response as a JSON object
                scene_info = json.loads(response.strip())
            
            # Validate the structure of the parsed JSON
            if not isinstance(scene_info, dict):
                raise ValueError("Parsed scene info must be a dictionary.")
                
            required_fields = ["domain", "scene_name", "odd_protocol"]
            for field in required_fields:
                if field not in scene_info:
                    # In a robust system, you might raise an error here
                    scene_info[field] = ""
                    
            if not isinstance(scene_info.get("odd_protocol"), dict):
                scene_info["odd_protocol"] = {}
                
            for section in ["overview", "design_concepts", "details"]:
                if not isinstance(scene_info["odd_protocol"].get(section), dict):
                    scene_info["odd_protocol"][section] = {}
                    
                    # Fill in placeholder values for missing required keys
                    for key in self.recommended_keys.get(section, []):
                        scene_info["odd_protocol"][section][key] = f"No information available about {key.replace('_', ' ')}"
            
            # Store the final validated scene information
            self.scene_info = scene_info
            
        except Exception as e:
            # Log the parsing error and create a fallback dictionary
            logger.error(f"Error parsing ODD protocol: {str(e)}")
            raise ValueError(f"Failed to generate a valid ODD protocol: {e}")

        # Return the final output, combining the generated ODD with original context
        return {
            "scene_info": self.scene_info,
            "domain": self.scene_info.get("domain", ""),
            "scene_name": self.scene_info.get("scene_name", ""),
            "scene_information": self.scene_info.get("odd_protocol", {}),
        }
        
    def _generate_fallback_scene_name(self, topic: str) -> str:
        """
        Generate a concise, meaningful scene name from the research topic as a fallback.
        
        Args:
            topic (str): The research topic.
            
        Returns:
            str: A concise snake_case scene name (3-5 words max).
        """
        # Keywords that indicate common research domains/phenomena
        domain_keywords = {
            'social': ['social', 'interaction', 'influence', 'network', 'group'],
            'market': ['market', 'economic', 'trade', 'competition', 'price'],
            'decision': ['decision', 'choice', 'voting', 'consensus'],
            'information': ['information', 'rumor', 'news', 'communication'],
            'cultural': ['cultural', 'culture', 'norm', 'tradition'],
            'opinion': ['opinion', 'belief', 'attitude', 'spiral'],
            'cooperation': ['cooperation', 'collaboration', 'trust'],
            'conflict': ['conflict', 'competition', 'dispute'],
            'learning': ['learning', 'adaptation', 'evolution']
        }
        
        # Extract key words from topic
        topic_lower = topic.lower()
        words = re.findall(r'\b\w+\b', topic_lower)
        
        # Find domain matches
        matched_domain = None
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                matched_domain = domain
                break
        
        # Generate core name based on key concepts
        core_words: List[str] = []
        if matched_domain:
            core_words.append(matched_domain)
        
        # Add specific phenomena terms
        phenomena_terms = ['dynamics', 'formation', 'propagation', 'emergence', 'diffusion']
        for word in words:
            if word in phenomena_terms and len(core_words) < 3:
                core_words.append(word)
                break
        
        # If no good match, use first 2-3 meaningful words from topic
        if len(core_words) < 2:
            meaningful_words = [w for w in words if len(w) > 3 and w not in ['study', 'research', 'analysis', 'investigation']]
            core_words.extend(meaningful_words[:3-len(core_words)])
        
        # Ensure we have at least some words
        if not core_words:
            # Fallback: use first few words from topic
            core_words = words[:3] if words else ['research', 'simulation']
        
        # Create scene name (max 3 words to keep it concise)
        scene_name = '_'.join(core_words[:3])
        
        # Clean up the name
        scene_name = re.sub(r'[^a-zA-Z0-9]', '_', scene_name)
        scene_name = re.sub(r'_+', '_', scene_name)
        scene_name = scene_name.strip('_')
        
        # Ensure reasonable length (max 30 chars for readability)
        if len(scene_name) > 30:
            scene_name = scene_name[:30].strip('_')
        
        # Final fallback
        if not scene_name:
            scene_name = "research_simulation"
            
        return scene_name
    
    def get_scene_info(self) -> Dict[str, Any]:
        """
        Get the scene info generated by the agent.
        
        Returns:
            Dict[str, Any]: The scene info dictionary.
        """
        return self.scene_info
    
    def save_scene_info(self, env_path: str) -> None:
        """
        Save the generated scene_info to the scene_info.json file.
        
        Args:
            env_path (str): Path to the environment directory.
        """
        scene_info_path = os.path.join(env_path, "scene_info.json")
        
        # Load existing scene_info.json if it exists, or initialize an empty dictionary
        existing_scene_info = {}
        if os.path.exists(scene_info_path):
            try:
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    existing_scene_info = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing scene_info.json from {scene_info_path}: {e}")
        
        # Update existing scene_info with the newly generated data
        logger.info(f"Saving new ODD protocol to {scene_info_path}")
        
        # Use the directory name as the scene_name for consistency
        dir_scene_name = os.path.basename(env_path)
        
        # Overwrite top-level fields with new data
        existing_scene_info["domain"] = self.scene_info.get("domain", "")
        existing_scene_info["scene_name"] = dir_scene_name
        
        # Overwrite the odd_protocol section
        existing_scene_info["odd_protocol"] = self.scene_info.get("odd_protocol", {})
        
        # Write the updated dictionary back to the file
        try:
            with open(scene_info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_scene_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Scene info (including ODD protocol) saved to {scene_info_path}")
        except Exception as e:
            logger.error(f"Error saving scene_info to {scene_info_path}: {e}")