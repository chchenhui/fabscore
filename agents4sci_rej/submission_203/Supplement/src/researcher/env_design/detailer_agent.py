"""
Detailer Agent for elaborating selected research questions and simulation scenarios.

This module provides the DetailerAgent class, which is responsible for
taking a selected research question and simulation scenario and developing it
into an ODD protocol for implementation.
"""

import json
import re
from typing import Dict, Any, List, Optional

from .agent_base import AgentBase
from loguru import logger
import os 

class DetailerAgent(AgentBase):
    """
    Agent responsible for elaborating selected research scenarios into ODD protocol.
    
    The detailer agent takes a selected research question and simulation scenario
    and generates an ODD (Overview, Design concepts, Details) protocol
    compatible with the OneSim framework.
    """
    
    def __init__(self, model_config=None):
        """
        Initialize the detailer agent.
        
        Args:
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        super().__init__("Detailer Agent", model_config)
        
        # Initialize the scene_info with a flexible structure
        self.scene_info = {
            "domain": "",
            "scene_name": "",
            "odd_protocol": {
                "overview": {},
                "design_concepts": {},
                "details": {}
            }
        }
        
        # Define recommended keys for each section (for guidance only)
        self.recommended_keys = {
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
        Construct the system prompt for the detailer agent.
        
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
        selected_scenario: Dict[str, Any], 
        evaluations: List[Dict[str, Any]],
        original_topic: str
    ) -> str:
        """
        Construct the user prompt for the detailer agent.
        
        Args:
            selected_scenario (Dict[str, Any]): The selected research scenario.
            evaluations (List[Dict[str, Any]]): The evaluations of all scenarios.
            original_topic (str): The original research topic.
            
        Returns:
            str: The user prompt.
        """
        # Extract relevant evaluation for the selected scenario
        scenario_id = selected_scenario.get("id", 0)
        relevant_eval = None
        
        for eval_item in evaluations:
            if eval_item.get("question_id") == scenario_id:
                relevant_eval = eval_item
                break
        
        # Format the selected scenario and evaluation as a string
        scenario_str = json.dumps(selected_scenario, indent=2)
        eval_str = json.dumps(relevant_eval, indent=2) if relevant_eval else "No specific evaluation available."
        
        return f"""Please develop the following selected research question and simulation scenario
into an ODD protocol for a multi-agent simulation:

ORIGINAL RESEARCH TOPIC: {original_topic}

SELECTED RESEARCH QUESTION AND SCENARIO:
{scenario_str}

EVALUATION AND SELECTION RATIONALE:
{eval_str}

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
            input_data (Dict[str, Any]): Input data containing the selected scenario,
                evaluations, and original topic.
            
        Returns:
            Dict[str, Any]: Output data containing the ODD protocol.
        """
        # Extract selected scenario, evaluations, and original topic
        selected_scenario = input_data.get("selected_scenario_details", {})
        evaluations = input_data.get("evaluations", [])
        original_topic = input_data.get("original_topic", "")
        
        if not selected_scenario:
            raise ValueError("Selected scenario is required.")
        
        # Generate response
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(selected_scenario, evaluations, original_topic)
        
        response = self.generate_response(system_prompt, user_prompt)

        print("=" * 20)
        print("System Prompt:")
        print(system_prompt)
        print("=" * 20)
        print("User Prompt:")
        print(user_prompt)
        print("=" * 20)
        print("Response from Agent:")
        print(response)
        print("=" * 20)
        
        # Extract JSON from response
        try:
            # Try to extract JSON block from response if enclosed in ```json blocks
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
                scene_info = json.loads(json_str)
            else:
                # Try to parse entire response as JSON
                scene_info = json.loads(response.strip())
            
            # Validate scene_info structure
            if not isinstance(scene_info, dict):
                raise ValueError("Scene info must be a dictionary")
                
            # Ensure all required fields are present
            required_fields = ["domain", "scene_name", "odd_protocol"]
            for field in required_fields:
                if field not in scene_info:
                    scene_info[field] = ""
                    
            # Ensure odd_protocol has all required sections
            if "odd_protocol" not in scene_info or not isinstance(scene_info["odd_protocol"], dict):
                scene_info["odd_protocol"] = {}
                
            for section in ["overview", "design_concepts", "details"]:
                if section not in scene_info["odd_protocol"] or not isinstance(scene_info["odd_protocol"][section], dict):
                    scene_info["odd_protocol"][section] = {}
                    
                    # Fill in minimal structure with recommended keys
                    for key in self.recommended_keys.get(section, []):
                        scene_info["odd_protocol"][section][key] = f"No information available about {key.replace('_', ' ')}"
            
            # Store scene_info
            self.scene_info = scene_info
            
        except Exception as e:
            logger.error(f"Error parsing ODD protocol: {str(e)}")
            # Create a minimal scene_info with basic structure
            self.scene_info = {
                "domain": "Sociology",  # Default domain
                "scene_name": self._generate_fallback_scene_name(original_topic),
                "odd_protocol": {
                    "overview": {},
                    "design_concepts": {},
                    "details": {}
                }
            }
            
            # Fill in minimal structure with recommended keys
            for section in ["overview", "design_concepts", "details"]:
                for key in self.recommended_keys.get(section, []):
                    self.scene_info["odd_protocol"][section][key] = f"No information available about {key.replace('_', ' ')}"
        
        # Include the detailed specification in the result for backward compatibility
        return {
            "scene_info": self.scene_info,
            "domain": self.scene_info.get("domain", ""),
            "scene_name": self.scene_info.get("scene_name", ""),
            "odd_protocol": self.scene_info.get("odd_protocol", {}),
            "research_question": selected_scenario.get("question", ""),
            "original_topic": original_topic
        }
        
    def _generate_fallback_scene_name(self, topic: str) -> str:
        """
        Generate a concise, meaningful scene name from the research topic.
        
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
        core_words = []
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
        Get the scene info.
        
        Returns:
            Dict[str, Any]: The scene info.
        """
        return self.scene_info
    
    def save_scene_info(self, env_path: str) -> None:
        """
        Save the scene_info to the scene_info.json file.
        
        Args:
            env_path (str): Path to the environment directory.
        """
        scene_info_path = os.path.join(env_path, "scene_info.json")
        
        # Load existing scene_info.json if it exists, or create a new dict
        existing_scene_info = {}
        if os.path.exists(scene_info_path):
            try:
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    existing_scene_info = json.load(f)
            except Exception as e:
                logger.error(f"Error loading scene_info.json: {e}")
        
        # Update scene_info with our data
        # First copy domain and scene_name
        logger.info(f"Saving scene_info to {scene_info_path}")
        logger.info(f"Scene info: {self.scene_info}")
        existing_scene_info["domain"] = self.scene_info["domain"]
        existing_scene_info["scene_name"] = env_path.split(os.sep)[-1] if env_path else self.scene_info["scene_name"]
        logger.info(f"Existing scene info: {existing_scene_info}")
        
        # Then update odd_protocol
        existing_scene_info["odd_protocol"] = self.scene_info["odd_protocol"]
        
        # Write back to file
        try:
            with open(scene_info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_scene_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Scene info (including ODD protocol) saved to {scene_info_path}")
        except Exception as e:
            logger.error(f"Error saving scene_info to {scene_info_path}: {e}") 