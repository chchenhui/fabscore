from typing import Optional, Any, List, Dict, Union
from loguru import logger
from onesim.models.core.message import Message
from .base import AgentBase
import json
import re
import os
from onesim.models import JsonBlockParser

class ODDAgent(AgentBase):
    """An intelligent agent for extracting and refining multi-agent system requirements."""

    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str = '',
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ODD (Overview, Design, Details) extraction agent.
        
        Arguments:
            name (str): The name of the agent.
            model_config_name (str): The name of the model config.
            sys_prompt (str): The system prompt for the agent.
            use_memory (bool): Whether the agent uses memory.
        """
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

        # Default system prompts
        if sys_prompt == '':
            self.sys_prompt = (
                "You are an intelligent assistant specializing in multi-agent system design. "
                "Your goal is to help users refine and clarify their multi-agent system requirements "
                "by asking intelligent, context-aware questions and making insightful inferences."
            )
            
        super().__init__(
            model_config_name=model_config_name,
            **kwargs
        )

    def call_llm(self, prompt: str, is_json: bool = False) -> Optional[Any]:
        """
        Call the LLM with the given prompt and return the response.
        
        Args:
            prompt (str): The input prompt for the LLM
            is_json_expected (bool): Whether JSON parsing is expected
        
        Returns:
            Optional response from the LLM
        """
        try:
            # Format the prompt
            if is_json:
                parser = JsonBlockParser()
                prompt+=parser.format_instruction
            formatted_prompt = self.model.format(
                Message("system",self.sys_prompt, role="system"),
                Message("user", prompt, role="user")
            )
            
            # Get LLM response
            response = self.model(formatted_prompt)
            
            # Parse JSON if expected
            if is_json:
                res = parser.parse(response)
                return res.parsed
            
            return response.text
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return None

    @staticmethod
    def odd_to_markdown(scene_info: Dict) -> str:
        """
        Convert the ODD protocol from scene_info to markdown format for display purposes.
        
        Args:
            scene_info (Dict): Scene info containing the ODD protocol
        
        Returns:
            str: Markdown representation of the ODD protocol
        """
        markdown = "# Multi-Agent System ODD Protocol\n\n"
        
        # Add domain information
        # if scene_info.get("domain"):
        #     markdown += f"## Domain\n- {scene_info['domain']}\n\n"
        
        # # Add scene name if available
        # if scene_info.get("scene_name"):
        #     markdown += f"## Scene Name\n- {scene_info['scene_name']}\n\n"
        
        # Get ODD protocol data
        odd_protocol = scene_info.get("odd_protocol", {})
        
        # Process each main section
        for section_name, section_title in [
            ("overview", "Overview"), 
            ("design_concepts", "Design Concepts"), 
            ("details", "Details")
        ]:
            section_data = odd_protocol.get(section_name, {})
            if section_data:
                markdown += f"## {section_title}\n"
                for key, value in section_data.items():
                    # Convert snake_case or camelCase to Title Case for display
                    display_key = ' '.join(word.capitalize() for word in re.sub(r'([a-z])([A-Z])', r'\1 \2', key).split('_'))
                    markdown += f"- {display_key}: {value}\n"
                markdown += "\n"
        
        return markdown

    def markdown_to_scene_info(self, markdown: str) -> Dict:
        """
        Convert a markdown representation of the ODD protocol to scene_info structure.
        This is used for backward compatibility or when parsing markdown input.
        
        Args:
            markdown (str): Markdown representation of ODD protocol
            
        Returns:
            Dict: scene_info structure containing ODD protocol
        """
        scene_info = {
            "domain": "",
            "scene_name": "",
            "odd_protocol": {
                "overview": {},
                "design_concepts": {},
                "details": {}
            }
        }
        
        # Extract domain if present
        domain_match = re.search(r"## Domain\s*\n\s*-\s*(.*?)(?:\n|$)", markdown)
        if domain_match:
            scene_info["domain"] = domain_match.group(1).strip()
            
        # Extract scene name if present
        scene_name_match = re.search(r"## Scene Name\s*\n\s*-\s*(.*?)(?:\n|$)", markdown)
        if scene_name_match:
            scene_info["scene_name"] = scene_name_match.group(1).strip()
        
        # Extract sections with a more generic approach
        section_patterns = {
            "overview": r"## Overview\s*\n(.*?)(?=##|\Z)",
            "design_concepts": r"## Design Concepts\s*\n(.*?)(?=##|\Z)",
            "details": r"## Details\s*\n(.*?)(?=##|\Z)"
        }
        
        for section_key, pattern in section_patterns.items():
            section_match = re.search(pattern, markdown, re.DOTALL)
            if section_match:
                section_content = section_match.group(1).strip()
                # Extract all key-value pairs in the section
                items = re.findall(r"-\s*(.*?):\s*(.*?)(?=\n-|\n##|\Z)", section_content + "\n", re.DOTALL)
                for key, value in items:
                    # Convert Title Case to snake_case for storage
                    storage_key = key.strip().lower().replace(' ', '_')
                    scene_info["odd_protocol"][section_key][storage_key] = value.strip()
        
        return scene_info

    def update_scene_info(self, user_input: str) -> dict:
        """
        Update the scene_info with user input using LLM.
        
        Args:
            user_input (str): User's description or input
        
        Returns:
            dict: Response containing updated scene_info and conversation status
        """
        # Convert current scene_info to markdown for LLM
        current_markdown = ODDAgent.odd_to_markdown(self.scene_info)
        
        # Prepare recommended keys as guidance
        recommended_keys_guidance = ""
        for section, keys in self.recommended_keys.items():
            section_title = section.replace("_", " ").title()
            recommended_keys_guidance += f"\nRecommended fields for {section_title}:\n"
            for key in keys:
                key_display = key.replace("_", " ").title()
                recommended_keys_guidance += f"- {key_display}: Description of {key_display.lower()}\n"
        
        # Prepare the update prompt
        # Improved update_prompt template
        update_prompt = """You are an expert assistant in generating and updating Multi-Agent System ODD (Overview, Design concepts, Details) protocols.

Task: Update the existing ODD protocol with new information from the user's input and determine if the conversation should conclude.

Existing ODD Protocol:
{existing_protocol}

User Input:
{user_input}

{recommended_keys_guidance}

Instructions:
1. Carefully review the existing ODD protocol and the user's input
2. CRITICALLY IMPORTANT: Extract and preserve ALL details and nuances from the user's input - do not summarize or omit any information
3. The updated ODD protocol must contain AT LEAST the same information content as the user's description, if not more
4. Incorporate all specific examples, edge cases, relationships, and technical details mentioned by the user
5. If the user mentions any implementation details, agent behaviors, or system characteristics, these MUST be preserved in the appropriate sections
6. Use the ODD structure to organize information, but prioritize completeness over brevity
7. When in doubt about where to place certain information, include it in the most relevant section(s)
8. Generate a scene_name in English with lowercase words connected by underscores, which describes the multi-agent system concisely
9. Determine which domain this multi-agent system belongs to from the following options only:
   - Economics
   - Sociology
   - Politics
   - Psychology
   - Organization
   - Demographics
   - Law
   - Communication
10. Consider the recommended fields as a guide, but create meaningful field names that reflect the actual content
11. CRITICAL: Each section (overview, design_concepts, details) must contain key-value pairs where values are ONLY strings, not nested objects or arrays

Output Format (JSON):
{{
    "domain": "ONE of the eight domains listed above that best matches the multi-agent system",
    "scene_name": "english_words_with_underscores_describing_the_scene",
    "odd_protocol": {{
        "overview": {{
            "system_goal": "Description of the overall goal of the multi-agent system",
            "agent_types": "Describe all types of agents in a single text string",
            "environment_description": "Description of the environment the agents operate in",
            "key1": "Additional information using a descriptive key name",
            "key2": "Additional information using a descriptive key name"
        }},
        "design_concepts": {{
            "interaction_patterns": "Description of how agents interact with each other",
            "communication_protocols": "Description of communication mechanisms between agents",
            "decision_mechanisms": "Description of how decisions are made in the system",
            "key1": "Additional information using a descriptive key name",
            "key2": "Additional information using a descriptive key name"
        }},
        "details": {{
            "agent_behaviors": "Description of specific behaviors of all agent types in a single text string",
            "decision_algorithms": "Description of algorithms used for decision-making in a single text string",
            "specific_constraints": "Description of constraints that affect the system",
            "key1": "Additional information using a descriptive key name",
            "key2": "Additional information using a descriptive key name"
        }}
    }},
    "end_conversation": boolean,
    "reason_for_end": "Brief explanation of why conversation might end"
}}

Important Requirements:
1. ALL values must be strings only - no nested objects, arrays, or complex structures
2. Do not use generic field names like "additional_field1" - create meaningful, descriptive field names
3. If information about multiple agents exists, combine it into a single descriptive string
4. Only include fields that contain actual information - don't create empty placeholders
5. You may add any number of fields to each section as appropriate, using descriptive field names

Guidelines for Information Preservation:
- Maintain high information density in each section
- Use specific and detailed language rather than general descriptions
- Include concrete examples and edge cases mentioned by the user
- Preserve numerical values, technical terms, and specialized concepts exactly as provided
- If the user provides step-by-step processes or sequences, maintain the full sequence with all steps
- Do not simplify complex relationships or interactions between system components

Evaluation Criteria for Ending Conversation:
- Has the user provided a comprehensive description of the multi-agent system?
- Does the input suggest satisfaction with the current protocol?
- Are all critical components of the multi-agent system design adequately addressed?
- Has ALL information from the user's descriptions been successfully incorporated?
- Look for both explicit and implicit signals about conversation completion
"""
        update_prompt = update_prompt.format(
            existing_protocol=current_markdown,
            user_input=user_input,
            recommended_keys_guidance=recommended_keys_guidance
        )
        
        # Call LLM to update the protocol
        update_response = self.call_llm(update_prompt, is_json=True)
        
        # Handle invalid response
        if not update_response:
            return {
                "updated_scene_info": self.scene_info,
                "end_conversation": False,
                "reason_for_end": ""
            }
        
        # Extract domain and scene_name from response
        domain = update_response.get('domain', self.scene_info.get('domain', ''))
        scene_name = update_response.get('scene_name', self.scene_info.get('scene_name', ''))
        
        # Get odd_protocol data from update response
        odd_protocol_data = update_response.get('odd_protocol', {})
        
        # Initialize a new scene_info structure
        updated_scene_info = {
            "domain": domain,
            "scene_name": scene_name,
            "odd_protocol": {
                "overview": {},
                "design_concepts": {},
                "details": {}
            }
        }
        
        # Update the sections with new data or preserve existing data
        for section in ["overview", "design_concepts", "details"]:
            if section in odd_protocol_data:
                updated_scene_info["odd_protocol"][section] = odd_protocol_data[section]
            else:
                updated_scene_info["odd_protocol"][section] = self.scene_info.get("odd_protocol", {}).get(section, {})
        
        # Update the scene_info
        self.scene_info = updated_scene_info
        
        return {
            "updated_scene_info": self.scene_info,
            "end_conversation": update_response.get('end_conversation', False),
            "reason_for_end": update_response.get('reason_for_end', '')
        }

    def generate_clarification_questions(self) -> dict:
        """
        Generate clarification questions for missing information.
        
        Returns:
            dict: Clarification questions and completeness assessment
        """
        # Convert scene_info to markdown for LLM
        current_markdown = ODDAgent.odd_to_markdown(self.scene_info)
        
        # Prepare the clarification prompt
        clarification_prompt = '''You are an expert at identifying missing or unclear information in a Multi-Agent System ODD protocol. 

Input: {existing_protocol}

Task: Comprehensively assess the completeness and clarity of the Multi-Agent System ODD protocol.

Output Format (JSON):
{{
    "is_complete": boolean,
    "missing_info_points": [
        "Specific missing information details"
    ],
    "question": "A single, user-friendly, conversational question that synthesizes the most critical clarification needed for the protocol"
}}

Evaluation Criteria:
1. Thoroughly examine the entire ODD protocol
2. Determine if all critical components of multi-agent system design are adequately addressed
3. If any significant information is missing, formulate a clear, concise, and conversational question
4. Prioritize the question that will provide the most comprehensive insight into the system's design, interactions, and operational principles

Key Areas to Assess:
- Agent characteristics and behaviors
- Interaction mechanisms
- Environment and context
- Decision-making processes
- Emergent system properties
- Potential limitations or constraints
'''
        clarification_prompt = clarification_prompt.format(
            existing_protocol=current_markdown
        )
        
        # Call LLM to identify missing information
        clarification_result = self.call_llm(clarification_prompt, is_json=True)
        
        # Return questions if generated, otherwise empty dict
        return clarification_result if clarification_result else {}

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input by updating the scene_info and checking for missing information.
        
        Args:
            user_input (str): User's description or response
        
        Returns:
            Dict[str, Any]: Response containing domain, scene_name, odd_protocol and clarification questions
        """
        # Update the scene_info
        response = self.update_scene_info(user_input)
        
        if response.get('end_conversation', False):
            return {
                "domain": self.scene_info.get("domain", ""),
                "scene_name": self.scene_info.get("scene_name", ""),
                "odd_protocol": self.scene_info.get("odd_protocol", {}),
                "clarification_question": '',
                "is_complete": True
            }
    
        # Generate clarification questions
        clarification_result = self.generate_clarification_questions()
        return {
            "domain": self.scene_info.get("domain", ""),
            "scene_name": self.scene_info.get("scene_name", ""),
            "odd_protocol": self.scene_info.get("odd_protocol", {}),
            "clarification_question": clarification_result.get('question', ''),
            "is_complete": clarification_result.get('is_complete', False)
        }

    def get_final_odd_protocol(self) -> Dict:
        """
        Retrieve the ODD protocol part of scene_info.
        
        Returns:
            Dict: The complete scene_info containing ODD protocol
        """
        return self.scene_info

    def save_scene_info(self, env_path: str) -> None:
        """
        Save the scene_info (including ODD protocol) to the scene_info.json file.
        
        Args:
            env_path (str): Path to the environment directory
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
        existing_scene_info["domain"] = self.scene_info["domain"]
        existing_scene_info["scene_name"] = env_path.split(os.sep)[-1] if env_path else self.scene_info["scene_name"]
        
        # Then update odd_protocol
        existing_scene_info["odd_protocol"] = self.scene_info["odd_protocol"]
        
        # Write back to file
        try:
            with open(scene_info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_scene_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Scene info (including ODD protocol) saved to {scene_info_path}")
        except Exception as e:
            logger.error(f"Error saving scene_info to {scene_info_path}: {e}")
    
    
    def load_scene_info(self, env_path: str) -> bool:
        """
        Load the scene_info (including ODD protocol) from the scene_info.json file.
        
        Args:
            env_path (str): Path to the environment directory
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        scene_info_path = os.path.join(env_path, "scene_info.json")
        
        if not os.path.exists(scene_info_path):
            logger.warning(f"scene_info.json not found at {scene_info_path}")
            return False
        
        try:
            with open(scene_info_path, 'r', encoding='utf-8') as f:
                loaded_scene_info = json.load(f)
            
            # Update our scene_info with the loaded data
            if "domain" in loaded_scene_info:
                self.scene_info["domain"] = loaded_scene_info["domain"]
                
            if "scene_name" in loaded_scene_info:
                self.scene_info["scene_name"] = loaded_scene_info["scene_name"]
            
            if "odd_protocol" in loaded_scene_info:
                self.scene_info["odd_protocol"] = loaded_scene_info["odd_protocol"]
                logger.info(f"Scene info loaded from {scene_info_path}")
                return True
            else:
                logger.warning(f"No odd_protocol found in {scene_info_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading scene_info from {scene_info_path}: {e}")
            return False
    

    def reset(self) -> None:
        """
        Reset the scene_info to its initial state.
        """
        self.scene_info = {
            "domain": "",
            "scene_name": "",
            "odd_protocol": {
                "overview": {},
                "design_concepts": {},
                "details": {}
            }
        }