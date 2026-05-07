"""
Base agent class for experimental design framework.

This module provides the base class for all agents in the experimental design framework.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from onesim.models import get_model, SystemMessage, UserMessage


class AgentBase:
    """
    Base class for all agents in the experimental design framework.
    
    This class provides common functionality for all agent types,
    including model access, message handling, and basic utility methods.
    """
    
    def __init__(self, name: str, model_name: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            name (str): The name of the agent.
            model_config (str, optional): The model configuration name to use.
                If None, the default model will be used.
        """
        self.name = name
        self.model = get_model(config_name=model_name)
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response from the language model.
        
        Args:
            system_prompt (str): The system prompt to use.
            user_prompt (str): The user prompt to use.
            
        Returns:
            str: The generated response.
        """
        response = self.model(self.model.format(
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ))
        
        return response.text
    
    def save_to_file(self, content: str, file_path: str) -> None:
        """
        Save content to a file.
        
        Args:
            content (str): The content to save.
            file_path (str): The path to the file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and produce output data.
        
        This method should be overridden by subclasses.
        
        Args:
            input_data (Dict[str, Any]): Input data for the agent.
            
        Returns:
            Dict[str, Any]: Output data from the agent.
        """
        raise NotImplementedError("Subclasses must implement process method.") 

if __name__ == "__main__":
    print("This is the base agent module. It should not be run directly.")
