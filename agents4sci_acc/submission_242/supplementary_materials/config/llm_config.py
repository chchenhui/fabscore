"""
LLM Configuration for SimulEval++

This module handles configuration for different LLM providers:
- Standard OpenAI API
- Custom proxy endpoints
- Future: Other providers (Anthropic, local models, etc.)
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

# 🎯 GLOBAL MODEL CONFIGURATION - Change this to update the model everywhere
DEFAULT_MODEL = "gpt-4o-mini"

@dataclass
class LLMConfig:
    """LLM configuration settings"""
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    provider: str = "openai"
    temperature: float = 0.7
    max_retries: int = 3

class LLMConfigManager:
    """Manages LLM configuration for different environments"""
    
    @staticmethod
    def get_config() -> LLMConfig:
        """
        Get LLM configuration based on available environment variables and config files.
        Priority order:
        1. Other proxy configuration (if available)
        2. Standard OpenAI configuration
        3. Error if neither is available
        """
        
        # Try other proxy configuration first
        other_config = LLMConfigManager._get_other_config()
        if other_config:
            return other_config
        
        # Fall back to standard OpenAI
        openai_config = LLMConfigManager._get_openai_config()
        if openai_config:
            return openai_config
        
        # No valid configuration found
        raise ValueError(
            "No valid LLM configuration found. Please set either:\n"
            "1. OPENAI_API_KEY environment variable, or\n"
            "2. OTHER_API_KEY environment variable, or\n"
            "3. Create config/private_config.py with your settings"
        )
    
    @staticmethod
    def _get_other_config() -> Optional[LLMConfig]:
        """Get other proxy configuration"""
        
        # Check for environment variable and config file
        other_key = os.getenv("OTHER_API_KEY")
        
        try:
            from config.private_config import OTHER_CONFIG
            # If env var is set, use it; otherwise use config file key
            if other_key:
                config = OTHER_CONFIG.copy()
                config["api_key"] = other_key
                return LLMConfig(**config)
            else:
                return LLMConfig(**OTHER_CONFIG)
        except ImportError:
            # Fallback if no config file exists
            if other_key:
                return LLMConfig(
                    api_key=other_key,
                    model=DEFAULT_MODEL,
                    base_url=None,  # Use standard OpenAI endpoint as fallback
                    provider="openai"
                )
        
        return None
    
    @staticmethod
    def _get_openai_config() -> Optional[LLMConfig]:
        """Get standard OpenAI configuration"""
        
        # Check environment variable
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return LLMConfig(
                api_key=openai_key,
                model=DEFAULT_MODEL,
                provider="openai"
            )
        
        # Check private config file
        try:
            from config.private_config import OPENAI_CONFIG
            return LLMConfig(**OPENAI_CONFIG)
        except ImportError:
            pass
        
        return None
    
    @staticmethod
    def print_config_info():
        """Print information about current configuration"""
        try:
            config = LLMConfigManager.get_config()
            print(f"✅ LLM Configuration Found:")
            print(f"   Provider: {config.provider}")
            print(f"   Model: {config.model}")
            if config.base_url:
                print(f"   Base URL: {config.base_url}")
            print(f"   API Key: {'*' * (len(config.api_key) - 4) + config.api_key[-4:] if len(config.api_key) > 4 else '****'}")
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")

def create_openai_client(config: LLMConfig):
    """Create OpenAI client with the given configuration"""
    from openai import OpenAI
    
    if config.base_url:
        return OpenAI(api_key=config.api_key, base_url=config.base_url)
    else:
        return OpenAI(api_key=config.api_key)
