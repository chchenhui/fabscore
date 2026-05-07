# -*- coding: utf-8 -*-
"""
Debugging Agent Configuration
Configuration classes and templates for the debugging agent.
"""

import os
from datetime import datetime
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from .debugging_data_structures import DebuggingResult


@dataclass
class DebuggingConfig:
    """Configuration class for the Debugging Agent."""
    
    # Docker-related configuration
    container_name: str = "onesim-backend"
    host_project_path: str = field(default_factory=lambda: os.path.abspath(os.getcwd()))
    container_project_path: str = "/app"
    docker_image: str = "ptss/yulan-onesim-backend:latest"
    
    # Iteration-related configuration
    max_iterations: int = 5
    timeout_per_iteration: int = 600  # 5 minutes
    
    # LLM-related configuration
    model_config_name: str = "openai-gpt4o"
    confidence_threshold: float = 0.3  # Confidence threshold for applying fixes
    
    # History configuration
    save_history: bool = True
    history_dir: Optional[str] = None  # Will be set dynamically per environment
    
    # Error handling configuration
    max_consecutive_failures: int = 3
    retry_on_timeout: bool = True
    
    # File backup configuration
    create_backups: bool = True
    backup_dir: Optional[str] = None  # Will be set dynamically per environment
    
    # Output configuration
    verbose_logging: bool = True
    log_level: str = "INFO"
    
    # Advanced configuration
    parallel_fixes: bool = False  # Whether to process fixes for multiple files in parallel
    auto_apply_high_confidence: bool = True  # Automatically apply high-confidence fixes
    
    # Debug logging configuration
    detailed_logging: bool = True  # Enable detailed step-by-step logging
    log_llm_calls: bool = True  # Log LLM inputs and outputs
    debug_log_file: Optional[str] = None  # Specific debug log file path
    
    # Global experience configuration
    enable_global_experience: bool = True  # Whether to enable global experience repository
    global_experience_storage_path: Optional[str] = None  # Global experience repository storage path
    
    # ReAct agent configuration
    enable_react_debugging: bool = True  # Enable ReAct debugging mode
    max_react_iterations: int = 10  # Maximum ReAct iterations per round
    
    # Enhanced context configuration
    include_environment_files: bool = True  # Include environment file list
    include_metric_definitions: bool = True  # Include metric definitions
    
    # Error tracking enhancement
    track_error_sources: bool = True  # Track real error sources
    analyze_stack_trace: bool = True  # Analyze complete stack trace
    
    def get_env_specific_history_dir(self, env_name: str) -> str:
        """Get environment-specific history directory."""
        if self.history_dir:
            return self.history_dir
        env_path = Path(self.host_project_path) / "src" / "envs" / env_name / "log" / "debug_history"
        return str(env_path)
    
    def get_env_specific_backup_dir(self, env_name: str) -> str:
        """Get environment-specific backup directory."""
        if self.backup_dir:
            return self.backup_dir
        env_path = Path(self.host_project_path) / "src" / "envs" / env_name / "log" / "code_backups"
        return str(env_path)
    
    def get_debug_log_file(self, env_name: str) -> str:
        """Get debug log file path."""
        if self.debug_log_file:
            return self.debug_log_file
        env_path = Path(self.host_project_path) / "src" / "envs" / env_name / "log"
        env_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(env_path / f"debug_agent_detailed_{timestamp}.log")
    
    def get_global_experience_storage_path(self) -> str:
        """Get global experience storage path."""
        if self.global_experience_storage_path:
            return self.global_experience_storage_path
        return str(Path(self.host_project_path) / "global_debug_patterns")


# Predefined configuration templates
DEFAULT_CONFIG = DebuggingConfig()

QUICK_DEBUG_CONFIG = DebuggingConfig(
    max_iterations=3,
    timeout_per_iteration=180,  # 3 minutes
    confidence_threshold=0.5,   # Higher confidence requirement
    save_history=False,
    create_backups=False  # Skip backups for quick debugging
)

THOROUGH_DEBUG_CONFIG = DebuggingConfig(
    max_iterations=10,
    timeout_per_iteration=600,  # 10 minutes
    confidence_threshold=0.2,   # Lower confidence requirement to try more fixes
    max_consecutive_failures=5,
    parallel_fixes=True
)


def get_config_template(template_name: str) -> DebuggingConfig:
    """
    Gets a predefined configuration template.
    
    Args:
        template_name: The name of the template ('default', 'quick', 'thorough').
        
    Returns:
        DebuggingConfig: The configuration object.
    """
    templates = {
        'default': DEFAULT_CONFIG,
        'quick': QUICK_DEBUG_CONFIG,
        'thorough': THOROUGH_DEBUG_CONFIG
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown configuration template: {template_name}. Available templates: {list(templates.keys())}")
    
    return templates[template_name]


def create_custom_config(**kwargs) -> DebuggingConfig:
    """
    Creates a custom configuration.
    
    Args:
        **kwargs: Configuration parameters.
        
    Returns:
        DebuggingConfig: The custom configuration object.
    """
    return DebuggingConfig(**kwargs)


async def debug_environment_quick(env_name: str,
                                model_config_name: str,
                                max_iterations: int = 3) -> DebuggingResult:
    """
    Convenience function to quickly debug an environment.
    
    Args:
        env_name: The name of the environment.
        model_config_name: The name of the model configuration.
        max_iterations: The maximum number of iterations.
        
    Returns:
        A DebuggingResult with the debugging results.
    """
    # Import here to avoid circular imports
    from onesim.agent.debugging_agent import DebuggingAgent
    
    config = create_custom_config(
        max_iterations=max_iterations,
        timeout_per_iteration=180
    )
    
    debugger = DebuggingAgent(
        model_config_name=model_config_name,
        config=config
    )
    
    return await debugger.auto_debug(env_name)