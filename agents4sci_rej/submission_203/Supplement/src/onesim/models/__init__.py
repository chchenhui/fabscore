"""
Models package for OneSim.

This package provides a unified interface for interacting with various language models,
with support for different model providers, response parsing, and configuration management.
"""

from loguru import logger

# Core classes
from .core.model_response import ModelResponse
from .core.message import Message, SystemMessage, UserMessage, AssistantMessage
from .core.model_base import ModelAdapterBase
from .core.model_manager import ModelManager
from .core.load_balancer import LoadBalancer, LoadBalancerStrategy, RoundRobinStrategy, RandomStrategy

# Model implementations
from .providers.openai import OpenAIChatAdapter, OpenAIEmbeddingAdapter

# Parser implementations
from .parsers import JsonBlockParser, JsonDictParser, CodeBlockParser, TagParser, MultiTagParser

# Token usage utilities
from .utils.token_usage import (
    get_token_usage_stats, reset_token_stats, 
    estimate_token_cost, export_token_usage_stats,
    log_token_usage
)

# Utility functions
def get_model_manager() -> ModelManager:
    """
    Get the singleton instance of the model manager.
    
    Returns:
        ModelManager: The model manager instance.
    """
    return ModelManager.get_instance()


def get_model(config_name: str = None, model_name: str = None,model_type: str = None) -> ModelAdapterBase:
    """
    Get a model instance by configuration name or selection criteria.
    
    This is a convenience function that delegates to the model manager.
    Models can be accessed in several ways:
    1. By specific config_name - returns the exact model configuration
    2. By model_name - returns a load balancer for all models with the given name
    3. By model_type - returns a load balancer for all models of that type
    4. With no parameters - returns the default load balancer (all chat models)
    
    Args:
        config_name: The name of the model configuration.
        model_type: Type of models to use ('chat', 'embedding', or specific provider type).
        model_name: Name of the model (e.g., 'gpt-4') to load balance across providers.
        
    Returns:
        ModelAdapterBase: An initialized model adapter.
    """
    return get_model_manager().get_model(config_name, model_name, model_type)


def get_embedding_model(config_name: str = None) -> ModelAdapterBase:
    """
    Get an embedding model instance by configuration name.
    
    This is a convenience function that delegates to the model manager.
    If no config_name is provided, returns the embedding load balancer if configured,
    otherwise falls back to a specific embedding model.
    
    Args:
        config_name: The name of the model configuration. If None, returns the embedding load balancer.
        
    Returns:
        ModelAdapterBase: An initialized model adapter.
    """
    manager = get_model_manager()
    
    if config_name is None:
        # Use embedding load balancer if available
        return manager.get_model(model_type="embedding")
    
    return manager.get_model(config_name)


def configure_load_balancer(model_configs=None, strategy="round_robin", config_name="chat_load_balancer", model_type="chat"):
    """
    Configure the load balancer for the model manager.
    
    Args:
        model_configs: List of model configuration names to load balance between.
                      If None, will use all available models of the specified type.
        strategy: Load balancing strategy to use ('round_robin' or 'random').
        config_name: Configuration name for the load balancer.
        model_type: Type of models to balance ('chat' or 'embedding').
    """
    get_model_manager().configure_load_balancer(
        model_configs=model_configs,
        strategy=strategy,
        config_name=config_name,
        model_type=model_type
    )


__all__ = [
    # Core classes
    'ModelResponse',
    'Message', 'SystemMessage', 'UserMessage', 'AssistantMessage',
    'ModelAdapterBase',
    'ModelManager',
    
    # Load balancer classes
    'LoadBalancer',
    'LoadBalancerStrategy',
    'RoundRobinStrategy',
    'RandomStrategy',
    
    # Model implementations
    'OpenAIChatAdapter',
    'OpenAIEmbeddingAdapter',
    
    # Parser implementations
    'JsonDictParser', 'CodeBlockParser', 'JsonBlockParser', 'TagParser', 'MultiTagParser',
    
    # Token usage utilities
    'get_token_usage_stats',
    'reset_token_stats',
    'estimate_token_cost',
    'export_token_usage_stats',
    'log_token_usage',
    
    # Utility functions
    'get_model_manager',
    'get_model',
    'get_embedding_model',
    'configure_load_balancer',
] 