"""Configuration management for literature query system"""

from .literature_config import LiteratureConfig, get_config, load_config

__all__ = [
    "LiteratureConfig",
    "get_config", 
    "load_config"
]